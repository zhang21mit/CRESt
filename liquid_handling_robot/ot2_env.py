import json
from time import sleep
from abc import ABC, abstractmethod
import pandas as pd
import paramiko
from paramiko.ssh_exception import NoValidConnectionsError
# from paramiko import SSHClient
from scp import SCPClient
from software_control.software_control import OBS_Linux, OBS_Win
from utils.utils import config_loader, get_project_path, email, hyperlapse_video, get_latest_file_name


class OT2_Env(ABC):
    def __init__(self, exp_name, connect_ot2=True, wireless=False):
        self.exp_name = exp_name
        self.reservoir_index = [f'{i}{j}' for i in ['A', 'B', 'C'] for j in range(1, 7)]
        self.parent_path = f'{get_project_path()}/liquid_handling_robot'
        self.status_path = f'{self.parent_path}/ot2_status'
        self.key_path = f'{self.status_path}/ot2_ssh_key'
        self.tip_tracker_path_20ul = f'{self.status_path}/tip_tracker_20ul.json'
        self.tip_tracker_path_300ul = f'{self.status_path}/tip_tracker_300ul.json'
        self.mixing_well_tracker_path = f'{self.status_path}/mixing_well_tracker.json'
        self.task_path = f'{self.status_path}/task.json'
        self.benchmark_path = f'{self.status_path}/benchmark.json'
        self.deck_configs_path = f'{self.status_path}/deck_configs.json'
        self.run_configs_path = f'{self.status_path}/run_configs.json'
        self.run_path = f'{self.parent_path}/ot2_run.py'
        self.wireless = wireless
        self.robot_ip = self.robot_ip_detect()

    def robot_ip_detect(self):
        connection_method = 'wireless' if self.wireless else 'wired'
        for ip in self.ip_dict[connection_method]:
            try:
                client = paramiko.SSHClient()
                client.set_missing_host_key_policy(paramiko.client.WarningPolicy)
                client.connect(hostname=ip, username='root', key_filename=self.key_path)
            except NoValidConnectionsError:
                continue
            print(f'OT2 detected via {connection_method} connection at {ip}!')
            return ip
        raise SystemError(f'No OT2 detected via {connection_method} connection!')

    def file_transfer(self, local_path, ot2_path, to_ot2=True):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.client.WarningPolicy)
        client.connect(hostname=self.robot_ip, username='root', key_filename=self.key_path)
        scp = SCPClient(client.get_transport())

        if to_ot2 is True:
            scp.put(local_path, remote_path=f'/data/user_storage/{ot2_path}')
            # remove ^M which is generated from Win
            stdin, stdout, stderr = client.exec_command(f'dos2unix /data/user_storage/{ot2_path}/*')
        else:
            scp.get(f'/data/user_storage/{ot2_path}', local_path=local_path)

    def execute(self, cmd_str):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.client.WarningPolicy)
        client.connect(hostname=self.robot_ip, username='root', key_filename=self.key_path)
        stdin, stdout, stderr = client.exec_command(cmd_str)
        ret_msg = {
            'stdout': list(stdout.readlines()),
            'stderr': list(stderr.readlines())
        }
        return ret_msg

    def execute_protocol(self, file_path, cmd):
        ret_msg = self.execute(f'export RUNNING_ON_PI=1; {cmd} /data/user_storage/protocols/{file_path}')
        return ret_msg

    def update_run_config(self, simulate):
        run_config = config_loader(self.exp_name, 'ot2_run_configs')
        run_config['exp_name'] = self.exp_name
        run_config['simulate'] = simulate
        json.dump(json.dumps(run_config), open(self.run_configs_path, "w"))
        self.file_transfer(self.run_configs_path, 'tasks/')

    def run_after_simulation(self, file_path, simulate, run):
        if simulate:
            self.update_run_config(simulate=True)
            ret_msg = self.execute_protocol(file_path, 'opentrons_simulate')
            if len(ret_msg['stderr']) == 0 or all('Loading defaults' in line for line in ret_msg['stderr']):
                for line in ret_msg['stdout']:
                    print(line.replace('\n', ''))
                print('\n***********task simulation passed***********\n')
                if run:
                    print('starting experiment...')
                    self.update_run_config(simulate=False)
                    ret_msg = self.execute_protocol(file_path, 'opentrons_execute')
                    for line in ret_msg['stdout'] + ret_msg['stderr']:
                        print(line.replace('\n', ''))
            else:
                for line in ret_msg['stderr']:
                    print(line.replace('\n', ''))
                print('\n***********task simulation failed***********\n')
        else:
            print('starting experiment...')
            self.update_run_config(simulate=False)
            ret_msg = self.execute_protocol(file_path, 'opentrons_execute')
            for line in ret_msg['stdout'] + ret_msg['stderr']:
                print(line.replace('\n', ''))

    def stop_ot2_server(self):
        self.execute(f'export RUNNING_ON_PI=1; systemctl stop opentrons-robot-server')

    def start_ot2_server(self):
        self.execute(f'export RUNNING_ON_PI=1; systemctl start opentrons-robot-server')

    # generate tasks and transfer the file to ot2
    def create_tasks(self, task_df: pd.DataFrame, benchmark: dict = None):
        task_df.to_json(self.task_path)
        self.file_transfer(local_path=self.task_path, ot2_path='tasks/')

        benchmark = benchmark or {}
        benchmark_df_to_dict = {k: v.to_dict(orient='records') for k, v in benchmark.items()}
        json.dump(json.dumps(benchmark_df_to_dict), open(self.benchmark_path, "w"))
        self.file_transfer(local_path=self.benchmark_path, ot2_path='tasks/')

    # method only for 15ml Falcon centrifuge tube
    @staticmethod
    def reservoir_vol_to_height(vol, tube_type):
        if tube_type == '15ml':
            two_ml_to_bottom = 25
            each_ml_length = 6.5
            immersing_offset = 10
            height = two_ml_to_bottom + (vol - 2) * each_ml_length - immersing_offset
        else:
            four_ml_to_bottom = 15
            each_ml_length = 1.9
            immersing_offset = 8
            height = four_ml_to_bottom + (vol - 4) * each_ml_length - immersing_offset
        return height

    # reservoir update
    def update_reservoir(self, reservoir_update_dict, tube_type='15ml'):
        assert tube_type in ['15ml', '50ml'], 'reservoir type can be 15ml or 50ml'
        self.file_transfer(
            local_path=f'{self.parent_path}/ot2_status/',
            ot2_path=f'ot2_status/{self.exp_name}.json',
            to_ot2=False,
        )
        reservoir_path = f'{self.status_path}/{self.exp_name}.json'
        heights_cur = json.loads(json.load(open(reservoir_path)))

        heights_new = heights_cur.copy()

        for reservoir_name, vol_update_dict in reservoir_update_dict.items():
            for coord, vol in vol_update_dict.items():
                assert coord in self.reservoir_index, 'coord must be from A1 to C6'
                if vol == 0:
                    heights_new[reservoir_name][coord] = 0
                else:
                    heights_new[reservoir_name][coord] = self.reservoir_vol_to_height(vol, tube_type)
        json.dump(json.dumps(heights_new), open(reservoir_path, "w"))
        self.file_transfer(local_path=reservoir_path, ot2_path='ot2_status/')
        print(f'{self.exp_name} reservoir successfully updated')

    # reservoir reset
    def reset_reservoir(self, reservoir_name_list):
        reservoir_path = f'{self.status_path}/{self.exp_name}.json'
        heights_new = dict(
            zip(reservoir_name_list, [dict(
                zip(self.reservoir_index, [0 for i in range(len(self.reservoir_index))])
            ) for j in range(len(reservoir_name_list))])
        )
        json.dump(json.dumps(heights_new), open(reservoir_path, "w"))
        self.file_transfer(local_path=reservoir_path, ot2_path='ot2_status/')
        print(f'{self.exp_name} reservoir successfully reset')

    # tip index reset, index equals to the number of tips already used
    def reset_tip_index(self, tip_type=None, tip_index=0):
        assert tip_type in ['20ul', '300ul'], 'tip type must be 20ul or 300ul'
        if tip_type == '20ul':
            path = self.tip_tracker_path_20ul
        else:
            path = self.tip_tracker_path_300ul
        json.dump(json.dumps(tip_index), open(path, "w"))
        self.file_transfer(local_path=path, ot2_path='ot2_status/')
        print(f'{tip_type} tip index successfully set to {tip_index}')

    # mixing well index reset, index equals to the number of rows of well already used
    def reset_mixing_well_index(self, mixing_well_index=0):
        json.dump(json.dumps(mixing_well_index), open(self.mixing_well_tracker_path, "w"))
        self.file_transfer(local_path=self.mixing_well_tracker_path, ot2_path='ot2_status/')
        print(f'mixing well index successfully set to {mixing_well_index}')

    def transfer_protocol(self):
        self.file_transfer(local_path=self.run_path, ot2_path='protocols/')
        deck_configs = config_loader(self.exp_name, 'ot2_deck_configs')
        json.dump(json.dumps(deck_configs), open(self.deck_configs_path, "w"))
        self.file_transfer(local_path=self.deck_configs_path, ot2_path='protocols/')

    def run_protocol(self, simulate, run):
        self.stop_ot2_server()
        self.run_after_simulation('ot2_run.py', simulate, run)
        self.start_ot2_server()

    def email_video(self, task_df, operator_name):
        email_address = config_loader(self.exp_name, 'robot_test_configs')['email'][operator_name]
        content = f'Hi {operator_name},\n\n' \
                  f'The sample preparation job of project {self.exp_name} is finished!\n\n' \
                  f'{task_df.shape[0]} samples are successfully prepared:\n\n' \
                  f'{task_df.to_string(index=False)}\n\n'
        raw_path = '/media/li/HDD/ot2_recordings/raw'
        hyperlapse_path = '/media/li/HDD/ot2_recordings/hyperlapse'
        file_name = get_latest_file_name(raw_path)
        sleep(120)
        hyperlapse_video_path = hyperlapse_video(file_name, raw_path, hyperlapse_path)
        email(email_address, '[Success] sample preparation recording', [content, hyperlapse_video_path])

    @abstractmethod
    def run(self, tasks: pd.DataFrame, benchmark: dict = None, simulate=True, run=True, operator_name=None,
            recording=True):
        pass


class OT2_Env_Linux(OT2_Env):
    def __init__(self, *args, **kwargs):
        self.ip_dict = {
            'wired': [
                '169.254.252.200',
                '169.254.177.202',
                '169.254.55.62',
                '169.254.46.151',
                '169.254.48.145',
                '169.254.72.67',
                '169.254.128.109',
                '169.254.70.46',
                '169.254.117.217',
            ],
            'wireless': [
                '192.168.1.112',
            ],
        }
        super().__init__(*args, **kwargs)

    def run(self, tasks: pd.DataFrame, benchmark: dict = None, simulate=True, run=True, operator_name=None,
            recording=True):
        self.create_tasks(tasks, benchmark)
        self.transfer_protocol()
        obs = OBS_Linux(
            exp_name=self.exp_name,
        ) if recording else None
        obs.start() if recording else None
        self.run_protocol(simulate, run)
        obs.stop() if recording else None
        if operator_name:
            self.email_video(tasks, operator_name)


class OT2_Env_Win(OT2_Env):
    def __init__(self, *args, **kwargs):
        self.ip_dict = {
            'wired': [
                '169.254.134.216',
            ],
            'wireless': [
                '192.168.0.247',
            ],
        }
        super().__init__(*args, **kwargs)

    def run(self, tasks: pd.DataFrame, benchmark: dict = None, simulate=True, run=True, operator_name=None,
            recording=True):
        self.create_tasks(tasks, benchmark)
        self.transfer_protocol()
        obs = OBS_Win(
            exp_name=self.exp_name,
        ) if recording else None
        obs.start() if recording else None
        self.run_protocol(simulate, run)
        obs.stop() if recording else None
        if operator_name:
            self.email_video(tasks, operator_name)
