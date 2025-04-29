import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from functools import wraps
from time import sleep
from typing import List, Callable
import pandas as pd
from software_control.software_control import EC_Lab, CameraControl
from db_control.database import Database
from robotic_testing.common.robotic_arms.xarm_control import xArm
from utils.utils import log_and_print, config_loader, get_dir, email, get_log_path, get_logger
import pyautogui


class RobotTest(ABC):
    def __init__(
            self,
            exp_name: str,
            ec_lab: EC_Lab,
            camera: CameraControl,
            arm: xArm,
            db: Database = None,
            to_log=True,
            prompt_window_bypass_list: List[Callable[[], None]] = None,
            *args,
            **kwargs
    ):
        self.exp_name = exp_name
        self.configs = config_loader(exp_name, config_type='robot_test_configs')
        self.to_log = to_log
        self.ec_lab = ec_lab
        self.camera = camera
        self.db = db
        self.arm = arm
        self.prompt_window_bypass_list = prompt_window_bypass_list

        # logger settings
        self.logger = get_logger(exp_name=exp_name, module_name=self.__class__.__name__) if self.to_log else None

    def log_errors(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                self.logger.exception(e)
                raise

        return wrapper

    @staticmethod
    def get_cur_time(time_format='log'):
        assert time_format in ['log', 'file_name', 'sql']
        cur_time = datetime.now()
        if time_format == 'log':
            return cur_time.strftime('%Y-%m-%d %H:%M:%S')
        elif time_format == 'file_name':
            return cur_time.strftime('%Y-%m-%d_%H-%M-%S')
        else:
            return cur_time.replace(microsecond=0)

    def get_ec_lab_file_name(self, test_name, technique_id, channel_id):
        technique = self.configs['protocol'][technique_id]
        if int(technique_id) < 10:
            technique_id = f'0{technique_id}'
        return f'{self.configs["data_dir"]}/{test_name}_{technique_id}_{technique}_C0{channel_id}'

    def analyze_test_results(self, sample_id, test_name, channel_id, **kwargs):
        data_dict = {'sample_id': [sample_id]}
        analysis_config = self.configs['data_analysis']
        for metric_name, metric_info in analysis_config['metrics'].items():
            technique_id = metric_info['technique_id']
            technique = self.configs['protocol'][technique_id]
            file_name = self.get_ec_lab_file_name(test_name, technique_id, channel_id)
            sample_area = self.configs['sample_area']
            analyzer = analysis_config['analyzer']['analyzer_class'](
                file_name,
                technique=technique,
                sample_area=sample_area,
                **{'kwargs': {}, **analysis_config['analyzer']}.get('kwargs')
            )
            analyze_result = metric_info['analyze_method'](analyzer, **{'kwargs': {}, **metric_info}.get('kwargs'))
            data_dict[metric_name] = [analyze_result]
        data_df = pd.DataFrame(data_dict)
        return data_df

    def run_initialization(
            self,
            sequence,
            sample_id_list,
            sample_rack_id_starting,
            benchmark
    ):
        """
        :param sequence: True if the tested sample index is in a continuous sequence, False if not in sequence
        :param sample_id_list: If sequence is True, input sample starting and ending index, e.g. [3, 10] for sample #3
        to #10. If sequence is False, input a list in the exact order of the sample placed on the sample rack,
        e.g. [3, 5, 2, 10] for sample #3, #5, #2, #10 respectively.
        :param sample_rack_id_starting: the starting id on sample rack, starting with 0
        :param benchmark: the benchmark sample id, if provided, the robot will test the benchmark sample first, format in {sample_rack_id: sample_id}
        """
        sample_id_list = [*range(sample_id_list[0], sample_id_list[1] + 1)] if sequence else sample_id_list
        task_len = len(sample_id_list)
        sample_rack_id_list = [*range(sample_rack_id_starting, sample_rack_id_starting + task_len)]
        sample_rack_id_to_sample_id_dict = dict(zip(sample_rack_id_list, sample_id_list))
        if benchmark:
            # check the sample_rack_id in benchmark does not conflict with the sample_rack_id in the task
            assert not set(benchmark.keys()).intersection(
                set(sample_rack_id_to_sample_id_dict.keys())), 'sample_rack_id in benchmark conflicts with the sample_rack_id in the task'
            # add benchmark to the task
            sample_rack_id_to_sample_id_dict = {**benchmark, **sample_rack_id_to_sample_id_dict}
        msg = f'started a batch of test, with sample_rack_id : sample_id = ' \
              f'{sample_rack_id_to_sample_id_dict}'
        log_and_print(self.to_log, self.logger, 'info', msg)
        return sample_rack_id_to_sample_id_dict

    def get_test_name(self, sample_id):
        return f'{self.exp_name}_sample#{sample_id}_{self.get_cur_time("file_name")}'

    @abstractmethod
    def test_main_workflow(self, *args, **kwargs):
        pass

    @abstractmethod
    def robot_warming_up(self):
        pass

    @abstractmethod
    def robot_returning_home(self):
        pass

    @abstractmethod
    def check_test_finished(self):
        pass

    def data_analyze(self, sample_id, test_name, channel_id):
        assert isinstance(self.db, Database), 'database must be initialized to use data analysis'
        analyze_result = self.analyze_test_results(sample_id, test_name, channel_id)
        self.db.add_data_to_server(self.db.configs['tables_dict']['active_learning.performance'], analyze_result)

    @staticmethod
    def mouse_moved(time_frame=10):
        x, y = pyautogui.position()
        sleep(time_frame)
        # check if human is operating PC, any cursor movement during 10s
        if (x, y) == pyautogui.position():
            return False
        else:
            return True


    def bypass_prompt_window(self):
        for bypass_func in self.prompt_window_bypass_list:
            bypass_func()

    def send_email(self, operator_name, sample_rack_id_to_sample_id_dict):
        email_address = self.configs['email'][operator_name]
        task_history = "\n".join(f"{k} : {v}" for k, v in sample_rack_id_to_sample_id_dict.items())
        content = f"{self.exp_name} finished with samples tested as follows:\n\n" \
                  f"{task_history}\n\n" \
                  f"Your research assistant,\n" \
                  f"Crest"
        email(email_address, f'[success] {self.exp_name} robotic testing finished', content)

    @log_errors
    def run(
            self,
            sequence,
            sample_id_list,
            sample_rack_id_starting,
            immerse_option='flask_contact_immersed',
            data_analysis=False,
            benchmark=None,
            operator_name=None,
    ):

        sample_rack_id_to_sample_id_dict = self.run_initialization(
            sequence, sample_id_list, sample_rack_id_starting, benchmark
        )

        self.robot_warming_up()

        for sample_rack_id, sample_id in sample_rack_id_to_sample_id_dict.items():
            # get test name
            test_name = self.get_test_name(sample_id)

            # start testing
            self.test_main_workflow(
                sample_rack_id=sample_rack_id,
                test_name=test_name,
                immerse_option=immerse_option,
            )

            # data analysis
            if data_analysis:
                channel_id = self.ec_lab.channel_id
                # give some time for the software to export csv file
                while True:
                    try:
                        sleep(2)
                        self.data_analyze(sample_id, test_name, channel_id)
                        break
                    except FileNotFoundError:
                        self.logger.warning(f'test raw file not found, retrying...')
                        continue

        self.robot_returning_home()

        # send the notification email if operator name is provided
        if operator_name:
            self.send_email(operator_name, sample_rack_id_to_sample_id_dict)


class RobotTest_AlkalinePlatform(RobotTest):

    def test_main_workflow(self, sample_rack_id, test_name, immerse_option, *args, **kwargs):
        # start recording
        self.camera.start()

        # move to mid-station
        self.arm.move_to_mid_station()

        # load sample
        self.logger.info(f'loading the sample from rack_id #{sample_rack_id}...')
        self.arm.load_sample(sample_rack_index=sample_rack_id, immerse_option=immerse_option)
        self.logger.info(f'sample successfully loaded into the cell')

        # stop recording
        self.camera.stop()
        sleep(3)

        # open software and start testing
        self.ec_lab.start(test_name)

        # wait and bypass warning window
        sleep(10)

        # check test finished
        self.check_test_finished()
        self.logger.info(f'test {test_name} finished')

        # start recording
        self.camera.start()

        # unload sample
        self.logger.info(f'unloading the sample to rack_id #{sample_rack_id}...')
        self.arm.unload_sample(sample_rack_index=sample_rack_id)
        self.logger.info(f'sample successfully unloaded from the cell')

        # stop recording
        sleep(3)
        self.camera.stop()

    def robot_warming_up(self):
        self.arm.home()

    def robot_returning_home(self):
        self.arm.home()

    def check_test_finished(self):
        while self.ec_lab.running:
            # check test status if no human movement in 10 seconds
            if not self.mouse_moved():
                # bypass teamviewer window if exist
                self.bypass_prompt_window()
                # open software
                self.ec_lab.open_software()
                # check test status
                if self.ec_lab.test_is_complete():
                    self.ec_lab.running = False
            else:
                self.logger.info(f'human operating, fail to detect...')


class RobotTest_AcidicPlatform(RobotTest):

    def test_main_workflow(self, sample_rack_id, test_name, *args, **kwargs):
        # start recording
        self.camera.start()

        # load sample
        self.logger.info(f'loading the sample from rack_id #{sample_rack_id}...')
        self.arm.load_sample(sample_rack_index=sample_rack_id)
        self.logger.info(f'sample successfully loaded into the cell')

        # stop recording
        self.camera.stop()
        sleep(3)

        # open software and start testing
        self.ec_lab.start(test_name)
        sleep(10)

        # check test finished
        self.check_test_finished()
        self.logger.info(f'test {test_name} finished')

        # start recording
        self.camera.start()

        # unload sample
        self.logger.info(f'unloading the sample to rack_id #{sample_rack_id}...')
        self.arm.unload_sample(sample_rack_index=sample_rack_id)
        self.logger.info(f'sample successfully unloaded from the cell')

        # stop recording
        sleep(3)
        self.camera.stop()

    def robot_warming_up(self):
        self.arm.home()
        self.arm.warm_up()

    def robot_returning_home(self):
        self.arm.warm_up()
        self.arm.home()

    def check_test_finished(self):
        while self.ec_lab.running:
            # check test status if no human movement in 10 seconds
            if not self.mouse_moved():
                # bypass prompt window if exist
                self.bypass_prompt_window()
                # open software
                self.ec_lab.open_software()
                # open channel
                self.ec_lab.open_channel()
                # check test status
                if self.ec_lab.test_is_complete():
                    self.ec_lab.running = False
            else:
                self.logger.info(f'human operating, fail to detect...')


    