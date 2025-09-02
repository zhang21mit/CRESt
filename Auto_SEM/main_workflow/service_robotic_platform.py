from threading import Thread

from flask import Flask, request

from db_control.database import Database
from robotic_testing.common.robot_test import RobotTest_AcidicPlatform
from robotic_testing.common.robotic_arms.xarm6.xarm6 import xArm6
from software_control.software_control import EC_Lab_Win11, Kamoer, Smartlife, LaserPecker, OBS_Win, TeamViewer, ToDesk

app = Flask(__name__)


@app.route("/")
def status_check():
    return "the robotic testing server is running ok", 200


@app.route("/laser_cut", methods=['POST'])
def laser_cut():
    kwargs = request.get_json()
    thread = Thread(target=laser_start, kwargs=kwargs)
    thread.start()
    return f"laser started", 200


def laser_start(exp_name):
    laser = LaserPecker(exp_name=exp_name)
    laser.start()


@app.route("/pump_control", methods=['POST'])
def pump_control():
    exp_name = request.get_json()['exp_name']
    action = request.get_json()['action']
    assert action in ['on', 'off']
    pump = Kamoer(exp_name=exp_name)
    if action == 'on':
        pump.start()
    else:
        pump.stop()
    return f"pump {action}", 200


@app.route("/gas_control", methods=['POST'])
def gas_control():
    exp_name = request.get_json()['exp_name']
    action = request.get_json()['action']
    assert action in ['on', 'off']
    gas = Smartlife(exp_name=exp_name)
    if action == 'on':
        gas.start()
    else:
        gas.stop()
    return f"gas {action}", 200


@app.route("/run_test", methods=['POST'])
def run_test():
    kwargs = request.get_json()
    thread = Thread(target=robot_test, kwargs=kwargs)
    thread.start()
    return 'robot test started!', 200


def robot_test(
        exp_name,
        sample_id_starting,
        sample_id_ending,
        sample_rack_id_starting,
        data_analysis,
        operator_name,
        **kwargs
):
    teamviewer = TeamViewer()
    todesk = ToDesk()
    robot = RobotTest_AcidicPlatform(
        exp_name=exp_name,
        ec_lab=EC_Lab_Win11(channel_id=2),
        camera=OBS_Win(),
        db=Database(exp_name=exp_name),
        arm=xArm6(),
        prompt_window_bypass_list=[
            teamviewer.bypass_side_window,
            teamviewer.bypass_session_end_prompt_window,
            todesk.bypass_side_window,
        ]
    )
    sample_id_list = [sample_id_starting, sample_id_ending]
    run_config = {
        'sequence': True,
        'sample_id_list': sample_id_list,
        'sample_rack_id_starting': sample_rack_id_starting,
        'data_analysis': data_analysis,
        'operator_name': operator_name,
    }
    robot.run(**run_config)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8010)
