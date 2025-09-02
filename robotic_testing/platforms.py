from software_control.software_control import EC_Lab_Win11, EC_Lab_Win10, OBS_Win, Camera_Win10, TeamViewer, ToDesk
from robotic_testing.common.robot_test import RobotTest_AcidicPlatform, RobotTest_AlkalinePlatform
from db_control.database import Database
from robotic_testing.common.robotic_arms.xarm6.xarm6 import xArm6
from robotic_testing.common.robotic_arms.xarm7.xarm7 import xArm7


def init_acidic_platform_1318(exp_name, channel_id=2, connect_db=True):
    teamviewer = TeamViewer(exp_name=exp_name)
    todesk = ToDesk(exp_name=exp_name)
    db = Database(exp_name=exp_name) if connect_db else None
    rt = RobotTest_AcidicPlatform(
        exp_name=exp_name,
        ec_lab=EC_Lab_Win11(exp_name=exp_name, channel_id=channel_id),
        camera=OBS_Win(exp_name=exp_name),
        db=db,
        arm=xArm6(exp_name=exp_name),
        prompt_window_bypass_list=[
            teamviewer.bypass_side_window,
            teamviewer.bypass_session_end_prompt_window,
            todesk.bypass_side_window,
        ]
    )
    return rt


def init_alkaline_platform_270(exp_name, connect_db=True):
    teamviewer = TeamViewer(exp_name=exp_name)
    todesk = ToDesk(exp_name=exp_name)
    db = Database(exp_name=exp_name) if connect_db else None
    rt = RobotTest_AlkalinePlatform(
        exp_name=exp_name,
        ec_lab=EC_Lab_Win10(exp_name=exp_name),
        camera=Camera_Win10(exp_name=exp_name),
        db=db,
        arm=xArm7(exp_name=exp_name),
        prompt_window_bypass_list=[
            teamviewer.bypass_side_window,
            teamviewer.bypass_session_end_prompt_window,
            todesk.bypass_side_window,
        ]
    )
    return rt
