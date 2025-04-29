import time
import traceback
import numpy as np
import importlib
from xarm import version
from xarm.wrapper import XArmAPI


class xArm(XArmAPI):
    def __init__(self, config_path, **kwargs):
        # import config python file, config_path example: 'robotic_testing.xarm7.xarm7_config'
        self.config = importlib.import_module(config_path)

        super().__init__(port=self.config.port, **kwargs)

        self.params = {
            'events': {},
            'variables': {},
            'callback_in_thread': True,
            'quit': False
        }

        self.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
        self.clean_warn()
        self.clean_error()
        self.motion_enable(True)
        self.set_mode(0)
        self.set_state(0)

        # register callback functions
        self.register_error_warn_changed_callback(self.error_warn_change_callback)
        self.register_state_changed_callback(self.state_changed_callback)
        self.register_count_changed_callback(self.count_changed_callback)
        self.register_connect_changed_callback(self.connect_changed_callback)

        # sample in flask
        self.sample_in_flask = False

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1],
                                       ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    # Register error/warn changed callback
    def error_warn_change_callback(self, data):
        if data and data['error_code'] != 0:
            self.params['quit'] = True
            self.pprint('err={}, quit'.format(data['error_code']))
            self.arm.release_error_warn_changed_callback(self.error_warn_change_callback)

    # Register state changed callback
    def state_changed_callback(self, data):
        if data and data['state'] == 4:
            if self.arm.version_number[0] >= 1 and self.arm.version_number[1] >= 1 and self.arm.version_number[2] > 0:
                self.params['quit'] = True
                self.pprint('state=4, quit')
                self.arm.release_state_changed_callback(self.state_changed_callback)

    # Register counter value changed callback
    def count_changed_callback(self, data):
        if not self.params['quit']:
            self.pprint('counter val: {}'.format(data['count']))

    # Register connect changed callback
    def connect_changed_callback(self, data):
        if data and not data['connected']:
            self.params['quit'] = True
            self.pprint('disconnect, connected={}, reported={}, quit'.format(data['connected'], data['reported']))
            self.arm.release_connect_changed_callback(self.error_warn_change_callback)

    def check_error(func):
        def error_checker(self, *args, **kwargs):
            if self.error_code == 0 and not self.params['quit']:
                code = func(self, *args, **kwargs)
                if code != 0:
                    self.params['quit'] = True
                    raise SystemError('robotic arm error!')
        return error_checker

    @property
    def has_sample_holder(self):
        if super().get_gripper_position()[1] < 20:
            return True
        else:
            return False

    def get_gripper_version(self):
        return super().get_gripper_version()

    @check_error
    def set_servo_angle(self, *args, **kwargs):
        return super().set_servo_angle(*args, **{**self.config.ang_settings_dict['default'], **kwargs})

    @check_error
    def set_position(self, *args, **kwargs):
        return super().set_position(*args, **{**self.config.pos_settings_dict['default'], **kwargs})

    def move_to_pos(self, pos, **kwargs):
        return self.set_position(*pos, **kwargs)

    @check_error
    def set_gripper_position(self, *args, **kwargs):
        return super().set_gripper_position(*args, **kwargs)

    def home(self):
        if self.has_sample_holder or self.sample_in_flask:
            raise SystemError('Cannot go home, unload sample first!')
        else:
            self.stretch()
            self.set_servo_angle(angle=self.config.ang_dict['home'])

    def stretch(self):
        if self.has_sample_holder or self.sample_in_flask:
            raise SystemError('Cannot stretch, unload sample first!')
        else:
            self.set_servo_angle(angle=self.config.ang_dict['stretch'])

    def close_gripper(self):
        self.set_gripper_position(0, wait=True, speed=5000, auto_enable=True)

    def open_gripper(self):
        self.set_gripper_position(self.config.gripper_open_dist, wait=True, speed=5000, auto_enable=True)

    def move_to_hover_pos(self, pos, hover_offset, **kwargs):
        pos_hover = pos.copy()
        pos_hover[2] += hover_offset
        self.move_to_pos(pos_hover, **kwargs)

    def pick_up_sample(self, sample_rack_index):
        raise NotImplementedError('sample pick up needs to be specified in the child class')

    def put_sample_back(self, sample_rack_index):
        raise NotImplementedError('sample put back needs to be specified in the child class')

    def upright_sample(self, reverse=False):
        if reverse:
            self.move_to_pos(
                self.config.pos_dict['vertical_rotation_reverse'],
                relative=True,
                # radius=1,
                **self.config.pos_settings_dict['default']
            )
        else:
            self.move_to_pos(
                self.config.pos_dict['vertical_rotation'],
                relative=True,
                **self.config.pos_settings_dict['default']
            )

    def move_to_mid_station(self):
        self.move_to_pos(self.config.pos_dict['mid_station'])

    def move_to_flask(self):
        self.move_to_hover_pos(self.config.pos_dict['flask_contact_immersed'], self.config.hover_offset_dict['flask'])

    def sink_in_flask(self, reverse=False, immerse_option='flask_contact_immersed'):
        if reverse:
            self.move_to_hover_pos(
                self.config.pos_dict['flask_contact_immersed'],
                self.config.hover_offset_dict['flask'],
                **self.config.pos_settings_dict['slow_3']
            )
            self.sample_in_flask = False
        else:
            self.move_to_pos(self.config.pos_dict[immerse_option], **self.config.pos_settings_dict['slow_3'])
            self.sample_in_flask = True

    def rinsing(self):
        raise NotImplementedError('function needs to be specified in the child class')

    def load_sample(self, sample_rack_index, **kwargs):
        raise NotImplementedError('function needs to be specified in the child class')

    def unload_sample(self, sample_rack_index):
        raise NotImplementedError('function needs to be specified in the child class')