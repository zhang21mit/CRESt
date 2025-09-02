from robotic_testing.common.robotic_arms.xarm_control import xArm


class xArm6(xArm):
    def __init__(self, config_path=None, **kwargs):
        config_path = config_path or 'robotic_testing.common.robotic_arms.xarm6.xarm6_config'
        super().__init__(config_path, **kwargs)

    def home(self):
        if self.has_sample_holder or self.sample_in_flask:
            raise SystemError('Cannot go home, unload sample first!')
        else:
            self.set_linear_track_pos(350)
            self.set_servo_angle(angle=self.config.ang_dict['home'])

    def disable_linear_track(self):
        self.set_linear_track_enable(False)

    def move_to_linear_track_pos(self, y, lin_pos_low_lim=0, lin_pos_high_lim=700, *args, **kwargs):
        """
        the function to present the advantage of this xarm6 with linear track
        :param y: the desired y position of the target, relative to right bottom point
        :param lin_pos_low_lim: the right most point on linear track, if specified other than 0,
        can stop at this point and robotic arm will compensate the residual
        :param lin_pos_high_lim: the left most point on linear track, if specified other than 0,
        can stop at this point and robotic arm will compensate the residual
        """

        # linear track moving speed
        _speed = 50

        if y < lin_pos_low_lim:
            self.set_linear_track_pos(lin_pos_low_lim, speed=_speed)
            return y - lin_pos_low_lim
        elif y < lin_pos_high_lim:
            self.set_linear_track_pos(y, speed=_speed)
            return 0
        else:
            self.set_linear_track_pos(lin_pos_high_lim, speed=_speed)
            return y - lin_pos_high_lim

    def move_to_pos(self, pos, *args, **kwargs):
        if ('relative', True) in kwargs.items():
            pos_rel = pos.copy()
        else:
            y_residual = self.move_to_linear_track_pos(pos[1], *args, **kwargs) if pos[1] else 0
            pos_rel = pos.copy()
            pos_rel[1] = y_residual
        return self.set_position(*pos_rel, *args, **kwargs)

    def warm_up(self):
        self.move_to_pos(self.config.pos_dict['warm_up'])

    def pick_up_sample(self, sample_rack_index):
        """
        :param sample_rack_index: index among all the slots, e.g. starting from 0 to 35
        """
        # initialize key position
        sample_position = self.config.pos_cal.get_sample_coord(sample_rack_index)

        # open gripper
        self.open_gripper()

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

        # move to sample position
        self.move_to_pos(sample_position)

        # close gripper
        self.close_gripper()

        # move backward to avoid hooking
        self.move_to_pos(self.config.pos_dict['backward_avoid_hooking'], relative=True)

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

    def load_sample(self, sample_rack_index, **kwargs):
        self.pick_up_sample(sample_rack_index)
        self.upright_sample()
        self.move_to_mid_station()
        self.move_to_flask()
        self.sink_in_flask(**kwargs)

    def unload_sample(self, sample_rack_index):
        self.sink_in_flask(reverse=True)
        self.rinsing()
        self.move_to_mid_station()
        self.upright_sample(reverse=True)
        self.put_sample_back(sample_rack_index)

    def put_sample_back(self, sample_rack_index):
        # check if sample still in flask
        if self.sample_in_flask:
            raise SystemError('cannot put back, sample still in the flask!')

        # initialize key position
        sample_position = self.config.pos_cal.get_sample_coord(sample_rack_index)

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

        # move to sample position
        self.move_to_pos(sample_position, **self.config.pos_settings_dict['slow_2'])

        # open gripper
        self.open_gripper()

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

    def rinsing(self):
        lin_settings = self.config.linear_track_dict['rinsing']
        self.move_to_hover_pos(self.config.pos_dict['rinsing'], self.config.hover_offset_dict['rinsing'],
                               **lin_settings)
        self.move_to_pos(self.config.pos_dict['rinsing'], **lin_settings)
        for i in range(5):
            self.move_to_pos(self.config.pos_dict['rinsing_shake'], relative=True, **lin_settings)
            self.move_to_pos(self.config.pos_dict['rinsing'], **lin_settings)
        self.move_to_hover_pos(self.config.pos_dict['rinsing'], self.config.hover_offset_dict['rinsing'],
                               **lin_settings)




