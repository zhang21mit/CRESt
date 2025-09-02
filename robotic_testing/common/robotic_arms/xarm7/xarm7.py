from robotic_testing.common.robotic_arms.xarm_control import xArm


class xArm7(xArm):
    def __init__(self, config_path=None, **kwargs):
        config_path = config_path or 'robotic_testing.common.robotic_arms.xarm7.xarm7_config'
        super().__init__(config_path, **kwargs)

    def load_sample(self, sample_rack_index, **kwargs):
        self.pick_up_sample(sample_rack_index)
        self.move_to_mid_station()
        self.move_to_flask()
        self.sink_in_flask(**kwargs)

    def unload_sample(self, sample_rack_index):
        self.sink_in_flask(reverse=True)
        self.rinsing()
        self.move_to_mid_station()
        self.put_sample_back(sample_rack_index)

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
        self.set_position(*sample_position)

        # close gripper
        self.close_gripper()

        # move backward to avoid hooking
        self.set_position(*self.config.pos_dict['backward_avoid_hooking'], relative=True)

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

    def put_sample_back(self, sample_rack_index):
        # check if sample still in flask
        if self.sample_in_flask:
            raise SystemError('cannot put back, sample still in the flask!')

        # initialize key position
        sample_position = self.config.pos_cal.get_sample_coord(sample_rack_index)

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

        # move to sample position
        self.set_position(*sample_position, **self.config.pos_settings_dict['slow_2'])

        # open gripper
        self.open_gripper()

        # move to hover position
        self.move_to_hover_pos(sample_position, self.config.hover_offset_dict['sample'])

    def rinsing(self):
        self.move_to_hover_pos(self.config.pos_dict['rinsing'], self.config.hover_offset_dict['rinsing'])
        self.move_to_pos(self.config.pos_dict['rinsing'])
        for i in range(5):
            self.move_to_pos(self.config.pos_dict['rinsing_shake'])
            self.move_to_pos(self.config.pos_dict['rinsing'])
        self.move_to_hover_pos(self.config.pos_dict['rinsing'], self.config.hover_offset_dict['rinsing'])