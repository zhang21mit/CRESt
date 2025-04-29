from robotic_testing.common.robotic_arms import pos_cal_utils

port = '192.168.1.227'

# robotic arm movement parameter settings
pos_settings_default = {
    'speed': 100,
    'acc': 2000,
    'radius': -1,
    'events': {},
    'callback_in_thread': True,
    'quit': False,
    'wait': True,
}

ang_settings_default = {
    'angle_speed': 20,
    'angle_acc': 500,
    'radius': -1,
    'wait': True,
}

pos_settings_dict = {
    'default': pos_settings_default,
    'slow_1': pos_cal_utils.speed_changer(pos_settings_default, ['speed', 'acc'], 0.5),
    'slow_2': pos_cal_utils.speed_changer(pos_settings_default, ['speed', 'acc'], 0.25),
    'slow_3': pos_cal_utils.speed_changer(pos_settings_default, ['speed', 'acc'], 0.125),
    'fast_1': pos_cal_utils.speed_changer(pos_settings_default, ['speed', 'acc'], 2),
}

ang_settings_dict = {
    'default': ang_settings_default,
}


# pos and ang list, all y value are relative to the bottom right starting point
pos_dict = {
    'sample_starting': [603.0, 597.0, -88.0, 180.0, 0.0, 0.0],
    'warm_up': [380, None, 200, 180.0, 0.0, 0.0],
    'mid_station': [500, None, 400, 180.0, -90.0, 0.0],
    'vertical_rotation': [0, 0, 300, 0, -90, 0],
    'vertical_rotation_reverse': [0, 0, -100, 0, 90, 0],
    'flask_contact_immersed': [553, 786, 129.0, 180.0, -90.0, 0.0],
    'flask_contact_not_immersed': [562, 786, 145.0, 180.0, -90.0, 0.0],
    'flask_without_cap': [562, 786, 145.0, 180.0, -90.0, 0.0],
    'rinsing': [415.0, 975.0, 20.0, 180.0, -90.0, 43.0],
    'rinsing_shake': [10, 10, 10, 0, 0, 0],
    'backward_avoid_hooking': [-2, 0, 0, 0, 0, 0],
}

ang_dict = {
    'home': [90.0, -30.0, 0.0, 0.0, 30.0, 0.0],
    'stretch': [0.0, -30.0, -30.0, 0.0, 0.0, 0.0],
}

hover_offset_dict = {
    'sample': 100,
    'flask': 150,
    'rinsing': 250
}

linear_track_dict = {
    'rinsing': {'lin_pos_high_lim': 450}
}

# position calculator
sample_pos_config = {
    'grid_d': 25,
    'num_slot': 6,
    'rack_grid_pos': {
        0: (0, 0),
        1: (0, -5),
        2: (0, -10),
        3: (0, -15),
        4: (0, -20),
        5: (-5, -0),
        6: (-5, -5),
        7: (-5, -10),
        8: (-5, -15),
        9: (-5, -20),
    },
    'rack_offset': {
        0: [(0, 0, 0), (0, 0, 0)],
        1: [(0, 0, 0), (0, 0, 0)],
        2: [(0, 0, 0), (0, 0, 0)],
        3: [(0, 0, 0), (0, 0, 0)],
        4: [(0, 0, 0), (0, 0, 0)],
        5: [(0, -4, 0), (0, -4, 0)],
        6: [(0, -4, 0), (0, -4, 0)],
        7: [(0, -4, 0), (0, -4, 0)],
        8: [(0, -4, 0), (0, -4, 0)],
        9: [(0, -4, 0), (0, -4, 0)],
    },
    'slot_layout_on_rack': [0, -20, -40, -60, -80, -100],
    'rack_grid_starting_pos': pos_dict['sample_starting'],
    'contact_foil_offset': 0.7
}

pos_cal = pos_cal_utils.SamplePosCal(**sample_pos_config)


# other variables
gripper_open_dist = 55
