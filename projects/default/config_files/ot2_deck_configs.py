configs = {

    'pipettor': {
        'p300_single_gen2': 'left',
        'p20_single_gen2': 'right',
    },

    'layout': {
        '1': 'strip_sample_stage',
        '2': 'nest_96_wellplate_200ul_flat',
        '3': 'opentrons_15_tuberack_falcon_15ml_conical',
        '4': 'piece_sample_stage',
        '5': 'nest_96_wellplate_200ul_flat',
        '6': 'opentrons_96_tiprack_300ul',
        '7': 'opentrons_96_tiprack_20ul',
        '8': 'opentrons_96_tiprack_20ul',
        '9': 'opentrons_96_tiprack_300ul',
        '10': 'opentrons_96_tiprack_20ul',
        '11': 'opentrons_96_tiprack_20ul',
    },

    # reservoir location records the name and location on the deck of each reservoir
    'reservoirs_loc': {
        'reservoir_1': '3'
    },

    # reservoir location records the name and the layout of each reservoir
    'reservoirs_layout': {
        'reservoir_1': {
            'Fe': 'A1',
            'Co': 'A2',
            'Ni': 'A3',
            'Cr': 'A4',
            'V': 'A5',
            'In': 'B1',
            'Sn': 'B2',
            'Y': 'B3',
            'W': 'B4',
            'Cu': 'B5',
            'Zn': 'C1',
            'Ag': 'C2',
            'Pt': 'C3',
            'Ru': 'C4',
            'Au': 'C5'
        },
    },

    'supported_module_list': {
        'mixing_well': [
            'nest_96_wellplate_200ul_flat',
            'corning_24_wellplate_3.4ml_flat',
        ],
        'reservoir': [
            'opentrons_15_tuberack_falcon_15ml_conical',
            'opentrons_6_tuberack_nest_50ml_conical',
            'opentrons_24_tuberack_nest_2ml_snapcap',
        ],
    },

    'sample_stage_mapping': {
        'heat_stage': 'temperature module gen2',
        'temp_module_nest_2ml_snapcap': ['temperature module gen2', 'opentrons_24_aluminumblock_nest_2ml_snapcap'],
        'strip_sample_stage': 'axygen_1_reservoir_90ml',
        'sheet_sample_stage': 'axygen_1_reservoir_90ml',
        'sheet_sample_stage_triangle': 'axygen_1_reservoir_90ml',
        'piece_sample_stage': 'axygen_1_reservoir_90ml',
        'piece_sample_stage_wood': 'axygen_1_reservoir_90ml',
        'piece_sample_stage_wood_thick': 'axygen_1_reservoir_90ml',
        'piece_sample_stage_steel': 'axygen_1_reservoir_90ml',
        'coin_cell_stage': 'axygen_1_reservoir_90ml',
    },

    # reservoir liquid level tracking settings
    'reservoir_settings': {
        'opentrons_15_tuberack_falcon_15ml_conical': {
            'tube_diameter': 14.9,
            'compensation_coeff': 1.05,
            'min_height': 15,
        },
        'opentrons_6_tuberack_nest_50ml_conical': {
            'tube_diameter': 35.43,
            'compensation_coeff': 1.1,
            'min_height': 15,
        },
        'opentrons_24_tuberack_nest_2ml_snapcap': {
            'tube_diameter': 10.18,
            'compensation_coeff': 1.1,
            'min_height': 5,
        },
        'temp_module_nest_2ml_snapcap': {
            'tube_diameter': 10.18,
            'compensation_coeff': 1.1,
            'min_height': 5,
        },
    },

    # mixing well pipette settings
    'mixing_well_settings': {
        'nest_96_wellplate_200ul_flat': {
            'd_to_well_bottom': -0.2,
            'touch_tip_radius': 1,
            'touch_tip_v_offset': -1,
        },
        'corning_24_wellplate_3.4ml_flat': {
            'd_to_well_bottom': 1,
            'touch_tip_radius': 0.8,
            'touch_tip_v_offset': -2,
        },
    },

    # pipette rate settings
    'pipetting_rate_to_mixing_well': 1,
    'pipetting_rate_to_sample': 0.05,

    # mixing settings
    'mixing_tip_type': '300ul',
    'mixing_vol': 300,
    'mixing_times': 5,
    'mixing_rate': 5,

    # total vol calculation settings, if mixing vol too small, may lead to mixing failure
    'vol_offset_multiplier': 1.8,
    'vol_offset_addition': 30,

    # dispense blow out setting
    'blow_out_to_mixing_well': True,
    'blow_out_to_sample': True,
    'blow_out_last_mixing': True,

    # tip air gap setting
    'air_gap': {'20ul': 0, '300ul': 0},

    # sample stage settings
    # grid is (x, y) capacity
    # grid spacing is the (x, y, z) distance between each slot
    # grid offset is the center of the first slot to the top left corner, coord base at left bottom

    'sample_stage_settings': {
        'heat_stage': {
            'grid': [5, 6],
            'grid_spacing': [15, -12.7, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [20, 81, 83.5],
            'grid_starting_point_offset': [0, 0, 0],
        },
        'strip_sample_stage': {
            'grid': [5, 6],
            'grid_spacing': [15, -13, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [44.76, 75.8, 10],
            'grid_starting_point_offset': [0, 0, -0.14],
        },
        'sheet_sample_stage': {
            'grid': [6, 6],
            'grid_spacing': [16, -16, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [25.88, 44, 10],
            'grid_starting_point_offset': [0, 0, -0.14],
        },
        'sheet_sample_stage_triangle': {
            'grid': [5, 4],
            'grid_spacing': [3, -3, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [50, 75, 10],
            'grid_starting_point_offset': [0, 0, -0.14],
        },
        'piece_sample_stage': {
            'grid': [6, 6],
            'grid_spacing': [21, -13, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [12.5, 76, 8],
            'grid_starting_point_offset': [0, 0, 0],
        },
        'piece_sample_stage_wood': {
            'grid': [6, 6],
            'grid_spacing': [21, -13, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [12.5, 76, 1.5],
            'grid_starting_point_offset': [0, 0, 0],
        },
        'piece_sample_stage_wood_thick': {
            'grid': [6, 6],
            'grid_spacing': [21, -13, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [12.5, 76, 4],
            'grid_starting_point_offset': [0, 1, 0],
        },
        'piece_sample_stage_steel': {
            'grid': [6, 6],
            'grid_spacing': [21, -13, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [11, 76, 8],
            'grid_starting_point_offset': [0, 0, -0.5],
        },
        'coin_cell_stage': {
            'grid': [5, 3],
            'grid_spacing': [25, -25, 0],
            'grid_spacing_offset': [0, 0, 0],
            'grid_starting_point': [13.88, 67.74, 8],
            'grid_starting_point_offset': [0, 0, 2.1],
        },
    }

}
