configs = {

    'layout': {
        # '1': 'strip_sample_stage',
        # '1': 'sheet_sample_stage_triangle',
        '1': 'piece_sample_stage',
        '2': 'nest_96_wellplate_200ul_flat',
        '3': 'opentrons_15_tuberack_falcon_15ml_conical',
        '4': 'piece_sample_stage',
        # '4': 'sheet_sample_stage_triangle',
        '5': 'nest_96_wellplate_200ul_flat',
        '6': 'opentrons_96_tiprack_300ul',
        '7': 'opentrons_96_tiprack_20ul',
        '8': 'opentrons_96_tiprack_20ul',
        '9': 'opentrons_96_tiprack_300ul',
        '10': 'opentrons_96_tiprack_20ul',
        '11': 'opentrons_96_tiprack_20ul',
        # 'corning_24_wellplate_3.4ml_flat',
        # 'nest_96_wellplate_200ul_flat'

    },

    'reservoirs_loc': {
        'reservoir_1': '3',
    },

    'reservoirs_layout': {
        'reservoir_1': {
            'Pd': 'A1',
            'Pt': 'A2',
            'Cu': 'A3',
            'Au': 'A4',
            'Ir': 'A5',
            'Ce': 'B1',
            'Nb': 'B2',
            'Cr': 'B3',
            'Fe': 'B4',
            'Ni': 'B5',
            'Co': 'C1',
            'Bi': 'C2',
            'Sn': 'C3',

        },
    },

    # 'vol_offset_addition': 80,
    'vol_offset_addition': 50, #for DARPA 1uL sample
}
