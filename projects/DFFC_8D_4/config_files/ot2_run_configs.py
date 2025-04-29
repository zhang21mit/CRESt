configs = {
    # prepare precursor and transfer to substrate
    'target_type': 'piece_sample_stage',
    # 'target_type': 'strip_sample_stage',
    # 'target_type': 'piece_sample_stage_steel',
    # 'target_type': 'sheet_sample_stage_triangle',
    'substrate_thickness': 0.37,
    'sample_vol': 10,
    'unit_vol': 10,
    'repeat': 1,
    'direction': 'y', # for AL
    # 'direction': 'x', # not for AL
    'pause': 0,
    # 'pause': 5, # for unit_vol = 5

    # for resuming interrupted run
    # 'target_starting_index': 0,
    # 'batch_size': None,

    # prepare precursor in mixing wells only
    'transfer_to_substrate': False,
    'total_vol': 80,
}
