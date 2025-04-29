configs = {
    # prepare precursor and transfer to substrate
    'target_type': 'sheet_sample_stage',  # 'strip_sample_stage' or 'sheet_sample_stage' or 'heat_stage'
    'substrate_thickness': 0.19,  # unit: mm
    'sample_vol': 10,  # total volume of sample on each spot, unit: uL
    'unit_vol': 5,  # volume to be transferred each time, unit: uL
    'repeat': 1,  # repeat times of each recipe
    'direction': 'y',  # 'x' or 'y', direction of sample preparation
    'pause': 0,  # pause time between each unit pipette, unit: s

    # for resuming interrupted run
    # 'target_starting_index': 0,
    # 'batch_size': None,

    # prepare precursor in mixing wells only
    # 'transfer_to_substrate': False,
    # 'total_vol': 1500,
}
