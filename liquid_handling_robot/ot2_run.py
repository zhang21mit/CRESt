import json
import math
import os
import sys
from time import sleep

import numpy as np
import pandas as pd
from opentrons import protocol_api, types

sys.path.append('/data/user_storage/protocols')

# metadata
metadata = {
    'protocolName': 'main',
    'author': 'Zhichu Ren <zc_ren@mit.edu>',
    'description': 'Active learning integrated pipetting task',
    'apiLevel': '2.15'
}


class OT2_Config:
    def __init__(self, protocol: protocol_api.ProtocolContext, exp_name, deck_configs, simulate):

        self.protocol = protocol

        self.simulate = simulate

        # initialize exp config, update default config if any setting is specified in the current exp
        self.exp_name = exp_name
        self.config = deck_configs

        self.ot2_status_path = '/data/user_storage/ot2_status_simulate' if self.simulate \
            else '/data/user_storage/ot2_status'
        self.tip_tracker_path_20ul = f'{self.ot2_status_path}/tip_tracker_20ul.json'
        self.tip_tracker_path_300ul = f'{self.ot2_status_path}/tip_tracker_300ul.json'
        self.mixing_well_tracker_path = f'{self.ot2_status_path}/mixing_well_tracker.json'
        self.task_path = '/data/user_storage/tasks/task.json'

        # register modules on the deck
        self.layout = {}
        for layout_index, module in self.config['layout'].items():
            # replace the sample stage with a surrogate module name
            if module in self.config['sample_stage_mapping'].keys():
                module = self.config['sample_stage_mapping'][module]
            if isinstance(module, str):
                try:
                    loading = self.protocol.load_labware(module, layout_index)
                except FileNotFoundError:
                        loading = self.protocol.load_module(module, layout_index)
            elif isinstance(module, list):
                base = module[0]
                adapter = module[1]
                base_module = self.protocol.load_module(base, layout_index)
                loading = base_module.load_labware(adapter)
            else:
                raise ValueError(f'cannot load module {module}')
            self.layout[layout_index] = loading

        # register tipracks
        self.tipracks = {
            '20ul': [self.layout[layout_index] for layout_index, module in self.config['layout'].items()
                     if module == 'opentrons_96_tiprack_20ul'],
            '300ul': [self.layout[layout_index] for layout_index, module in self.config['layout'].items()
                      if module == 'opentrons_96_tiprack_300ul'],
        }

        self.mixing_wells = [self.layout[layout_index] for layout_index, module in self.config['layout'].items()
                             if module in self.config['supported_module_list']['mixing_well']]
        self.mixing_wells_cap = len(self.mixing_wells[0].wells())
        self.mixing_well_model = self.mixing_wells[0].name
        self.mixing_well_settings = self.config['mixing_well_settings'][self.mixing_well_model]
        self.d_to_well_bottom = self.mixing_well_settings['d_to_well_bottom']

        # pipettors
        self.pipettor_300 = self.protocol.load_instrument(
            'p300_single_gen2',
            self.config['pipettor']['p300_single_gen2'],
            tip_racks=self.tipracks['300ul']
        )
        self.pipettor_20 = self.protocol.load_instrument(
            'p20_single_gen2',
            self.config['pipettor']['p20_single_gen2'],
            tip_racks=self.tipracks['20ul']
        )
        self.pipettors = {'300ul': self.pipettor_300, '20ul': self.pipettor_20}

        # reservoir
        self.reservoirs = [self.layout[layout_index] for layout_index, module in self.config['layout'].items()
                           if module in self.config['supported_module_list']['reservoir']]
        self.reservoirs_layout = self.config['reservoirs_layout']

        # sample stage
        self.sample_stage_settings = self.config['sample_stage_settings']
        self.stage_indices = {
            stage_type: [layout_index for layout_index, module in self.config['layout'].items() if module == stage_type]
            for stage_type in self.config['sample_stage_mapping'].keys()
        }

        # pipette coord history
        self.pipette_coord_log = []

    # duplicate ot2 status dir if in simulation
    def duplicate_ot2_status_dir(self):
        if self.simulate:
            os.system('cp -rT /data/user_storage/ot2_status /data/user_storage/ot2_status_simulate')

    def turn_on_light(self, on=True):
        self.protocol.set_rail_lights(on)

    # normalize recipe if not add up to 1
    @staticmethod
    def normalize_task(task: pd.DataFrame):
        task_norm = task.copy()
        for task_index, task_recipe in task_norm.iterrows():
            # skip if all 0
            if any(task_recipe) != 0:
                task_norm.iloc[task_index] = np.round(task_norm.iloc[task_index] / task_norm.iloc[task_index].sum(), 3)
        return task_norm

    # liquid level tracking function
    def h_track(self, vol, tube_index, reservoir_name):

        # reservoir config setting
        reservoir_model = self.config['layout'][self.config['reservoirs_loc'][reservoir_name]]
        reservoir_settings = self.config['reservoir_settings'][reservoir_model]
        tube_diameter = reservoir_settings['tube_diameter']
        compensation_coeff = reservoir_settings['compensation_coeff']
        min_height = reservoir_settings['min_height']

        reservoir_path = f'{self.ot2_status_path}/{self.exp_name}.json'

        # if no exp corresponding reservoir registered, use default one
        try:
            heights = json.loads(json.load(open(reservoir_path)))
        except FileNotFoundError:
            reservoir_path = f'{self.ot2_status_path}/default.json'
            heights = json.loads(json.load(open(reservoir_path)))

        # calculate height decrement based on volume
        dh = round((vol / (math.pi * ((tube_diameter / 2) ** 2))) * compensation_coeff, 3)

        # make sure height decrement will not crash into the bottom of the tube
        if heights[reservoir_name][tube_index] - dh > min_height:
            h = heights[reservoir_name][tube_index] - dh
        else:
            raise ValueError(f'{tube_index} is out of stock')
        heights[reservoir_name][tube_index] = h

        # write current heights to external json file
        json.dump(json.dumps(heights), open(reservoir_path, "w"))
        return h

    # tip array tracker, type could be '20ul' or '300ul'
    def tip_tracking(self, tip_type):
        assert tip_type in ['20ul', '300ul'], 'tip type must be 20ul or 300ul'
        if tip_type == '20ul':
            path = self.tip_tracker_path_20ul
        else:
            path = self.tip_tracker_path_300ul
        tip_tracker = json.loads(json.load(open(path)))
        tip_tracker_new = tip_tracker + 1

        # reset to 0 if ran out of tip in the last tiprack, be sure to refill the first tiprack after using
        # all tips in the first tiprack
        if tip_tracker_new < 96 * len(self.tipracks[tip_type]):
            json.dump(json.dumps(tip_tracker_new), open(path, "w"))
        else:
            self.pipettors[tip_type].reset_tipracks()
            tip_tracker_new = 0
            json.dump(json.dumps(tip_tracker_new), open(path, "w"))
            print(f'ran out of tips of type {tip_type}, go back rack 1')
        return tip_tracker

    def get_tip_position(self, tip_type):
        assert tip_type in ['20ul', '300ul'], 'tip type must be 20ul or 300ul'
        div, mod = divmod(self.tip_tracking(tip_type), 96)
        # return self.tipracks[tip_type][int(div)].well(self.tip_array[int(mod)])
        return [tip for rows in self.tipracks[tip_type][int(div)].rows() for tip in rows][int(mod)]

    def mixing_well_tracking(self):
        mixing_well_tracker = json.loads(json.load(open(self.mixing_well_tracker_path)))
        mixing_well_tracker_new = mixing_well_tracker + 1
        if mixing_well_tracker_new < self.mixing_wells_cap * len(self.mixing_wells):
            json.dump(json.dumps(mixing_well_tracker_new), open(self.mixing_well_tracker_path, "w"))
        else:
            mixing_well_tracker_new = 0
            json.dump(json.dumps(mixing_well_tracker_new), open(self.mixing_well_tracker_path, "w"))
            print(f'used all rows of mixing well, go back to well 1')
        return mixing_well_tracker

    def get_well_target_coord(self):
        div, mod = divmod(self.mixing_well_tracking(), self.mixing_wells_cap)
        return [well for rows in self.mixing_wells[int(div)].rows() for well in rows][int(mod)]

    def pick_up_tip(self, tip_type):
        assert tip_type in ['20ul', '300ul'], 'tip type must be 20ul or 300ul'
        self.pipettors[tip_type].starting_tip = self.get_tip_position(tip_type)
        self.pipettors[tip_type].pick_up_tip()

    # only can pipette vol not exceeding pipettor limit
    def pipette_base(
            self,
            tip_type: str,
            source: protocol_api.labware.Well,
            d_to_bottom: float,
            volume: float,
            target_type: str,
            target: tuple,
            change_tip: bool = True
    ):
        assert tip_type in ['20ul', '300ul'], 'tip type must be 20ul or 300ul'
        assert target_type in ['sample', 'well'], 'target type must be sample or well'

        if tip_type == '20ul':
            pipettor = self.pipettors['20ul']
        else:
            pipettor = self.pipettors['300ul']

        if change_tip and not pipettor.has_tip:
            self.pick_up_tip(tip_type)

        pipettor.aspirate(volume, source.bottom(d_to_bottom))

        air_gap_config = self.config.get('air_gap', {})
        air_gap = air_gap_config.get(tip_type, 0)
        if air_gap:
            pipettor.air_gap(volume=air_gap)

        if target_type == 'sample':
            x, y, z = target[1][0], target[1][1], target[1][2]
            pipettor.move_to(self.protocol.deck.position_for(target[0]).move(types.Point(x, y, z)))
            pipettor.dispense(volume + air_gap, rate=self.config['pipetting_rate_to_sample'])

            # if blow_out_to_sample
            blow_out_to_sample = self.config.get('blow_out_to_sample', False)
            if blow_out_to_sample:
                pipettor.dispense().blow_out()

        elif target_type == 'well':
            assert isinstance(target, protocol_api.labware.Well), 'well type target must be a well instance'
            pipettor.move_to(target.bottom(self.d_to_well_bottom))
            pipettor.dispense(volume + air_gap, rate=self.config['pipetting_rate_to_mixing_well'])

            # if blow_out_to_mixing_well
            blow_out_to_mixing_well = self.config.get('blow_out_to_mixing_well', False)
            if blow_out_to_mixing_well:
                pipettor.dispense().blow_out()

            # touch tip to mixing well to remove residual hanging droplet
            pipettor.touch_tip(
                radius=self.mixing_well_settings['touch_tip_radius'],
                v_offset=self.mixing_well_settings['touch_tip_v_offset']
            )

        if change_tip:
            pipettor.drop_tip()

    # handle situation that pipette vol exceeds pipettor limit
    def pipette(
            self,
            source: protocol_api.labware.Well,
            d_to_bottom: float,
            volume: float,
            target_type: str,
            target: tuple,
            change_tip: bool = True
    ):
        pipette_config = {
            'source': source,
            'd_to_bottom': d_to_bottom,
            'target_type': target_type,
            'target': target,
            'change_tip': change_tip
        }
        # only one type pipette case
        if len(self.tipracks['20ul']) == 0 or len(self.tipracks['300ul']) == 0:
            if len(self.tipracks['20ul']) == 0:
                tip_type = '300ul'
            else:
                tip_type = '20ul'
            div, mod = divmod(volume, self.pipettors[tip_type].max_volume)
            for i in range(int(div)):
                v = self.pipettors[tip_type].max_volume
                self.pipette_base(
                    volume=v,
                    tip_type=tip_type,
                    **pipette_config,
                )
            # if pipetting vol is 0, ot2 will pipette maximum capacity of current tip by default, which is undesired
            if mod > 0:
                self.pipette_base(
                    volume=mod,
                    tip_type=tip_type,
                    **pipette_config,
                )

        # two types pipette available case, auto-switch
        else:
            div, mod = divmod(volume, 300)
            while div > 0:
                tip_type = '300ul'
                v = self.pipettors[tip_type].max_volume
                self.pipette_base(
                    volume=v,
                    tip_type=tip_type,
                    **pipette_config,
                )
                div -= 1
            if mod > 60:
                tip_type = '300ul'
                self.pipette_base(
                    volume=mod,
                    tip_type=tip_type,
                    **pipette_config,
                )
            # if pipetting vol is 0, ot2 will pipette maximum capacity of current tip by default, which is undesired
            elif mod > 0:
                tip_type = '20ul'
                div_2, mod_2 = divmod(mod, 20)
                for i in range(int(div_2)):
                    v = self.pipettors[tip_type].max_volume
                    self.pipette_base(
                        volume=v,
                        tip_type=tip_type,
                        **pipette_config,
                    )
                if mod_2 > 0:
                    self.pipette_base(
                        volume=mod_2,
                        tip_type=tip_type,
                        **pipette_config,
                    )

    def get_target_pos(self, target_type, target_coord, substrate_thickness):
        sample_stage_settings = self.sample_stage_settings[target_type]

        grid = sample_stage_settings['grid']

        grid_spacing = np.array(sample_stage_settings['grid_spacing'])
        grid_spacing_offset = np.array(sample_stage_settings['grid_spacing_offset'])
        grid_spacing_updated = grid_spacing + grid_spacing_offset

        grid_starting_point = np.array(sample_stage_settings['grid_starting_point'])
        grid_starting_point_offset = np.array(sample_stage_settings['grid_starting_point_offset'])
        grid_starting_point_updated = grid_starting_point + grid_starting_point_offset

        # switch stage if exceeds first stage capacity
        div, mod = divmod(target_coord[1], grid[1])
        if target_type == 'sheet_sample_stage':
            # sheet sample stage occupies two slots
            assert div < len(self.stage_indices[target_type]) / 2, 'target coord exceeds sample stage capacity'
            # needs special handling for two sheet sample stage (4 slots)
            if div > 0:
                raise NotImplementedError('stage indices need special handling in two sheet sample stage case')
            # sheet sample stage starts with the slot with larger index (top half)
            stage_index = self.stage_indices[target_type][1]
        else:
            assert div < len(self.stage_indices[target_type]), 'target coord exceeds sample stage capacity'
            stage_index = self.stage_indices[target_type][div]
        target_coord_array = np.array([target_coord[0], mod, 0])

        target_pos = grid_starting_point_updated + target_coord_array * grid_spacing_updated

        # add substrate thickness to z
        target_pos[2] += substrate_thickness

        return stage_index, target_pos

    def get_coord_list_for_one_recipe(self, starting_coord, step, repeat, direction):
        """
        :param starting_coord: starting point in the grid
        :param step: step >= 1, 1 means next to each other
        :param repeat: repeating times >= 1, including starting point
        :param direction: x means horizontally pipette, y means vertically pipette
        :return: list of coords for a specific recipe
        """
        coord_list = []
        for i in range(repeat):
            if direction == 'x':
                coord_list.append([starting_coord[0] + i * step, starting_coord[1]])
            else:
                coord_list.append([starting_coord[0], starting_coord[1] + i * step])
        return coord_list

    def get_total_vol(self, sample_vol, repeat):
        total_vol = repeat * sample_vol * self.config['vol_offset_multiplier'] + self.config['vol_offset_addition']
        assert total_vol <= 300, f'total vol is {total_vol}, which is larger than the capacity of 300ul well'
        return total_vol

    def find_precursor_loc(self, target_precursor):
        for reservoir_name, reservoir_layout in self.config['reservoirs_layout'].items():
            if reservoir_layout.get(target_precursor):
                reservoir = self.layout[self.config['reservoirs_loc'][reservoir_name]]
                return reservoir, reservoir_name, reservoir_layout.get(target_precursor)
        # if not returned above, meaning target precursor not found
        raise KeyError(f'cannot find precursor {target_precursor} in reservoir!')

    def make_precursor_mixture(self, task_recipe, total_vol, well_target):
        # pipette precursor into target well
        for precursor, percentage in task_recipe.items():
            if percentage != 0:
                reservoir, reservoir_name, reservoir_index = self.find_precursor_loc(precursor)

                volume = round(percentage * total_vol, 3)

                # get reservoir height and update reservoir
                h = self.h_track(volume, self.reservoirs_layout[reservoir_name][precursor], reservoir_name)

                # note the tube here is an OT-2 instance rather than a str
                tube = reservoir[reservoir_index]
                self.pipette(tube, h, volume, 'well', well_target)

        # mix precursor in the target well
        mixing_tip_type = self.config['mixing_tip_type']
        mixing_vol = self.config['mixing_vol']
        mixing_times = self.config['mixing_times']
        mixing_rate = self.config['mixing_rate']
        self.pick_up_tip(mixing_tip_type)
        self.pipettors[mixing_tip_type].mix(mixing_times, mixing_vol, well_target.bottom(self.d_to_well_bottom),
                                            rate=mixing_rate)

        # if blow_out_last_mixing
        blow_out_last_mixing = self.config.get('blow_out_last_mixing', False)
        if blow_out_last_mixing:
            self.pipettors[mixing_tip_type].dispense().blow_out()

        self.pipettors[mixing_tip_type].drop_tip()

    def get_starting_coord(self, task_index, grid, direction, target_starting_index):
        if direction == 'y':
            div, mod = divmod(task_index + target_starting_index, grid[0])
            starting_coord = [mod, div]
        else:
            starting_coord = [0, task_index + target_starting_index]
        return starting_coord

    @staticmethod
    def get_benchmark_coord(stage_id, grid):
        div, mod = divmod(stage_id - 1, grid[0])
        return [mod, div]

    def check_benchmark_conflict(self, benchmark_coord):
        return benchmark_coord in self.pipette_coord_log

    def get_benchmark_coord_and_check_conflict(self, stage_id, grid):
        benchmark_coord = self.get_benchmark_coord(stage_id, grid)
        if self.check_benchmark_conflict(benchmark_coord):
            raise ValueError(
                f'stage_id {stage_id} points to row {benchmark_coord[1]} col {benchmark_coord[0]}, which conflicts with previous sample pipetting')
        return benchmark_coord

    @staticmethod
    def get_step(batch_size, tasks, direction, grid):
        batch_size = batch_size or len(tasks)
        if direction == 'y':
            div, mod = divmod(batch_size - 1, grid[0])
            step = div + 1
        else:
            step = 1
        return step

    def transfer_mixture_to_coord_list(
            self,
            well_target,
            coord_list,
            target_type,
            substrate_thickness,
            sample_vol,
            unit_vol,
            pause,
    ):
        if unit_vol:
            times, res = divmod(sample_vol, unit_vol)
            vol = unit_vol
        else:
            times, res = 1, 0
            vol = sample_vol

        for i in range(int(times)):
            self.transfer_mixture_to_coord_list_base(
                well_target=well_target,
                coord_list=coord_list,
                target_type=target_type,
                substrate_thickness=substrate_thickness,
                vol=vol,
                pause=pause,
            )
        if res:
            self.transfer_mixture_to_coord_list_base(
                well_target=well_target,
                coord_list=coord_list,
                target_type=target_type,
                substrate_thickness=substrate_thickness,
                vol=res,
                pause=pause,
            )

    def transfer_mixture_to_coord_list_base(
            self,
            well_target,
            coord_list,
            target_type,
            substrate_thickness,
            vol,
            pause,
    ):
        for coord in coord_list:
            pos = self.get_target_pos(target_type, coord, substrate_thickness)
            self.pipette(well_target, self.d_to_well_bottom, vol, 'sample', pos, change_tip=False)

            # do not pause in simulation
            pause = pause if not self.simulate else None
            if pause:
                sleep(pause)

    def ot2_main(
            self,
            tasks,
            benchmark_tasks: dict = None,
            target_type=None,
            substrate_thickness=0.19,
            transfer_to_substrate=True,
            total_vol=None,
            sample_vol=10,
            unit_vol=None,
            repeat=3,
            direction='y',
            pause=None,
            target_starting_index=0,
            batch_size=None,
            **kwargs
    ):
        """
        the main function for running ot2 platform
        :param tasks: a dataframe containing all the tasks, each row is a recipe, each column is a component element
        :param benchmark_tasks: a dict containing all the benchmark tasks, in the format of {stage_id: benchmark_recipe}. Note that stage_id here is a 1-d indexing of sample stage, starting from 1, in the direction of x. e.g. for strip sample stage of 5 col and 6 rows, stage_id = 6 means the first slot of the second row. benchmark_recipe is in the format of pandas dataframe, each row is a recipe, each column is a component element.
        :param exp_name: the name of the reservoir to be used in this round
        :param target_type: sheet, strip sample stage or the heat stage
        :param substrate_thickness: the thickness of the substrate, used for calculating the height of pipetting, unit: mm
        :param transfer_to_substrate: whether transfer the liquid mixture to carbon substrate
        :param total_vol: total volume to mix, if not specified, will be calculated based on sample vol and repeat
        :param sample_vol: the volume of each sample
        :param unit_vol: if specified (as int), unit vol is the vol to be used when transfer mixture to stage each time,
        otherwise, transfer in sample vol in one time
        :param repeat: the repeating times of each sample
        :param direction: can be either x (for horizontal pipetting) or y (for vertically pipetting)
        :param pause: if specified (as int), pause for that number of seconds after each transfer mixture to stage
        :param target_starting_index: starting offset when calculating the starting point
        :param batch_size: batch_size and target_starting_index are used for resuming an interrupted task, like the
        original batch_size is 10, an accident showed up when preparing recipe 4, then batch size info is necessary
        for resuming the remaining 6 recipes. No need to specify it if start from index 0.
        :return: None
        """

        # asserts
        assert target_type in self.config[
            'sample_stage_mapping'].keys(), f'target_type must be one of the sample stage in list {self.config["sample_stage_mapping"].keys()}'
        assert direction in ['x', 'y'], 'direction can be either x or y'
        assert isinstance(pause, (int, type(None))), 'pause can be either None or an int for sleep seconds'

        # duplicate ot2 status directory if in simulation
        self.duplicate_ot2_status_dir()

        # initialize task variables
        total_vol = total_vol or self.get_total_vol(sample_vol, repeat)
        grid = self.sample_stage_settings[target_type]['grid']
        step = self.get_step(batch_size, tasks, direction, grid)

        # iterate over all the recipes in the current task list
        for task_index, task_recipe in tasks.iterrows():

            # skip if current task has no recipe assigned on purpose
            if np.all(task_recipe.values == 0):
                continue

            # make precursor mixture in the next well
            well_target = self.get_well_target_coord()
            self.make_precursor_mixture(task_recipe, total_vol, well_target)

            if transfer_to_substrate:
                self.pick_up_tip('20ul')
                # transfer the mixture to stage
                starting_coord = self.get_starting_coord(task_index, grid, direction, target_starting_index)
                coord_list = self.get_coord_list_for_one_recipe(starting_coord, step, repeat, direction)
                self.pipette_coord_log += coord_list
                self.transfer_mixture_to_coord_list(well_target, coord_list, target_type, substrate_thickness,
                                                    sample_vol, unit_vol, pause)

                # drop tip after job done
                self.pipettors['20ul'].drop_tip()

        # iterate over all the recipes in the benchmark task list
        if benchmark_tasks:
            for stage_id, benchmark_task in benchmark_tasks.items():

                # make precursor mixture in the next well
                well_target = self.get_well_target_coord()
                self.make_precursor_mixture(benchmark_task.iloc[0], total_vol, well_target)

                if transfer_to_substrate:
                    # transfer the mixture to stage
                    benchmark_coord = [self.get_benchmark_coord_and_check_conflict(stage_id, grid)]
                    self.transfer_mixture_to_coord_list(well_target, benchmark_coord, target_type, substrate_thickness,
                                                        sample_vol, unit_vol, pause)

                    # drop tip after job done
                    self.pipettors['20ul'].drop_tip()


def run(protocol: protocol_api.ProtocolContext):

    run_configs = json.loads(json.load(open('/data/user_storage/tasks/run_configs.json')))
    deck_configs = json.loads(json.load(open('/data/user_storage/protocols/deck_configs.json')))
    benchmark = json.loads(json.load(open('/data/user_storage/tasks/benchmark.json')))

    # initialize exp
    exp_name = run_configs['exp_name']
    simulate = run_configs['simulate']

    exp = OT2_Config(protocol, exp_name, deck_configs, simulate)

    # turn on the lights
    exp.turn_on_light()

    # import tasks
    tasks = exp.normalize_task(pd.read_json(exp.task_path))

    # import benchmark
    benchmark_tasks = {int(stage_id): exp.normalize_task(pd.DataFrame(benchmark_task)) for stage_id, benchmark_task in
                       benchmark.items()} if benchmark else None

    # heat stage
    # exp.heat_stage.set_temperature(70)

    exp.ot2_main(tasks=tasks, benchmark_tasks=benchmark_tasks, **run_configs)

    # cool stage
    # exp.heat_stage.deactivate()

    # turn off the light
    exp.turn_on_light(False)
