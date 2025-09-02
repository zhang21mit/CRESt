import numpy as np


def pos_cal(ref_point, offset):
    """
    calculate the position with an offset to the reference point
    :param ref_point: a position to be referenced
    :param offset: a tuple, e.g. (0, 20) stands for x_offset = 0, y_offset = 20;
    (0, 1.7, 0.2) stands for x_offset = 0, y_offset = 1.7, z_offset = 0.2
    :return: new position
    """
    ref_point_copy = ref_point.copy()
    for axis, offset_value in enumerate(offset):
        ref_point_copy[axis] += offset_value
    return ref_point_copy


def speed_changer(setting_dict, kwargs, multiplier):
    new_setting_dict = setting_dict.copy()
    for kwarg in kwargs:
        new_setting_dict[kwarg] = new_setting_dict[kwarg] * multiplier
    return new_setting_dict


class SamplePosCal:
    def __init__(
            self,
            grid_d,
            num_slot,
            rack_grid_pos,
            rack_offset,
            slot_layout_on_rack,
            rack_grid_starting_pos,
            contact_foil_offset,
    ):
        """
        :param grid_d: the unit distance of the grid
        :param num_slot: number of slots on each rack
        :param rack_grid_pos: the pattern of the rack grid and how they are indexed
        :param rack_offset: the installation offset of each rack
        :param slot_layout_on_rack: the distance of each slot on one rack
        :param rack_grid_starting_pos: the pos of the first rack first slot
        :param contact_foil_offset: the offset of the thickness of the contact gold foil
        """
        self.grid_d = grid_d
        self.num_slot = num_slot
        self.rack_grid_pos = rack_grid_pos
        self.rack_offset = rack_offset
        self.slot_layout_on_rack = slot_layout_on_rack
        self.rack_grid_starting_pos = rack_grid_starting_pos
        self.contact_foil_offset = contact_foil_offset
    
    def get_sample_coord(self, sample_rack_index):
        div, mod = divmod(sample_rack_index, self.num_slot)

        # get target rack grid offset
        target_rack_grid_offset = np.array(self.rack_grid_pos[div]) * self.grid_d

        # get the starting pos of the rack of the target sample
        target_rack_starting_coord = pos_cal(self.rack_grid_starting_pos, target_rack_grid_offset)
        
        # get the pos of the target sample on the target rack
        sample_pos = pos_cal(target_rack_starting_coord, (0, self.slot_layout_on_rack[mod]))

        # add correction offset to the pos
        sample_pos_corrected = pos_cal(sample_pos, self.get_slot_offset(div, mod))

        # add contact foil offset to the pos
        sample_pos_corrected_2 = pos_cal(sample_pos_corrected, (0, -self.contact_foil_offset))

        return sample_pos_corrected_2

    def get_slot_offset(self, rack_index, slot_index_on_rack):
        """
        given the starting and ending offset value of a rack, calculate the offset in the middle slot using linear gradient
        :param slot_index_on_rack: slot index on the rack, e.g. starting from 0 to 5
        :param rack_index: index of the rack in the rack grid
        :return: the offset of the target slot on the target rack
        """
        starting_offset = np.array(self.rack_offset[rack_index][0])
        ending_offset = np.array(self.rack_offset[rack_index][1])
        gradient = (ending_offset - starting_offset) / (self.num_slot - 1)
        offset = starting_offset + gradient * slot_index_on_rack
        return offset

