import torch
import copy
import glob
import os
import csv
import functools
import random
import SimpleITK as sitk
import numpy as np
import collections
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple


candidate_info_tuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, diameter_mm, series_uid, center_xyz')
xyz_tuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])
irc_tuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])


@functools.lru_cache(1)
def get_candidate_list(data_root, require_disk_bool=True):
    mhd_path_list = glob.glob(os.path.join(data_root, 'mhd/*.mhd'))
    on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_path_list}
    diameter_dict = {}
    with open(os.path.join(data_root, 'annotations.csv'), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            diameter_dict.setdefault(series_uid, []).append((annotation_center_xyz, annotation_diameter_mm))
    f.close()

    candidate_list = []
    with open(os.path.join(data_root, 'candidates.csv'), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in on_disk_set and require_disk_bool:
                continue

            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])

            candidate_diameter_mm = 0.0
            for annotation_tup in diameter_dict.get(series_uid, []):
                annotation_center_xyz, annotation_diameter_mm = annotation_tup
                for i in range(3):
                    delta_mm = abs(candidate_center_xyz[i] - annotation_center_xyz[i])
                    if delta_mm > annotation_diameter_mm / 4:
                        break
                else:
                    candidate_diameter_mm = annotation_diameter_mm
                    break

            candidate_list.append(
                candidate_info_tuple(
                    is_nodule_bool,
                    candidate_diameter_mm,
                    series_uid,
                    candidate_center_xyz,
                )
            )
    f.close()
    candidate_list.sort(reverse=True)
    return candidate_list


def xyz2irc(coord_xyz, origin_xyz, vx_size_xyz, direction_array):
    origin_array = np.array(origin_xyz)
    vx_size_array = np.array(vx_size_xyz)
    coord_array = np.array(coord_xyz)
    cri_array = ((coord_array - origin_array) @ np.linalg.inv(direction_array)) / vx_size_array
    cri_array = np.round(cri_array)
    return irc_tuple(int(cri_array[2]), int(cri_array[1]), int(cri_array[0]))


class CtRreader:
    def __init__(self, series_uid, data_root):
        self.series_uid = series_uid
        self.data_root = data_root
        mhd_path = glob.glob(os.path.join(self.data_root, 'mhd', str(self.series_uid)+'.mhd'))[0]
        ct_mhd = sitk.ReadImage(mhd_path)
        ct_array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        ct_array.clip(-1000, 1000, ct_array)

        self.hu_a = ct_array
        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_array = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(
            center_xyz,
            self.origin_xyz,
            self.vx_size_xyz,
            self.direction_array,
        )

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis] / 2))
            end_ndx = int(start_ndx + width_irc[axis])
            assert 0 <= center_val < self.hu_a.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]

        return ct_chunk, center_irc


def get_ct_raw_candidate(data_root, series_uid, center_xyz, width_irc):
    ct = CtRreader(series_uid, data_root)
    ct_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    return ct_chunk, center_irc


class LunaDataset(Dataset):
    def __init__(
            self,
            data_root,
            val_stride,
            split='train',
            series_uid=None,
            sort_str='random',
    ):
        self.data_root = data_root
        self.val_stride = val_stride
        self.split = split
        self.series_uid = series_uid
        self.sort_str = sort_str
        self.candidate_info_list = copy.copy(get_candidate_list(self.data_root))

        if series_uid:
            self.candidate_info_list = [x for x in self.candidate_info_list if x.series_uid == series_uid]

        if self.split == 'val':
            self.candidate_info_list = self.candidate_info_list[::val_stride]
        elif val_stride > 0:
            del self.candidate_info_list[::val_stride]
            assert self.candidate_info_list
        if self.sort_str == 'random':
            random.shuffle(self.candidate_info_list)
        elif self.sort_str == 'series_uid':
            self.candidate_info_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif self.sort_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sort_str))

    def __len__(self):
        return len(self.candidate_info_list)

    def __getitem__(self, item):
        candidate_info_tup = self.candidate_info_list[item]
        width_irc = (32, 48, 48)
        candidate_a, center_irc = get_ct_raw_candidate(
            self.data_root,
            candidate_info_tup.series_uid,
            candidate_info_tup.center_xyz,
            width_irc,
        )

        candidate_t = torch.from_numpy(candidate_a).to(torch.float32).unsqueeze(0)
        pos_t = torch.tensor(
            [not candidate_info_tup.isNodule_bool, candidate_info_tup.isNodule_bool],
            dtype=torch.long,
        )
        return candidate_t, pos_t, candidate_info_tup.series_uid, torch.tensor(center_irc)


train_dataloader = DataLoader(
    LunaDataset(
        data_root='/data_hdd2/users/ZhouHeng/Projects/HW/ParallelComputerVision/Luna/',
        val_stride=10
    ),
    batch_size=2,
    shuffle=True,
    pin_memory=True
)

for batch in train_dataloader:
    print('1')


