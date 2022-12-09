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
from utils import *

raw_cache = get_cache('cache_raw')
candidate_info_tuple = namedtuple(
    'candidate_info_tuple',
    'is_nodule_bool, has_annotation_bool, is_mal_bool, diameter_mm, series_uid, center_xyz'
)
xyz_tuple = collections.namedtuple('xyz_tuple', ['x', 'y', 'z'])
irc_tuple = collections.namedtuple('irc_tuple', ['index', 'row', 'col'])


@functools.lru_cache(1)
def get_candidate_list(data_root, require_disk_bool=True):
    mhd_path_list = glob.glob(os.path.join(data_root, 'mhd/*.mhd'))
    on_disk_set = {os.path.split(p)[-1][:-4] for p in mhd_path_list}
    candidate_list = []
    with open(os.path.join(data_root, 'annotations_with_malignancy.csv'), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotation_center_xyz = tuple([float(x) for x in row[1:4]])
            annotation_diameter_mm = float(row[4])
            is_mal_bool = {'False': False, 'True': True}[row[5]]
            candidate_list.append(
                candidate_info_tuple(
                    True,
                    True,
                    is_mal_bool,
                    annotation_diameter_mm,
                    series_uid,
                    annotation_center_xyz,
                )
            )
    f.close()

    with open(os.path.join(data_root, 'candidates.csv'), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in on_disk_set and require_disk_bool:
                continue
            is_nodule_bool = bool(int(row[4]))
            candidate_center_xyz = tuple([float(x) for x in row[1:4]])
            if not is_nodule_bool:
                candidate_list.append(
                    candidate_info_tuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidate_center_xyz
                    )
                )
    f.close()

    candidate_list.sort(reverse=True)
    return candidate_list


@functools.lru_cache(1)
def get_candidate_dict(data_root, require_disk_bool=True):
    candidate_list = get_candidate_list(data_root, require_disk_bool)
    candidate_dict = {}
    for candidate_tup in candidate_list:
        candidate_dict.setdefault(candidate_tup.series_uid, []).append(candidate_tup)

    return candidate_dict


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
        mhd_path = glob.glob(os.path.join(self.data_root, 'mhd', str(self.series_uid) + '.mhd'))[0]
        ct_mhd = sitk.ReadImage(mhd_path)

        self.hu_array = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.origin_xyz = xyz_tuple(*ct_mhd.GetOrigin())
        self.vx_size_xyz = xyz_tuple(*ct_mhd.GetSpacing())
        self.direction_array = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidate_list = get_candidate_dict(self.data_root)[self.series_uid]
        self.positive_list = [candidate_tup for candidate_tup in candidate_list if candidate_tup.is_nodule_bool]
        self.positive_mask = self.build_annotation_mask(self.positive_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1, 2)).nonzero()[0].tolist())

    def build_annotation_mask(self, positive_list, threshold_hu=-700):
        bbox_array = np.zeros_like(self.hu_array, dtype=np.bool)

        for candidate_tup in positive_list:
            center_irc = xyz2irc(
                candidate_tup.center_xyz,
                self.origin_xyz,
                self.vx_size_xyz,
                self.direction_array,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_array[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_array[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_array[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_array[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_array[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_array[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            bbox_array[ci - index_radius: ci + index_radius + 1,
                       cr - row_radius: cr + row_radius + 1,
                       cc - col_radius: cc + col_radius + 1] = True

        mask_array = bbox_array & (self.hu_array > threshold_hu)

        return mask_array

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
            assert 0 <= center_val < self.hu_array.shape[axis], \
                repr([self.series_uid, center_xyz, self.origin_xyz, self.vx_size_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_array.shape[axis]:
                end_ndx = self.hu_array.shape[axis]
                start_ndx = int(self.hu_array.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_array[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def get_ct_raw_candidate(data_root, series_uid, center_xyz, width_irc):
    ct = CtRreader(series_uid, data_root)
    ct_chunk, pos_chunk, center_irc = ct.get_raw_candidate(center_xyz, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc


@raw_cache.memoize(typed=True)
def get_ct_sample_size(series_uid, data_root):
    ct = CtRreader(series_uid, data_root)
    return int(ct.hu_array.shape[0]), ct.positive_indexes


class LunaSegDataset(Dataset):
    def __init__(
            self,
            data_root,
            val_stride,
            split='train',
            series_uid=None,
            context_slices_count=3,
            full_ct_bool=False,
    ):
        self.data_root = data_root
        self.split = split
        self.context_slices_count = context_slices_count
        self.full_ct_bool = full_ct_bool
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(get_candidate_dict(self.data_root).keys())
        if self.split == 'val':
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            index_count, positive_indexes = get_ct_sample_size(series_uid, data_root=self.data_root)
            if self.full_ct_bool:
                self.sample_list += [(series_uid, slice_idx) for slice_idx in range(index_count)]
            else:
                self.sample_list += [(series_uid, slice_idx) for slice_idx in positive_indexes]

        self.candidate_list = get_candidate_list(self.data_root)
        series_set = set(self.series_list)
        self.candidate_list = [cit for cit in self.candidate_list if cit.series_uid in series_set]
        self.pos_list = [nt for nt in self.candidate_list if nt.is_nodule_bool]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, item):
        series_uid, slice_ndx = self.sample_list[item % len(self.sample_list)]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_full_slice(self, series_uid, slice_idx):
        ct = CtRreader(series_uid)
        ct_t = torch.zeros((self.context_slices_count * 2 + 1, 512, 512))
        start_idx = slice_idx - self.contextSlices_count
        end_idx = slice_idx + self.contextSlices_count + 1
        for i, context_idx in enumerate(range(start_idx, end_idx)):
            context_idx = max(context_idx, 0)
            context_idx = min(context_idx, ct.hu_array.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_array[context_idx].astype(np.float32))
        ct_t.clamp_(-1000, 1000)
        pos_t = torch.from_numpy(ct.positive_mask[slice_idx]).unsqueeze(0)
        return ct_t, pos_t, ct.series_uid, slice_idx


class LunaSegDatasetTrain(LunaSegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2

    def __len__(self):
        return 300000

    def shuffle_samples(self):
        random.shuffle(self.candidate_list)
        random.shuffle(self.pos_list)

    def __getitem__(self, idx):
        candidate_tup = self.pos_list[idx % len(self.pos_list)]
        return self.getitem_training_crop(candidate_tup)

    def getitem_training_crop(self, candidate_tup):
        ct_a, pos_a, center_irc = get_ct_raw_candidate(
            self.data_root,
            candidate_tup.series_uid,
            candidate_tup.center_xyz,
            (7, 96, 96),
        )
        pos_a = pos_a[3: 4]

        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_t = torch.from_numpy(ct_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.float32)
        pos_t = torch.from_numpy(pos_a[:, row_offset:row_offset+64, col_offset:col_offset+64]).to(torch.long)

        slice_ndx = center_irc.index

        return ct_t, pos_t, candidate_tup.series_uid, slice_ndx


train_dataloader = DataLoader(
    LunaSegDatasetTrain(
        data_root='/data_hdd2/users/ZhouHeng/Projects/HW/ParallelComputerVision/Luna/',
        val_stride=10
    ),
    batch_size=2,
    shuffle=True,
    pin_memory=True
)

for batch in train_dataloader:
    print('1')
