import torch

import numpy as np
import pandas as pd

from operator import itemgetter
from sklearn.preprocessing import OrdinalEncoder

# TODO 列表表达式修改
class TrajDatasetCond(torch.utils.data.Dataset):
    def __init__(self, data, label, cond):
        super(TrajDatasetCond, self).__init__()
        self.data = data
        self.label = label
        self.cond = cond
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_x = self.data[idx, :]
        data_y = self.label[idx, :]
        cond = self.cond[idx, :]
        return data_x, data_y, cond


class TrajDatasetCondSeq(torch.utils.data.Dataset):
    def __init__(self, data, label, cond, cond_seq):
        super(TrajDatasetCondSeq, self).__init__()
        self.data = data
        self.label = label
        self.cond = cond
        self.cond_seq = cond_seq
        return

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_x = self.data[idx, :]
        data_y = self.label[idx, :]
        cond = self.cond[idx, :]
        cond_seq = self.cond_seq[idx,:]
        return data_x, data_y, cond, cond_seq


class PflowTranslator:
    def __init__(self, unique_code_proj, all_unique_code, all_unique_coordinate):
        uniq_ix = np.arange(len(unique_code_proj))
        self.code2ix = dict(zip(unique_code_proj, uniq_ix))
        self.ix2code = dict(zip(uniq_ix, unique_code_proj))

        self.code2lng = dict(zip(all_unique_code, all_unique_coordinate[:, 0]))
        self.code2lat = dict(zip(all_unique_code, all_unique_coordinate[:, 1]))

        self.num_class = len(unique_code_proj)
        return

    def trans_code2ix(self, code):
        orig_shape = code.shape
        code = code.reshape(-1)
        code = itemgetter(*code.tolist())(self.code2ix)
        code = np.array(code, dtype=int)
        code = code.reshape(orig_shape)
        return code

    def trans_ix2code(self, ix):
        orig_shape = ix.shape
        ix = ix.reshape(-1)
        ix = itemgetter(*ix.tolist())(self.ix2code)
        ix = np.array(ix, dtype="U8")
        ix = ix.reshape(orig_shape)
        return ix

    def trans_ix2coordi(self, ix):
        code = self.trans_ix2code(ix)
        orig_shape = code.shape
        code = code.reshape(-1)
        lng = itemgetter(*code.tolist())(self.code2lng)
        lat = itemgetter(*code.tolist())(self.code2lat)
        lng, lat = np.array(lng), np.array(lat)
        lng, lat = lng.reshape(orig_shape), lat.reshape(orig_shape)
        lng, lat = np.expand_dims(lng, -1), np.expand_dims(lat, -1)
        coordi = np.concatenate([lng, lat], axis=-1)
        return coordi


def load_pflow(data_path, dict_folder, begin_ix, time_interval, num_per_day,
             begin_code=None):
    all_unique_code = np.loadtxt(f"{dict_folder}/unique_code.txt", dtype="U8")
    all_unique_coordinate = np.loadtxt(f"{dict_folder}/unique_coordinate.txt")
    if not (begin_code is None):
        all_unique_code = np.append(all_unique_code, begin_code)
        all_unique_coordinate = np.concatenate([all_unique_coordinate, [[0, 0]]], axis=0)

    code = pd.read_csv(data_path, header=None, usecols=[14])
    code_arr = code.loc[:, 14].to_numpy(dtype="U8")

    code_arr = code_arr.reshape(-1, 1440)

    ix = begin_ix + np.arange(num_per_day) * time_interval
    code_arr = code_arr[:, ix]
    uniq_code = np.unique(code_arr)

    if not (begin_code is None):
        uniq_code = np.append(uniq_code, begin_code)

    pflow_translator = PflowTranslator(uniq_code, all_unique_code, all_unique_coordinate)

    dataset = pflow_translator.trans_code2ix(code_arr)
    return dataset, pflow_translator


def load_condition(cond_path):
    cond = pd.read_csv(cond_path, header=None)
    cond = cond.to_numpy(dtype=int)
    cond = cond.reshape(-1, 1440, 3)
    cond = cond[:, 0, :]
    return cond

def load_cond_seq(data_path, begin_ix, time_interval, num_per_day):
    cond_seq = pd.read_csv(data_path, header=None, usecols=[13])
    cond_seq = cond_seq.to_numpy(dtype=int)
    cond_seq = cond_seq.reshape(-1, 1440)
    cond_seq = cond_seq[:, begin_ix:(begin_ix+num_per_day*time_interval):time_interval]
    cond_seq = cond_seq.reshape(-1, num_per_day)
    enc = OrdinalEncoder(dtype=int)
    cond_seq = enc.fit_transform(cond_seq)
    return cond_seq


if __name__ == "__main__":
    cond = load_condition("~/data/pflow_mini_condition.csv")
    dt, trans = load_pflow("~/data/pflow_mini_preprocessed.csv",
                             "../dict_file", 480, 15, 4*5)
