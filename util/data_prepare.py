import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def cond_prepare(file_path, num_sample):
    cond = pd.read_csv(file_path, header=None, nrows=num_sample).to_numpy(dtype=int)
    enc = OrdinalEncoder(dtype=int)
    cond = enc.fit_transform(cond)

    n_categ = [len(_cat) for _cat in enc.categories_]
    return cond, n_categ

def cond_seq_prepare(file_path, num_sample, usecols=None):
    cond_seq = pd.read_csv(file_path, header=None, nrows=num_sample, usecols=usecols)
    cond_seq = cond_seq.to_numpy(dtype=int)
    enc = LabelEncoder()
    original_shape = cond_seq.shape
    cond_seq = enc.fit_transform(cond_seq.flatten()).reshape(original_shape)

    pad_idx = enc.transform([-1]).item() if -1 in enc.classes_ else None
    return cond_seq, len(enc.classes_), pad_idx


class TrajTranslator:
    def __init__(self, code, coordinates):
        self.code2coordinates = pd.DataFrame(coordinates, index=code)
        self.ix2coordinates = pd.DataFrame(coordinates)
        self.ix2code = pd.DataFrame(code)
        self.code2ix = pd.DataFrame(data=self.ix2code.index, index=self.ix2code.values.flatten())
        return

    def trans_ix2coordinate(self, ix):
        original_shape = list(ix.shape) + [2]
        coordi = self.ix2coordinates.loc[ix.flatten(), :].to_numpy()
        coordi = coordi.reshape(original_shape)
        return coordi

    def trans_code2ix(self, code):
        if type(code) is list:
            ix = self.code2ix.loc[code]
            ix = ix.to_numpy().flatten().tolist()
        else:
            raise Exception("Not implemented")
        return ix


#TODO EOS PAD SOS
def traj_prepare(file_path, dict_path, num_sample, use_cols, add_idx=None):
    idx_dict = ["0/SOS", "0/EOS", "0/UNK"]
    if not (add_idx is None):
        assert all(_add_idx in idx_dict for _add_idx in add_idx)
    traj = pd.read_csv(file_path, header=None, nrows=num_sample, usecols=use_cols)
    code_dict = pd.read_csv(dict_path, index_col=0)
    code_dict.index = code_dict.index.astype("U8")
    if not (add_idx is None):
        for _idx in add_idx:
            code_dict.loc[_idx] = 0, 0

    traj = traj.to_numpy(dtype="U8")
    enc = LabelEncoder()
    ori_shape = traj.shape
    if not (add_idx is None):
        traj = enc.fit_transform(np.append(traj, add_idx))
        traj = traj[:-len(add_idx)]
    else:
        traj = enc.fit_transform(traj.flatten())

    traj = traj.reshape(ori_shape)

    code_dict = code_dict.loc[enc.classes_, :]

    translator = TrajTranslator(code_dict.index, code_dict.to_numpy())
    return traj, len(enc.classes_), translator


if __name__ == "__main__":
    source_folder = "/home/ubuntu/pflow_data/filtered"
    code_path = f"{source_folder}/code.csv"
    cond_path = f"{source_folder}/cond.csv"
    cond_seq_path = f"{source_folder}/cond_seq_pad.csv"
    cond_seq_whole_day_path = f"{source_folder}/cond_seq_whole_day.csv"
    traj_path = f"{source_folder}/traj.csv"

    num_samples = 15
    time_cols = np.arange(5) * 30 + 480
    cond, n_cond_categ = cond_prepare(cond_path, num_samples)
    cond_seq, n_cond_seq, pad_idx = cond_seq_prepare(cond_seq_path, num_samples)
    cond_seq2, n_cond_seq2, pad_idx2 = cond_seq_prepare(cond_seq_whole_day_path,
                                                     num_samples, usecols=time_cols)
    traj, n_traj, translator = traj_prepare(traj_path, code_path,
                                            num_samples, use_cols=time_cols, add_idx=["0/SOS", "0/EOS"])

