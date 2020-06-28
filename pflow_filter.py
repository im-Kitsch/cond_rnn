import pandas as pd
import numpy as np
import os
from util.coordi_map import Coordinate2MeshCodePandas, parse_MeshCode

def remove_if_exist(file):
    file = os.path.expanduser(file)
    if os.path.exists(file):
        os.remove(file)
    return


def pflow_filter_original(source_file, target_folder, chunksize):
    dt = pd.read_csv(source_file, header=None, chunksize=chunksize)
    code_path = os.path.expanduser(f"{target_folder}/code.csv")
    filtered_file_path = os.path.expanduser(f"{target_folder}/filtered_pflow.csv")
    remove_if_exist(filtered_file_path)
    remove_if_exist(code_path)

    counter = 0
    last_t = 0

    unique_code = set()
    for _dt in dt:
        _dt_time = _dt.loc[:, 3]
        ref_t = np.insert(_dt_time.to_numpy(), 0, last_t)
        mask = _dt_time != ref_t[:-1]
        _filtered_dt = _dt[mask]


        _code = Coordinate2MeshCodePandas(_filtered_dt.loc[:, 4], _filtered_dt.loc[:, 5])
        unique_code.update(_code.to_list())

        _filtered_dt = pd.concat([_filtered_dt, _code], axis=1)

        _filtered_dt.to_csv(filtered_file_path, mode="a", index=False, header=False)

        last_t = ref_t[-1]
        counter += 1
        print("processed ", f"{counter*chunksize} samples")

    unique_code = list(unique_code)
    unique_coordi = [parse_MeshCode(_code) for _code in unique_code]
    unique_code_pd = pd.DataFrame(unique_coordi, columns=["lng", "lat"], index=unique_code)
    unique_code_pd.to_csv(code_path)
    return

def gen_condition(file_folder, condition_cols):
    source_data_path = f"{file_folder}/filtered_pflow.csv"
    cond_path = f"{file_folder}/cond.csv"

    remove_if_exist(cond_path)

    dt = pd.read_csv(source_data_path, header=None, chunksize=1440, usecols=condition_cols)

    for _i, _dt in enumerate(dt):
        if _dt.shape[0] < 1440:
            break
        valid_idx = (_dt.index%1440 == 0)
        _cond = _dt.loc[valid_idx]
        _cond.to_csv(cond_path, mode="a", index=False, header=False)
        print(f"condition processed {_i*1440} samples")
    return

def gen_traj(file_folder, traj_col):
    source_file = f"{file_folder}/filtered_pflow.csv"
    traj_file = f"{file_folder}/traj.csv"
    remove_if_exist(traj_file)

    dt = pd.read_csv(source_file, header=None, chunksize=1440, usecols=traj_col)
    for _i, _dt in enumerate(dt):
        if _dt.shape[0] < 1440:
            break
        values = _dt.to_numpy()
        values = pd.DataFrame(values.reshape(-1, 1440))
        values.to_csv(traj_file, mode="a", index=False, header=False)
        print(f"traj processed {_i * 1440} lines")
    return

def gen_cond_seq(file_folder, cond_col):
    source_file = f"{file_folder}/filtered_pflow.csv"
    purpose_file = f"{file_folder}/cond_seq_long.csv"
    purpose_file2 = f"{file_folder}/cond_seq_pad.csv"
    purpose_file3 = f"{file_folder}/cond_seq_whole_day.csv"
    remove_if_exist(purpose_file)
    remove_if_exist(purpose_file2)
    remove_if_exist(purpose_file3)

    dt = pd.read_csv(source_file, header=None, chunksize=1440, usecols=cond_col)
    max_length = 0
    for _i,_dt in enumerate(dt):
        if _dt.shape[0] < 1440:
            break
        pd.DataFrame(_dt.to_numpy().reshape(1, 1440)).to_csv(purpose_file3, mode="a", index=False, header=False)

        _dt = _dt.to_numpy().flatten()
        ref = np.roll(_dt, 1)
        ref[0] = -1
        _dt = _dt[ref != _dt]
        _padded_seq = np.zeros((1, 1440), dtype=int)
        _padded_seq -= 1
        _padded_seq[0, 0:_dt.shape[0]] = _dt

        _len = _dt.shape[0]
        if _len >max_length:
            max_length = _len

        _dt = pd.DataFrame(_padded_seq)
        _dt.to_csv(purpose_file, mode="a", index=False, header=False)
        print(f"cond seq processed {_i*1440} lines")

    _dt.to_csv(f"{file_folder}/max_len{max_length}.csv")

    dt = pd.read_csv(purpose_file, header=None, chunksize=30000)
    for _dt in dt:
        _dt = _dt.loc[:, range(max_length)]
        _dt.to_csv(purpose_file2, mode="a", index=False, header=False)
    remove_if_exist(purpose_file)
    return

def traj_limit(file_folder, lng_range, lat_range):
    traj_path = f"{file_folder}/traj.csv"
    code_path = f"{file_folder}/code.csv"
    traj_limited_path = f"{file_folder}/traj_limited.csv"
    traj = pd.read_csv(traj_path, header=None)
    code = pd.read_csv(code_path, index_col=0)
    lng_min, lng_max = lng_range
    lat_min, lat_max = lat_range

    limited_code = code[(code["lat"]<=lat_max) &
                              (code["lat"]>=lat_min) &
                              (code["lng"]>=lng_min) &
                              (code["lng"]<=lng_max)].index.to_list()
    available_list = []
    for i in range(traj.shape[0]):
        sub_traj = traj.loc[i].to_numpy().flatten()
        true_list = [_p_traj in limited_code for _p_traj in sub_traj]
        if all(true_list):
            available_list.append(i)

        print(i, "th sample")

    limited_code = traj.loc[available_list]
    limited_code.to_csv(traj_limited_path, header=None, index=False)
    return

if __name__ == "__main__":
    # pflow_filter_original("~/pflow_data/pflow.csv", "~/pflow_data/filtered/", 30000)
    gen_cond_seq("~/pflow_data/filtered/", [10])
    gen_condition("~/pflow_data/filtered/", [6, 7, 9])
    gen_traj("~/pflow_data/filtered/", [14])

    traj_limit("~/pflow_data/filtered/", [139, 140], [35.33, 36])
