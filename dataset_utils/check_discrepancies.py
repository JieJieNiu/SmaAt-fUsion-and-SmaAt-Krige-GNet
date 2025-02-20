import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch

def beep():
    """Create a dataset that has target images containing at least `rain_amount_thresh` (percent) of rain."""

    data_type = 'train'

    precipitation_folder = ROOT_DIR
    nodes_folder = Path(ROOT_DIR / "data" / "Set6")
    nodes = [f for f in nodes_folder.iterdir() if f.is_dir()]
    node_train = None
    for i,node in enumerate(nodes):
        df = pd.read_csv(node / "data.csv")
        df['DTG'] = pd.to_datetime(df['DTG'])
        #make node_train have years
        nt = df['DTG']
        if data_type == 'test':
            node_train = nt[(nt.dt.year < 2024) & (nt.dt.year >= 2022)]
        else:
            node_train = nt[(nt.dt.year < 2022) & (nt.dt.year >= 2013)]
    result = result = pd.concat([node_train.copy(),node_train.copy()])

    for i, node in enumerate(node_train):
        date = node
        new_date = date + pd.Timedelta(minutes=5)
        result.iloc[i+1] = new_date

    nodes_length = len(result)
    train_timestamps_len = None
    set_a = set(result)
    #print(set_a)
    set_b = set()
    nan_values = None
    with h5py.File(
        precipitation_folder / "RAD_NL21_PRECIP4.h5",
        "r",
    ) as orig_f:
        train_timestamps = orig_f[data_type]["timestamps"]
        train_timestamps_len = len(train_timestamps)
        train_timestamps = pd.DataFrame(train_timestamps)
        print(train_timestamps)
        timestamps = train_timestamps.apply(lambda x: x[0].decode('utf-8').replace(';', ' '), axis=1)
        print(timestamps)
        timestamps = pd.to_datetime(timestamps)
        print(timestamps)
        #timestamps = train_timestamps
        set_b = set(timestamps)
        

    print(sorted(set_a)[0])
    print(sorted(set_b)[0])
    diff = set_b ^ set_a
    output = pd.DataFrame(sorted(diff), columns=['DTG'])
    #print(output)
    #print nodes_length and train_timestamps length
    print(nodes_length)
    print(train_timestamps_len)
    output.to_csv("bad_index7_" + data_type + '.csv')
    #print(df)

if __name__ == "__main__":
    beep()