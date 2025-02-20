import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch


def func():
    precipitation_folder = ROOT_DIR / "data" / "precipitation"
    nodes_folder = Path(ROOT_DIR / "data" / "Set6")
    nodes = [f for f in nodes_folder.iterdir() if f.is_dir()]
    node_train = None
    for i,node in enumerate(nodes):
        df = pd.read_csv(node / "data.csv")
        df['DTG'] = pd.to_datetime(df['DTG'])
        #make node_train have years
        nt = df['DTG']
        node_train = nt[nt.dt.year == 2019]
    result = pd.concat([node_train, node_train])

    for i, node in enumerate(node_train):
        date = node
        new_date = date + pd.Timedelta(minutes=5)
        result[i+1] = new_date

    #load hdf5 timestamps
    timestamps = pd.DataFrame(np.array(h5py.File(precipitation_folder / "RAD_NL25_RAC_5min_train_test_2016-2019.h5")['test']['timestamps']))

    merged_stamps = pd.merge(timestamps, result)
    print(len(merged_stamps))
    #save merged timestamps
    merged_stamps.to_csv('merged_timestamps.csv')


