from cgi import test
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import numpy as np
import os
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class dataset_graph(Dataset):
    def __init__(self, dataset_path, inputTimesteps, predictTimestep, train, time = False):
        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep

        root = Path(dataset_path)
        nodes = [f for f in root.iterdir() if f.is_dir()]
        num_nodes = len(nodes)
        scaler = StandardScaler()
        #data list is a tensor of all the data SHAPE = (NODES, TIME, FEATURES)
        train_data = torch.empty(size= (num_nodes, 366335, 8), dtype=torch.float)
        test_data = torch.empty(size= (num_nodes, 166321, 8), dtype=torch.float)
        scale_values = torch.empty(size= (num_nodes, 2, 8), dtype=torch.float)
        for i,node in enumerate(nodes):
            df = pd.read_csv(node / "data.csv")
            df['DTG'] = pd.to_datetime(df['DTG'])
            if train:
                df = df[df['DTG'].dt.year < 2021]
            else:
                df = df[df['DTG'].dt.year >= 2021]
            #drop dtg column
            df = df.drop(columns=['DTG'])
            data = scaler.fit_transform(df.values)
            scale_values[i] = torch.tensor(
                                np.array([scaler.mean_, scaler.var_])
                                ,dtype=torch.float)
            if train:
                train_data[i] = torch.tensor(data, dtype=torch.float)
            else:
                test_data[i] = torch.tensor(data, dtype=torch.float)
        self.x = train_data if train else test_data
        self.scale_values = scale_values

        #TRAINING SUBSET IF NEEDED:
        #self.x = train_data[:,:1000,:] if train else test_data[:,:1000,:]

        # Create an adjacency matrix
        self.edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)

    def __getitem__(self, item):
        x = self.x[:, item:item + self.inputTimesteps, :]
        #take windspeed as target 'FF_10M_10' which has index 3
        y = self.x[:, item + self.inputTimesteps + self.predictTimestep - 1, 3]

        data = Data(x=x, edge_index=self.edge_index, y=y)
        return data

    def __len__(self):
        return self.x.shape[1] - self.inputTimesteps - self.predictTimestep + 1