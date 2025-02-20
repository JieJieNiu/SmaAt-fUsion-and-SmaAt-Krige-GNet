import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
from utils.interpolation_maps import makeKrigeMap

precipitation_folder = ROOT_DIR / "data" / "precipitation"
#create kriging dataset from regular node dataset
K_SIZE = 64 #size of kriging map

def create_dataset():
    with h5py.File(
        precipitation_folder / "hybrid_train_test_2014-2023_input-length_12_img-ahead_6_rain-threshold_50.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        train_nodes = orig_f["train"]["nodes"]
        test_nodes = orig_f["test"]["nodes"]

        filename = (
            precipitation_folder / "hybrid2_kriging_train_test_2014-2023_input-length_12_img-ahead_6_rain-threshold_50.h5"
        )
        #copy images and timestamps, create kriging dataset
        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            orig_f["train"].copy("images", train_set)
            orig_f["train"].copy("timestamps", train_set)
            orig_f["test"].copy("images", test_set)
            orig_f["test"].copy("timestamps", test_set)

            kriging_train_images = np.empty((train_set["images"].shape[0],18,8,K_SIZE,K_SIZE), dtype="float32")
            kriging_test_images = np.empty((test_set["images"].shape[0],18,8,K_SIZE,K_SIZE), dtype="float32")

            for i in tqdm(range(kriging_train_images.shape[0])):
                for j in range(kriging_train_images.shape[2]):
                    input = train_nodes[i][:, j, :]
                    kriging_map = makeKrigeMap(input)
                    kriging_train_images[i,j] = kriging_map
            for i in tqdm(range(kriging_test_images.shape[0])):
                for j in range(kriging_test_images.shape[2]):
                    input = test_nodes[i][:, j, :]
                    kriging_map = makeKrigeMap(input)
                    kriging_test_images[i,j] = kriging_map
            
            train_kriging_dataset = train_set.create_dataset(
                "kriging",
                data = kriging_train_images,
                maxshape=(None, 18, 8,K_SIZE,K_SIZE),
                dtype="float32",
                compression="gzip",
                compression_opts=5,
            )
            test_kriging_dataset = test_set.create_dataset(
                "kriging",
                data = kriging_test_images,
                maxshape=(None, 18, 8,K_SIZE,K_SIZE),
                dtype="float32",
                compression="gzip",
                compression_opts=5,
            )

create_dataset()