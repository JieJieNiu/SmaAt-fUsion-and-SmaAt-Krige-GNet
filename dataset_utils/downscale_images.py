import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from root import ROOT_DIR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import cv2

precipitation_folder = ROOT_DIR / "data" / "precipitation"

#Downscale images down to 64x64
def create_dataset():
    train_images = None
    test_images = None
    with h5py.File(
        precipitation_folder / "hybrid_kriging_train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50_size_64.h5",
        "r",
    ) as orig_f:
        train_images = orig_f["train"]["images"]
        train_timestamps = orig_f["train"]["timestamps"]
        test_images = orig_f["test"]["images"]
        test_timestamps = orig_f["test"]["timestamps"]
        train_kriging = orig_f["train"]["kriging"]
        test_kriging = orig_f["test"]["kriging"]

        filename = (
            precipitation_folder / "hybrid_kriging_train_test_2016-2019_input-length_12_img-ahead_6_rain-threshold_50_size_64_64.h5"
        )
        #copy
        with h5py.File(filename, "w") as f:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            orig_f["train"].copy("kriging", train_set)
            orig_f["train"].copy("timestamps", train_set)
            orig_f["test"].copy("kriging", test_set)
            orig_f["test"].copy("timestamps", test_set)

            downscaled_train_images = np.empty((5733,18,64,64), dtype="float32")
            downscaled_test_images = np.empty((1557,18,64,64), dtype="float32")
            
            for i in tqdm(range(downscaled_train_images.shape[0])):
                for j in range(downscaled_train_images.shape[1]):
                    image = train_images[i][j]
                    downscaled_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                    downscaled_train_images[i,j] = downscaled_image
            for i in tqdm(range(downscaled_test_images.shape[0])):
                for j in range(downscaled_test_images.shape[1]):
                    image = test_images[i][j]
                    downscaled_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
                    downscaled_test_images[i,j] = downscaled_image

            train_images_dataset = train_set.create_dataset(
                "images",
                data = downscaled_train_images,
                maxshape=(None, 18, 64, 64),
                dtype="float32",
                compression="gzip",
                compression_opts=5,
            )
            test_images_dataset = test_set.create_dataset(
                "images",
                data = downscaled_test_images,
                maxshape=(None, 18, 64, 64),
                dtype="float32",
                compression="gzip",
                compression_opts=5,
            )

                    
create_dataset()