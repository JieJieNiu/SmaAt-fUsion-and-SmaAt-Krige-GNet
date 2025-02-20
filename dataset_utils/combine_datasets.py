import h5py
import glob
import numpy as np
import cv2
import re
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from torchvision.transforms import CenterCrop
import os

# List all HDF5 files to combine, sorted by filename

max_rain = 0
year_threshold = 1300
day_threshold = 174

test_timestamps = np.empty((210225), dtype=object)
test_images = np.empty((210225, 64, 64), dtype=np.float32)
test_count = 0
train_timestamps = np.empty((837675), dtype=object)
train_images = np.empty((837675, 64, 64), dtype=np.float32)
train_count = 0

bad_ind = pd.read_csv('bad_index6_train.csv')
bad_ind['DTG'] = pd.to_datetime(bad_ind['DTG'])
bad_ind['DTG_str'] = bad_ind['DTG'].dt.strftime("%Y-%m-%d %H:%M")
print(bad_ind['DTG_str'].values)
bad_timestamps = bad_ind['DTG_str'].values
bad_timestamps = set(bad_timestamps)  # Convert to set for O(1) membership checks

bad_ind_test = pd.read_csv('bad_index6_test.csv')
bad_ind_test['DTG'] = pd.to_datetime(bad_ind_test['DTG'])
bad_ind_test['DTG_str'] = bad_ind_test['DTG'].dt.strftime("%Y-%m-%d %H:%M")
print(bad_ind_test['DTG_str'].values)
bad_timestamps_test = bad_ind_test['DTG_str'].values
bad_timestamps_test = set(bad_timestamps_test)

bad_timestamps.update(bad_timestamps_test)

timestamp_pattern = re.compile(r"_(\d{12})_")
max_rains = []
old_timestamp = datetime(2013, 12, 31, 23, 55)
day_sum = []
year_sum = []
img_shape = (64, 64)

dir = "data/precipitation/rain_maps"

years = sorted([f.path for f in os.scandir(dir) if f.is_dir()])
valid_pixels = np.full(img_shape, True)
max_vals = np.zeros(img_shape)

for year_path in years:
    if year_path.split("/")[-1] == '.ipynb_checkpoints' or year_path.split("/")[-1] == 'RAD_NL21_RAC_MFBS_5min':
        continue
    year = int(year_path.split("/")[-1])
    #print(str(year))
    files = files_to_combine = sorted(glob.glob("data/precipitation/rain_maps/" + str(year) +"/*/*.h5"))
    
    year_sum = np.zeros(img_shape)
    for file_path in tqdm(files_to_combine):
        with h5py.File(file_path, "r") as h5_file:
            # Verify the required groups/keys only once at the top
            if 'image1' in h5_file and 'image_data' in h5_file['image1']:
                # Load image data directly and perform necessary processing in place
                image_data = h5_file['image1']['image_data'][...]
                image_data[image_data == 65535] = 0  # Set blank values to zero
    
                # Extract timestamp using the pre-compiled regex pattern
                match = timestamp_pattern.search(file_path)
                if match:
                    # Parse and format timestamp only if it’s not in `bad_timestamps`
                    raw_timestamp = match.group(1)
                    timestamp = datetime.strptime(raw_timestamp, "%Y%m%d%H%M")
                    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M")
                    
                    if timestamp.day != old_timestamp.day:
                        day_sum = np.zeros(img_shape)
    
                    if timestamp_str not in bad_timestamps:
                        # Center crop and find max rain
                        image = image_data[108:172, 125:189]
                        max_value = np.max(image)
                        if max_value < 1500:
                            max_vals = np.maximum(max_vals, image)
                            max_rains.append(max_value)
        
                            # Append to the correct list based on the year
                            if 2014 <= timestamp.year <= 2021:
                                train_images[train_count] = image
                                train_timestamps[train_count] = timestamp_str
                                #print(train_timestamps[train_count])
                                train_count = train_count + 1
                                #print(timestamp_str)

                                #update precipitation sums
                                arr_mm = image * 0.01
                                year_sum += arr_mm
                                day_sum += arr_mm
                                valid_pixels = np.logical_and(valid_pixels, (day_sum <= day_threshold))

                                old_timestamp = timestamp

                            else:
                                test_images[test_count] = image
                                test_timestamps[test_count] = timestamp_str
                                test_count = test_count + 1
                        else:
                            print(f"Rain exceeds 1500 in file: {file_path}")
                    else:
                        print(f"Bad timestamp found: {timestamp}")
                else:
                    print(f"Bad timestamp in file: {file_path}")
            else:
                print(f"Missing 'image_data' in file: {file_path}")
    if 2014 <= year <= 2021:
        valid_pixels = np.logical_and(valid_pixels, (year_sum <= year_threshold))

print("MAX RAIN:") #1498
max_rain = np.max(max_vals)
print(max_rain)
print("Invalid Pixels:")
print(4096 - valid_pixels.sum())
norm = max_rain
#np.save("histogram_data", max_rains)

# Step 2: Create the combined HDF5 file, resize/normalize images, and add timestamps

test_image_data = test_images[:test_count]
train_image_data = train_images[:train_count]

test_image_data = test_image_data / norm
train_image_data = train_image_data / norm

#remove bad pixels
for i, image in enumerate(train_image_data):
    image = valid_pixels * image
    train_image_data[i] = image
for i, image in enumerate(test_image_data):
    image = valid_pixels * image
    test_image_data[i] = image

train_timestamps = train_timestamps[:train_count]
test_timestamps = test_timestamps[:test_count]

with h5py.File("RAD_NL21_PRECIP5.h5", "w") as f:
    imgSize = 64
    train_set = f.create_group("train")
    test_set = f.create_group("test")
    train_image_dataset = train_set.create_dataset(
        "images",
        dtype="float32",
        data = train_image_data,
    )
    test_image_dataset = test_set.create_dataset(
        "images",
        dtype="float32",
        data = test_image_data,
    )
    train_timestamp_dataset = train_set.create_dataset(
        "timestamps",
        dtype=h5py.special_dtype(vlen=str),
        data = train_timestamps
    )
    test_timestamp_dataset = test_set.create_dataset(
        "timestamps",
        dtype=h5py.special_dtype(vlen=str),
        data = test_timestamps
    )
