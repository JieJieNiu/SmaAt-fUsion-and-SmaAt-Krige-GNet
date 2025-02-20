import h5py
import numpy as np
from tqdm import tqdm
import os
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import random

month_strings = {
    "01": "JAN",
    "02": "FEB",
    "03": "MAR",
    "04": "APR",
    "05": "MAY",
    "06": "JUN",
    "07": "JUL",
    "08": "AUG",
    "09": "SEP",
    "10": "OCT",
    "11": "NOV",
    "12": "DEC",
}


# Function used to retrieve timestamp from a path
def get_timestamp_from_path(path):
    timestamp = path
    timestamp = timestamp.split("/")
    timestamp = timestamp[len(timestamp) - 1]
    timestamp = timestamp[23:35]
    year = timestamp[0:4]
    month = timestamp[4:6]
    day = timestamp[6:8]
    hour = timestamp[8:10]
    minute = timestamp[10:12]
    string = f"{day}-{month_strings[month]}-{year};{hour}:{minute}:00.000"
    return year, month, day, hour, minute, string


# Precipitation maps contain a large area around the netherlands without data (indicated by 65535 values)
# This function calculates the smallest possible box around the pixels with actual values
def get_bounding_box(arr):
    # Find the rows and columns with non-background values
    non_bg_rows, non_bg_cols = np.where(arr < 65535)

    # Calculate the bounding box coordinates
    min_row = np.min(non_bg_rows)
    min_col = np.min(non_bg_cols)
    max_row = np.max(non_bg_rows)
    max_col = np.max(non_bg_cols)

    return min_row, min_col, max_row + 1, max_col + 1


# Removes invalid pixels and normalizes the image
def clean_images(images, max_value, valid_pixels):
    for i, img in enumerate(images):
        img_normalized = (img * valid_pixels) / max_value
        images[i] = img_normalized
    return images


def clean_data(
    dir,  # Directory with the dataset
    years_to_clean,  # List of strings with the years that need to be cleaned
    day_threshold,  # 24 threshold in mm, values above this threshold are marked as invalid (because they are likely ground echoes)
    year_threshold,  # all year threshold in mm, values above this threshold are marked as invalid (because they are likely ground echoes)
    normalize=True,  # Indicates whether to normalize the data or not
    train=True,  # Indicates whether the specified data is for the train set or not
    valid_pixels=np.array(
        []
    ),  # Initial value for the array which keeps tracks of all valid pixels
    max_value=0,  # Initial value for the maximum pixel value
):
    min_row, min_col, max_row, max_col = (0, 0, 0, 0)
    img_shape = (0, 0)
    orig_img_shape = (0, 0)
    is_first_file = True

    valid_pixels = valid_pixels
    valid_pixel_count = 0
    max_vals = np.array([])
    year_sum = []
    day_sum = []

    images = []
    timestamps = []

    # Loop over all years in the data dir
    years = sorted([f.path for f in os.scandir(dir) if f.is_dir()])
    for year_path in years:
        dir_name = year_path.split("/")[-1]
        # Skip folder if not in specified years
        if dir_name not in years_to_clean:
            continue
        # Loop over the all months
        months = sorted([f.path for f in os.scandir(year_path) if f.is_dir()])
        for month_path in months:
            print(month_path)
            files = sorted(
                [f.path for f in os.scandir(month_path) if f.path.endswith(".h5")]
            )
            for i, file in enumerate(files):
                # Get timestamp
                year, month, day, hour, minute, timestamp = get_timestamp_from_path(
                    file
                )
                # Convert to np array
                h5_file = h5py.File(file)
                data = h5_file.get("image1").get("image_data")
                arr = np.asarray(data)
                # For the first file some initialization is needed
                if is_first_file:
                    # Check if the file is not empty
                    not_empty_rows, not_empty_cols = np.where(arr < 65535)
                    if not_empty_rows.size > 0 and not_empty_cols.size > 0:
                        is_first_file = False
                        min_row, min_col, max_row, max_col = get_bounding_box(arr)
                        img_shape = (max_row - min_row, max_col - min_col)
                        orig_img_shape = arr.shape

                        if train:
                            year_sum = np.zeros(img_shape)
                            day_sum = np.zeros(img_shape)
                            max_vals = np.zeros(img_shape)
                            valid_pixels = np.full(img_shape, True)
                            # count valid values
                            condition = (
                                arr[min_row:max_row, min_col:max_col] < 65535
                            )  # count the elements
                            valid_pixel_count += np.count_nonzero(condition)
                    else:
                        continue

                # If there is no image data
                if arr.ndim < 1:
                    arr = np.zeros(orig_img_shape)
                    print("[WARNING] NO DATA: " + file)
                    
                # Crop
                arr = arr[min_row:max_row, min_col:max_col]
                # Set non pixel values to 0
                arr[arr >= 65535] = 0
                # Convert to float
                arr = arr.astype(np.float32)

                # Update precipitation sums
                if train:
                    arr_mm = arr * 0.01
                    year_sum += arr_mm
                    day_sum += arr_mm
                    # Update max value
                    max_vals = np.maximum(max_vals, arr)
                    # append to train images
                    images.append(arr)
                    timestamps.append([timestamp])

                # append to test images
                images.append(arr)
                timestamps.append([timestamp])

                # End of day
                if hour == "00" and train:
                    # Update valid pixels
                    valid_pixels = np.logical_and(
                        valid_pixels, (day_sum <= day_threshold)
                    )
                    day_sum = np.zeros(img_shape)
        if train:
            # Update valid pixels
            valid_pixels = np.logical_and(valid_pixels, (year_sum <= year_threshold))
            print("Year sum:")
            print(np.max(year_sum))
            # Reset year sum
            year_sum = np.zeros(img_shape)

    if train:
        max_value = np.max((max_vals * valid_pixels))

        # Remove invalid pixels from the valid pixel count
        invalid_pixel_count = (~valid_pixels).sum()
        valid_pixel_count -= invalid_pixel_count
    print(max_value)

    if normalize:
        print("Normalizing images...")
        images = clean_images(images, max_value, valid_pixels)

    return (
        images,
        timestamps,
        valid_pixels,
        valid_pixel_count,
        max_value,
    )


# Function that preprocesses the data and combines it into a single .h5 file
def create_dataset(
    train_images,
    train_timestamps,
    test_images,
    test_timestamps,
    train_years,  # list of years used for the train set
    test_years,  # List of years to be used for the test set
    input_length,  # number of input frames
    image_ahead,  # number of target frames
    rain_amount_thresh,  # threshold in mm/h which indicates rain or no rain
    valid_pixel_count,  # number of valid pixels of a single input image
    append=False,
):
    # Print data shapes
    print("Train shape", len(train_images))
    print("Test shape", len(test_images))
    print("Train shape", len(train_timestamps))
    print("Test shape", len(test_timestamps))
    # Get image shape
    if len(train_images) > 0:
        imgWidth = train_images[0].shape[0]
        imgHeight = train_images[0].shape[1]
    else:
        imgWidth = test_images[0].shape[0]
        imgHeight = test_images[0].shape[1]
    num_pixels = valid_pixel_count

    # File name of new dataset
    filename = f"[DATA_LOCATION]/train_test_{train_years[0]}-{test_years[-1]}_input-length_{input_length}_img-ahead_{image_ahead}_rain-threshhold_{int(rain_amount_thresh * 100)}_normalized.h5"

    with h5py.File(filename, "a", rdcc_nbytes=1024**3) as f:
        train_image_dataset = []
        train_timestamp_dataset = []
        test_image_dataset = []
        train_timestamp_dataset = []

        # Create new datasets
        if not append:
            train_set = f.create_group("train")
            test_set = f.create_group("test")
            train_image_dataset = train_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgWidth, imgHeight),
                maxshape=(None, input_length + image_ahead, imgWidth, imgHeight),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            train_timestamp_dataset = train_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
            test_image_dataset = test_set.create_dataset(
                "images",
                shape=(1, input_length + image_ahead, imgWidth, imgHeight),
                maxshape=(None, input_length + image_ahead, imgWidth, imgHeight),
                dtype="float32",
                compression="gzip",
                compression_opts=9,
            )
            test_timestamp_dataset = test_set.create_dataset(
                "timestamps",
                shape=(1, input_length + image_ahead, 1),
                maxshape=(None, input_length + image_ahead, 1),
                dtype=h5py.special_dtype(vlen=str),
                compression="gzip",
                compression_opts=9,
            )
        # Load existing datasets to append to
        else:
            train_image_dataset = f["train"]["images"]
            train_timestamp_dataset = f["train"]["timestamps"]
            print(len(train_timestamp_dataset))
            test_image_dataset = f["test"]["images"]
            test_timestamp_dataset = f["test"]["timestamps"]

        origin = [[train_images, train_timestamps], [test_images, test_timestamps]]
        datasets = [
            [train_image_dataset, train_timestamp_dataset],
            [test_image_dataset, test_timestamp_dataset],
        ]

        for origin_id, [images, timestamps] in enumerate(origin):
            print(f"id: {origin_id}")
            image_dataset, timestamp_dataset = datasets[origin_id]
            first = True
            count = 0
            # Loop over possible image sequences
            for i in tqdm(range(input_length + image_ahead, len(images))):
                last_imgs = images[i - (image_ahead) : i]
                # Check if the target images have enough rainy pixels
                if (
                    np.sum(np.array(last_imgs) > 0)
                    >= num_pixels * image_ahead * rain_amount_thresh
                ):
                    # Get total sequence
                    imgs = images[i - (input_length + image_ahead) : i]
                    count += 1
                    # Get corresponding timestamps
                    timestamps_img = timestamps[
                        i - (input_length + image_ahead) : i
                    ]
                    # extend the dataset by 1 and add the entry
                    if first:
                        first = False
                    else:
                        image_dataset.resize(image_dataset.shape[0] + 1, axis=0)
                        timestamp_dataset.resize(
                            timestamp_dataset.shape[0] + 1, axis=0
                        )
                    image_dataset[-1] = imgs
                    timestamp_dataset[-1] = timestamps_img

            print(f"total seqs: {count}")


if __name__ == "__main__":
    data_dir = "[DATA_LOCATION]/RAD_NL21_RAC_MFBS_5min"
    train = [str(x) for x in range(1998, 2017)]
    test = [str(x) for x in range(2017, 2023)]
    input_length = 12  # number of input frames for the model (1 frame is 5 min)
    image_ahead = 12  # number of output frames for the model (1 frame is 5 min)
    year_threshold = 1300  # all year threshold in mm, values above this threshold are marked as invalid (because they are likely ground echoes)
    day_threshold = 174  # 24 threshold in mm, values above this threshold are marked as invalid (because they are likely ground echoes)
    rain_amount_thresh = 0.5  # fraction of valid pixels that are required to have rain
    print(len(train))
    print(len(test))

    # Get valid pixels and highest value
    data = clean_data(
        dir=data_dir,
        years_to_clean=train,
        year_threshold=year_threshold,
        day_threshold=day_threshold,
        normalize=False,
        train=True,
    )
    # Unpack tuple
    (
        _,
        _,
        valid_pixels,
        valid_pixel_count,
        highest_value,
    ) = data
    # Create training set
    for i, year in enumerate(train):
        data = clean_data(
            dir=data_dir,
            years_to_clean=[year],
            year_threshold=year_threshold,
            day_threshold=day_threshold,
            normalize=True,
            train=False,
            valid_pixels=valid_pixels,
            max_value=highest_value,
        )
        # unpack tuple
        (
            images,
            timestamps,
            _,
            _,
            _,
        ) = data
        create_dataset(
            images,
            timestamps,
            [],
            [],
            train_years=train,
            test_years=test,
            input_length=input_length,
            image_ahead=image_ahead,
            rain_amount_thresh=rain_amount_thresh,
            valid_pixel_count=valid_pixel_count,
            append=True if i > 0 else False,
        )
    # Create test set
    for year in test:
        data = clean_data(
            dir=data_dir,
            years_to_clean=[year],
            year_threshold=year_threshold,
            day_threshold=day_threshold,
            normalize=True,
            train=False,
            valid_pixels=valid_pixels,
            max_value=highest_value,
        )
        # Unpack tuple
        (
            images,
            timestamps,
            _,
            _,
            _,
        ) = data
        create_dataset(
            [],
            [],
            images,
            timestamps,
            train_years=train,
            test_years=test,
            input_length=input_length,
            image_ahead=image_ahead,
            rain_amount_thresh=rain_amount_thresh,
            valid_pixel_count=valid_pixel_count,
            append=True,
        )
