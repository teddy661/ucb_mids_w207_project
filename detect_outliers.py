import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

ROOT_DIR = (Path(__file__).parent).resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("training.csv")
PROCESSED_TRAIN_CSV = DATA_DIR.joinpath("processed_training.csv")

OUTPUTS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]
if not TRAIN_CSV.is_file():
    print(
        "Terminating, Test data csv file doe not exist or is not a file {0}".format(
            TRAIN_CSV
        )
    )
    exit()


def int_string_to_numpy(int_string):
    return (
        np.array(int_string.split())
        .astype(np.uint8)
        .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    )


cols = 5


def max_count_of_values(np_array):
    values, counts = np.array(np.unique(np_array, return_counts=True))
    return counts.max()


train_df = pd.read_csv(TRAIN_CSV, encoding="utf8")
train_df["image_as_np"] = train_df["Image"].apply(int_string_to_numpy)
train_df["gray_levels"] = train_df["image_as_np"].apply(lambda x: len(np.unique(x)))
train_df["max_gray_count"] = train_df["image_as_np"].apply(max_count_of_values)
train_df["delta_eye_centers_x"] = (
    train_df["left_eye_center_x"] - train_df["right_eye_center_x"]
)
train_df["delta_eye_centers_y"] = (
    train_df["left_eye_center_y"] - train_df["right_eye_center_y"]
)

print(train_df["max_gray_count"].describe())
lots_of_nothing_images = train_df.loc[train_df["max_gray_count"] >= 1600]
print(lots_of_nothing_images.shape[0])
rows = math.floor(lots_of_nothing_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
i = 1
for index, row in lots_of_nothing_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
# plt.show()
processed_train_df = train_df.drop(train_df[train_df["max_gray_count"] >= 1600].index)

print(train_df["delta_eye_centers_x"].describe())
low_eye_x_distance_images = train_df.loc[train_df["delta_eye_centers_x"] <= 20]
print(low_eye_x_distance_images.shape[0])

rows = math.floor(low_eye_x_distance_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
i = 1
for index, row in low_eye_x_distance_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
# plt.show()
processed_train_df.drop(
    processed_train_df[processed_train_df["delta_eye_centers_x"] <= 20].index,
    inplace=True,
)

print(train_df["delta_eye_centers_y"].describe())
big_eye_y_distance_images = train_df.loc[
    (train_df["delta_eye_centers_y"] <= -12) | (train_df["delta_eye_centers_y"] >= 12)
]
print(big_eye_y_distance_images.shape[0])
rows = math.floor(big_eye_y_distance_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
i = 1
for index, row in big_eye_y_distance_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
# plt.show()


print(train_df["gray_levels"].describe())
low_gray_count_images = train_df.loc[train_df["gray_levels"] <= 110]
print(low_gray_count_images.shape[0])


rows = math.floor(low_gray_count_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
i = 1
for index, row in low_gray_count_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
# plt.show()
processed_train_df.drop(
    processed_train_df[processed_train_df["gray_levels"] <= 110].index, inplace=True
)
processed_train_df.drop(
    columns=[
        "image_as_np",
        "gray_levels",
        "max_gray_count",
        "delta_eye_centers_x",
        "delta_eye_centers_y",
    ],
    inplace=True,
)
processed_train_df.to_csv(PROCESSED_TRAIN_CSV, index=False, encoding="utf-8")
