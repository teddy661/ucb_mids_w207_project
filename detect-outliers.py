import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
IMAGE_LIMIT = 400
IMAGE_PER_ROW = 10

ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("training.csv")
PROCESSED_TRAIN_CSV = DATA_DIR.joinpath("processed_training_mtcnn.csv")
MTNN_NOFACES = [
    6,
    63,
    72,
    73,
    120,
    127,
    140,
    144,
    149,
    156,
    172,
    206,
    239,
    263,
    359,
    392,
    405,
    411,
    418,
    564,
    585,
    596,
    605,
    622,
    670,
    746,
    757,
    763,
    768,
    805,
    810,
    817,
    819,
    838,
    859,
    873,
    980,
    1071,
    1133,
    1233,
    1234,
    1319,
    1361,
    1379,
    1408,
    1429,
    1571,
    1585,
    1636,
    1642,
    1643,
    1654,
    1708,
    1731,
    1758,
    1772,
    1780,
    1795,
    1808,
    1876,
    1878,
    1881,
    1883,
    1886,
    1913,
    1914,
    1916,
    1927,
    1929,
    1940,
    1946,
    1953,
    1966,
    1967,
    1987,
    2015,
    2062,
    2063,
    2075,
    2077,
    2080,
    2086,
    2088,
    2093,
    2099,
    2101,
    2108,
    2113,
    2117,
    2119,
    2181,
    2193,
    2201,
    2202,
    2219,
    2221,
    2241,
    2257,
    2288,
    2289,
    2330,
    2333,
    2354,
    2362,
    2404,
    2430,
    2444,
    2449,
    2453,
    2459,
    2484,
    2492,
    2509,
    2532,
    2551,
    2565,
    2572,
    2573,
    2606,
    2616,
    2627,
    2643,
    2655,
    2676,
    2679,
    2683,
    2685,
    2686,
    2694,
    2702,
    2724,
    2738,
    2756,
    2758,
    2768,
    2782,
    2783,
    2787,
    2789,
    2791,
    2835,
    2840,
    2880,
    2890,
    2909,
    2926,
    2936,
    2962,
    2967,
    2969,
    3025,
    3056,
    3066,
    3089,
    3127,
    3137,
    3173,
    3179,
    3205,
    3257,
    3269,
    3280,
    3287,
    3295,
    3296,
    3297,
    3306,
    3315,
    3323,
    3342,
    3346,
    3359,
    3368,
    3375,
    3414,
    3429,
    3447,
    3527,
    3530,
    3532,
    3544,
    3549,
    3556,
    3591,
    3596,
    3609,
    3647,
    3653,
    3666,
    3669,
    3680,
    3697,
    3704,
    3766,
    3776,
    3786,
    3806,
    3809,
    3813,
    3815,
    3840,
    3860,
    3887,
    3903,
    3924,
    3940,
    3943,
    3971,
    3987,
    4006,
    4014,
    4015,
    4024,
    4026,
    4043,
    4050,
    4053,
    4099,
    4157,
    4180,
    4208,
    4262,
    4284,
    4288,
    4312,
    4318,
    4339,
    4358,
    4371,
    4454,
    4459,
    4460,
    4488,
    4496,
    4501,
    4551,
    4571,
    4577,
    4587,
    4634,
    4635,
    4656,
    4664,
    4668,
    4673,
    4695,
    4715,
    4733,
    4740,
    4819,
    4841,
    4887,
    4904,
    4913,
    4917,
    4925,
    4940,
    4943,
    4999,
    5013,
    5062,
    5063,
    5092,
    5098,
    5101,
    5149,
    5156,
    5167,
    5179,
    5180,
    5198,
    5199,
    5206,
    5229,
    5231,
    5240,
    5267,
    5273,
    5332,
    5443,
    5459,
    5504,
    5512,
    5520,
    5565,
    5582,
    5614,
    5616,
    5645,
    5653,
    5660,
    5667,
    5680,
    5685,
    5707,
    5745,
    5776,
    5786,
    5789,
    5793,
    5810,
    5830,
    5831,
    5890,
    5914,
    5943,
    5957,
    5985,
    5996,
    6020,
    6034,
    6053,
    6137,
    6156,
    6207,
    6217,
    6228,
    6233,
    6270,
    6271,
    6278,
    6295,
    6300,
    6322,
    6347,
    6357,
    6371,
    6385,
    6394,
    6405,
    6415,
    6434,
    6435,
    6478,
    6492,
    6493,
    6496,
    6505,
    6507,
    6547,
    6559,
    6568,
    6569,
    6577,
    6585,
    6603,
    6604,
    6615,
    6782,
    6799,
    6805,
    6820,
    6824,
    6844,
    6859,
    6861,
    6873,
    6879,
    6889,
    6902,
    6926,
    6962,
    6964,
    6978,
    6992,
    6993,
    7004,
    7011,
    7021,
    7023,
    7035,
]

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
    return np.fromstring(int_string, dtype=np.uint8, sep=" ").reshape(
        IMAGE_WIDTH, IMAGE_HEIGHT, 1
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

mtnn_nofaces = train_df.loc[train_df.index[MTNN_NOFACES]]
mtnn_numpy = mtnn_nofaces["image_as_np"]
unique_images_annotated = mtnn_numpy.tolist()

num_rows = math.ceil(IMAGE_LIMIT / IMAGE_PER_ROW)
dst = Image.new("RGB", (96 * IMAGE_PER_ROW, 96 * num_rows))
i = 0
print(f"Build composite image for top {IMAGE_LIMIT} images...")
for x in range(IMAGE_PER_ROW):
    for y in range(num_rows):
        if i == len(unique_images_annotated):
            break
        c_image = Image.fromarray(np.squeeze(unique_images_annotated[i]))
        dst.paste(c_image, (x * 96, y * 96))
        i = i + 1

print("No faces Save composite image...")
dst.save("nofaces.png", format="png", optimize=True)
#processed_train_df = train_df.drop(train_df.index[MTNN_NOFACES])
processed_train_df = train_df

print(train_df["max_gray_count"].describe())
lots_of_nothing_images = train_df.loc[train_df["max_gray_count"] >= 1550]
print(lots_of_nothing_images.shape[0])
rows = math.floor(lots_of_nothing_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Number of Pixels with Same Gray Level Greater Than 1550")

i = 1
for index, row in lots_of_nothing_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
plt.show()
processed_train_df.drop(
    processed_train_df[processed_train_df["max_gray_count"] >= 1550].index, inplace=True
)

print(train_df["delta_eye_centers_x"].describe())
low_eye_x_distance_images = train_df.loc[train_df["delta_eye_centers_x"] <= 20]
print(low_eye_x_distance_images.shape[0])

rows = math.floor(low_eye_x_distance_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Eye Center X Distance Less than Twenty")

i = 1
for index, row in low_eye_x_distance_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
plt.show()
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
#plt.show()


print(train_df["gray_levels"].describe())
low_gray_count_images = train_df.loc[train_df["gray_levels"] <= 110]
print(low_gray_count_images.shape[0])


rows = math.floor(low_gray_count_images.shape[0] / (cols - 1))
fig = plt.figure(figsize=(8, 8))
fig.suptitle("Number of Gray Levels Less Than 110")
i = 1
for index, row in low_gray_count_images.iterrows():
    fig.add_subplot(rows, cols, i)
    img = plt.imshow(row["image_as_np"], cmap="gray", vmin=0, vmax=255)
    i += 1
plt.show()

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
