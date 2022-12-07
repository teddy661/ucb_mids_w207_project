from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from mtcnn import MTCNN

import sys

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("training.csv")
TEST_CSV = DATA_DIR.joinpath("test.csv")

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

if not TRAIN_CSV.is_file():
    print(
        "Terminating, Training data csv file doe not exist or is not a file {0}".format(
            TRAIN_CSV
        )
    )
    exit()

if not TEST_CSV.is_file():
    print(
        "Terminating, Test data csv file doe not exist or is not a file {0}".format(
            TEST_CSV
        )
    )
    exit()


def int_string_to_numpy(int_string):
    return np.fromstring(int_string, dtype=np.uint8, sep=" ").reshape(
        IMAGE_WIDTH, IMAGE_HEIGHT, 1
    )


def crop_image(img, bbox):
    # y=box[1] h=box[3] x=box[0] w=box[2]
    #    biggest=0
    #    if data !=[]:
    #        for faces in data:
    #            box=faces['box']
    #            # calculate the area in the image
    #            area = box[3]  * box[2]
    #            if area>biggest:
    #                biggest=area
    #                bbox=box
    bbox[0] = 0 if bbox[0] < 0 else bbox[0]
    bbox[1] = 0 if bbox[1] < 0 else bbox[1]
    img = img[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    img = cv2.copyMakeBorder(
        img,
        bbox[1],
        IMAGE_HEIGHT - bbox[1] - bbox[3],
        bbox[0],
        IMAGE_WIDTH - bbox[0] - bbox[2],
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
    return img


test_df = pd.read_csv(TEST_CSV, encoding="utf8")
test_df["image_as_np"] = test_df["Image"].apply(int_string_to_numpy)

train_df = pd.read_csv(TRAIN_CSV, encoding="utf8")
classes = train_df.select_dtypes(include=[np.number]).columns
num_classes = len(classes)
train_df["image_as_np"] = train_df["Image"].apply(int_string_to_numpy)

y_all = np.array(train_df[classes])


imgs_all = []
for idx, r in train_df.iterrows():
    imgs_all.append(train_df["image_as_np"])

mtnn_nofaces = train_df.loc[train_df.index[MTNN_NOFACES]]


imgs_bad = mtnn_nofaces["image_as_np"].tolist()
imgs_test = train_df["image_as_np"].tolist()

detector = MTCNN()

nofaces = []
manyfaces = []
for idx, image in enumerate(imgs_test):
    current_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # mtcnn wants color images
    result = detector.detect_faces(current_image)
    numfaces = len(result)
    print(f"{idx}: {numfaces}", file=sys.stderr)
    if numfaces == 0:
        nofaces.append(idx)
    if numfaces > 1:
        manyfaces.append(idx)

print(nofaces, file=sys.stderr)
print(f"Length of nofaces: {len(nofaces)}", file=sys.stderr)
print("=" * 80, file=sys.stderr)
print(manyfaces, file=sys.stderr)
print(f"Length of manyfaces: {len(manyfaces)}", file=sys.stderr)
exit()
bounding_box = result[0]["box"]
keypoints = result[0]["keypoints"]

corrected_bounding_box = []
for x in bounding_box:
    if x > 96:
        x = 96
    if x < 0:
        x = 0
    corrected_bounding_box.append(x)

cv2.rectangle(
    image,
    (corrected_bounding_box[0], corrected_bounding_box[1]),
    (
        corrected_bounding_box[0] + corrected_bounding_box[2],
        corrected_bounding_box[1] + corrected_bounding_box[3],
    ),
    (0, 155, 255),
    2,
)

print(corrected_bounding_box)

image = crop_image(image, corrected_bounding_box)
print(type(image))
print(image.dtype)
print(image.shape)
cv2.circle(image, (keypoints["left_eye"]), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints["right_eye"]), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints["nose"]), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints["mouth_left"]), 2, (0, 155, 255), 2)
cv2.circle(image, (keypoints["mouth_right"]), 2, (0, 155, 255), 2)

cv2.imwrite("test.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(type(image))
print(image.dtype)
print(image.shape)
print(image.flatten())
print(result[0]["confidence"])
