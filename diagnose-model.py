import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
from PIL import Image
from io import BytesIO
import pandas as pd
import io
import numpy as np
import sys


IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("training.csv")

MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "final-model"
TF_MODEL = MODEL_DIR.joinpath(FINAL_MODEL_NAME)

OUTPUTS = [
    "p_left_eye_center_x",
    "p_left_eye_center_y",
    "p_right_eye_center_x",
    "p_right_eye_center_y",
    "p_left_eye_inner_corner_x",
    "p_left_eye_inner_corner_y",
    "p_left_eye_outer_corner_x",
    "p_left_eye_outer_corner_y",
    "p_right_eye_inner_corner_x",
    "p_right_eye_inner_corner_y",
    "p_right_eye_outer_corner_x",
    "p_right_eye_outer_corner_y",
    "p_left_eyebrow_inner_end_x",
    "p_left_eyebrow_inner_end_y",
    "p_left_eyebrow_outer_end_x",
    "p_left_eyebrow_outer_end_y",
    "p_right_eyebrow_inner_end_x",
    "p_right_eyebrow_inner_end_y",
    "p_right_eyebrow_outer_end_x",
    "p_right_eyebrow_outer_end_y",
    "p_nose_tip_x",
    "p_nose_tip_y",
    "p_mouth_left_corner_x",
    "p_mouth_left_corner_y",
    "p_mouth_right_corner_x",
    "p_mouth_right_corner_y",
    "p_mouth_center_top_lip_x",
    "p_mouth_center_top_lip_y",
    "p_mouth_center_bottom_lip_x",
    "p_mouth_center_bottom_lip_y",
]

if not TRAIN_CSV.is_file():
    print(
        "Terminating, Train data csv file doe not exist or is not a file {0}".format(
            TRAIN_CSV
        )
    )
    exit()

if not TF_MODEL.is_dir():
    print("Terminating, tensorflow model doesn't exist {0}".format(TF_MODEL))
    exit()


def create_image_from_pixels(pixels) -> Image.Image:
    temp_image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT))
    temp_image.putdata([int(x) for x in pixels.split()])

    return temp_image


def create_png(pixel_string):
    """
    Create Images from the integer text lists in the csv file
    """

    temp_image = create_image_from_pixels(pixel_string)

    buf = io.BytesIO()
    temp_image.save(buf, format="PNG")
    png_image = buf.getvalue()
    return png_image


train_df = pd.read_csv(TRAIN_CSV, encoding="utf8")

imgs_all = []
for idx, r in train_df.iterrows():
    imgs_all.append(
        np.array(r["Image"].split())
        .astype(np.int64)
        .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    )

train_np_data = np.array(imgs_all)


model = tf.keras.models.load_model(TF_MODEL)
results = model.predict(train_np_data, batch_size=100, verbose=2)
np_results = np.transpose(np.array(results))
np_results = np.squeeze(np_results)
results_df = pd.DataFrame(np_results, columns=OUTPUTS)
all_data_df = train_df.join(results_df)
all_data_df.drop(columns=["Image"], inplace=True)
all_data_df["delta_left_eye_center_x"] = (
    all_data_df["left_eye_center_x"] - all_data_df["p_left_eye_center_x"]
)
all_data_df["delta_left_eye_center_y"] = (
    all_data_df["left_eye_center_y"] - all_data_df["p_left_eye_center_y"]
)
all_data_df["delta_right_eye_center_x"] = (
    all_data_df["right_eye_center_x"] - all_data_df["p_right_eye_center_x"]
)
all_data_df["delta_right_eye_center_y"] = (
    all_data_df["right_eye_center_y"] - all_data_df["p_right_eye_center_y"]
)
all_data_df["delta_left_eye_inner_corner_x"] = (
    all_data_df["left_eye_inner_corner_x"] - all_data_df["p_left_eye_inner_corner_x"]
)
all_data_df["delta_left_eye_inner_corner_y"] = (
    all_data_df["left_eye_inner_corner_y"] - all_data_df["p_left_eye_inner_corner_y"]
)
all_data_df["delta_left_eye_outer_corner_x"] = (
    all_data_df["left_eye_outer_corner_x"] - all_data_df["p_left_eye_outer_corner_x"]
)
all_data_df["delta_left_eye_outer_corner_y"] = (
    all_data_df["left_eye_outer_corner_y"] - all_data_df["p_left_eye_outer_corner_y"]
)
all_data_df["delta_right_eye_inner_corner_x"] = (
    all_data_df["right_eye_inner_corner_x"] - all_data_df["p_right_eye_inner_corner_x"]
)
all_data_df["delta_right_eye_inner_corner_y"] = (
    all_data_df["right_eye_inner_corner_y"] - all_data_df["p_right_eye_inner_corner_y"]
)
all_data_df["delta_right_eye_outer_corner_x"] = (
    all_data_df["right_eye_outer_corner_x"] - all_data_df["p_right_eye_outer_corner_x"]
)
all_data_df["delta_right_eye_outer_corner_y"] = (
    all_data_df["right_eye_outer_corner_y"] - all_data_df["p_right_eye_outer_corner_y"]
)
all_data_df["delta_left_eyebrow_inner_end_x"] = (
    all_data_df["left_eyebrow_inner_end_x"] - all_data_df["p_left_eyebrow_inner_end_x"]
)
all_data_df["delta_left_eyebrow_inner_end_y"] = (
    all_data_df["left_eyebrow_inner_end_y"] - all_data_df["p_left_eyebrow_inner_end_y"]
)
all_data_df["delta_left_eyebrow_outer_end_x"] = (
    all_data_df["left_eyebrow_outer_end_x"] - all_data_df["p_left_eyebrow_outer_end_x"]
)
all_data_df["delta_left_eyebrow_outer_end_y"] = (
    all_data_df["left_eyebrow_outer_end_y"] - all_data_df["p_left_eyebrow_outer_end_y"]
)
all_data_df["delta_right_eyebrow_inner_end_x"] = (
    all_data_df["right_eyebrow_inner_end_x"]
    - all_data_df["p_right_eyebrow_inner_end_x"]
)
all_data_df["delta_right_eyebrow_inner_end_y"] = (
    all_data_df["right_eyebrow_inner_end_y"]
    - all_data_df["p_right_eyebrow_inner_end_y"]
)
all_data_df["delta_right_eyebrow_outer_end_x"] = (
    all_data_df["right_eyebrow_outer_end_x"]
    - all_data_df["p_right_eyebrow_outer_end_x"]
)
all_data_df["delta_right_eyebrow_outer_end_y"] = (
    all_data_df["right_eyebrow_outer_end_y"]
    - all_data_df["p_right_eyebrow_outer_end_y"]
)
all_data_df["delta_nose_tip_x"] = (
    all_data_df["nose_tip_x"] - all_data_df["p_nose_tip_x"]
)
all_data_df["delta_nose_tip_y"] = (
    all_data_df["nose_tip_y"] - all_data_df["p_nose_tip_y"]
)
all_data_df["delta_mouth_left_corner_x"] = (
    all_data_df["mouth_left_corner_x"] - all_data_df["p_mouth_left_corner_x"]
)
all_data_df["delta_mouth_left_corner_y"] = (
    all_data_df["mouth_left_corner_y"] - all_data_df["p_mouth_left_corner_y"]
)
all_data_df["delta_mouth_right_corner_x"] = (
    all_data_df["mouth_right_corner_x"] - all_data_df["p_mouth_right_corner_x"]
)
all_data_df["delta_mouth_right_corner_y"] = (
    all_data_df["mouth_right_corner_y"] - all_data_df["p_mouth_right_corner_y"]
)
all_data_df["delta_mouth_center_top_lip_x"] = (
    all_data_df["mouth_center_top_lip_x"] - all_data_df["p_mouth_center_top_lip_x"]
)
all_data_df["delta_mouth_center_top_lip_y"] = (
    all_data_df["mouth_center_top_lip_y"] - all_data_df["p_mouth_center_top_lip_y"]
)
all_data_df["delta_mouth_center_bottom_lip_x"] = (
    all_data_df["mouth_center_bottom_lip_x"]
    - all_data_df["p_mouth_center_bottom_lip_x"]
)
all_data_df["delta_mouth_center_bottom_lip_y"] = (
    all_data_df["mouth_center_bottom_lip_y"]
    - all_data_df["p_mouth_center_bottom_lip_y"]
)

all_data_df.to_csv("train_compare.csv", index=True, encoding="utf-8")
