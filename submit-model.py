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
TEST_CSV = DATA_DIR.joinpath("test.csv")
ID_LOOKUP_TABLE = DATA_DIR.joinpath("IdLookupTable.csv")
MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "final-model"
TF_MODEL = MODEL_DIR.joinpath(FINAL_MODEL_NAME)

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
if not ID_LOOKUP_TABLE.is_file():
    print(
        "Terminating, ID_Lookup_table  csv file doe not exist or is not a file {0}".format(
            ID_LOOKUP_TABLE
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


test_df = pd.read_csv(TEST_CSV, encoding="utf8")
test_df["png"] = test_df["Image"].apply(create_png)

imgs_all = []
np.random.seed(1234)
for idx, r in test_df.iterrows():
    img = tf.keras.preprocessing.image.load_img(
        BytesIO(r["png"]),
        color_mode="grayscale",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    imgs_all.append(img)

test_np_data = np.array(imgs_all)


model = tf.keras.models.load_model(TF_MODEL)
results = model.predict(test_np_data, batch_size=100, verbose=2)
np_results = np.transpose(np.array(results))
np_results = np.squeeze(np_results)
results_df = pd.DataFrame(np_results, columns=OUTPUTS)
results_df.index += 1
print(results_df.head())
results_df.to_csv("test_results.csv", index=False, encoding="utf-8")

reformatted_results = []
for index, row in results_df.iterrows():
    row_df = row.to_frame()
    row_df.rename(columns={index: "Location"}, inplace=True)
    row_df.reset_index(inplace=True)
    row_df.rename(columns={"index": "FeatureName"}, inplace=True)
    row_df.insert(0, "ImageId", index)
    reformatted_results.append(row_df)
reformatted_results_df = pd.concat(reformatted_results, ignore_index=True)

id_lookup_df = pd.read_csv(ID_LOOKUP_TABLE, encoding="utf8")

submission_df = pd.merge(
    id_lookup_df, reformatted_results_df, how="left", on=["ImageId", "FeatureName"]
)
submission_df.drop(columns=["Location_x"], inplace=True)
submission_df.rename(columns={"Location_y": "Location"}, inplace=True)
submission_df.to_csv("submission.csv", index=False, encoding="utf-8")
