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


train = pd.read_csv(TRAIN_CSV, encoding="utf8")
train["png"] = train["Image"].apply(create_png)


def create_df_from_results(results):
    for idx, output in enumerate(OUTPUTS):
        pass


imgs_all = []
np.random.seed(1234)
for idx, r in train.iterrows():
    img = tf.keras.preprocessing.image.load_img(
        BytesIO(r["png"]),
        color_mode="grayscale",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    imgs_all.append(img)

train_data = np.array(imgs_all)


model = tf.keras.models.load_model(TF_MODEL)
results = model.predict(train_data[:2], batch_size=16, verbose=1)
np_results = np.transpose(np.array(results))
np_results = np.squeeze(np_results)
df = pd.DataFrame(columns=OUTPUTS)
df = df.append(pd.DataFrame(np_results, columns=OUTPUTS), ignore_index=True)
print(df)
# print(np_results.shape)
# np.set_printoptions(threshold=sys.maxsize)
# print(np_results)
# converted = np.array(test.iloc[0]['Image'].split()).astype(np.int64).reshape(IMAGE_WIDTH,IMAGE_HEIGHT)
# print(np.array_equal(imgs_all[0], converted))

# print(converted.shape)
# np.set_printoptions(threshold=sys.maxsize)
# print(imgs_all[0])