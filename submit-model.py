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

MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "final-model"
TF_MODEL = MODEL_DIR.joinpath(FINAL_MODEL_NAME)

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


test = pd.read_csv(TEST_CSV, encoding="utf8")
test["png"] = test["Image"].apply(create_png)

imgs_all = []
np.random.seed(1234)
for idx, r in test.iterrows():
    img = tf.keras.preprocessing.image.load_img(
        BytesIO(r["png"]),
        color_mode="grayscale",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    imgs_all.append(img)

test_data = np.array(imgs_all)


model = tf.keras.models.load_model(TF_MODEL)
results = model.predict(test_data, batch_size=16, verbose=1)

# converted = np.array(test.iloc[0]['Image'].split()).astype(np.int64).reshape(IMAGE_WIDTH,IMAGE_HEIGHT)
# print(np.array_equal(imgs_all[0], converted))

# print(converted.shape)
# np.set_printoptions(threshold=sys.maxsize)
# print(imgs_all[0])
