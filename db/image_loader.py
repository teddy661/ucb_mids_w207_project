import io
from hashlib import blake2b

import numpy as np
import pandas as pd
from PIL import Image

IMAGE_SIZE = (96, 96)


def load_image_data(data_path) -> pd.DataFrame:
    """
    Load the data from the csv file into a pandas dataframe
    """

    df = pd.read_csv(data_path, encoding="utf8")
    df.rename(columns={"Image": "image_raw_pixels"}, inplace=True)
    df["png_image"] = df["image_raw_pixels"].apply(create_png)
    df["png_hash"] = df["png_image"].apply(calc_hash)

    return df


def create_image_from_pixels(pixels) -> Image.Image:
    temp_image = Image.new("L", IMAGE_SIZE)
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


def calc_hash(input_bytes):
    """
    Calculate the blake2b hash for the png image
    """
    my_hash = blake2b()
    my_hash.update(input_bytes)
    return my_hash.hexdigest()
