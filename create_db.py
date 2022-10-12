"import os
import io
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from hashlib import blake2b

ROOT_DIR = Path(
    r"C:\Users\teddy\Documents\01-Berkeley\W207\facial-keypoints-detection"
)
TEST_DATA = ROOT_DIR.joinpath("data","test.csv")
TRAIN_DATA = ROOT_DIR.joinpath("data", "training.csv")
TRAIN_DB = ROOT_DIR.joinpath("db", "training.db")


def create_image(pixel_string):
    """
    Create Images from the integer text lists in the csv file
    """
    image_size = (96, 96)
    temp_image = Image.new("L", image_size)
    temp_image.putdata([int(x) for x in pixel_string.split()])
    buf = io.BytesIO()
    temp_image.save(buf, format="PNG")
    png_image = buf.getvalue()
    return png_image


def display_image(png_string):
    # im = Image.open(io.BytesIO(df.iloc[0]['png_image']))
    im = Image.open(io.BytesIO(png_string))
    im.show()
    return


def calc_hash(input):
    """
    Calculate the blake2b hash for the png image
    """
    myhash = blake2b()
    myhash.update(input)
    return myhash.hexdigest()


def main():
    """
    Generate an sqlite database from a pandas table.
    :return:
    :rtype:
    """
    df = pd.read_csv(TRAIN_DATA, encoding="utf8")
    df["png_image"] = df["Image"].apply(create_image)
    df["png_hash"] = df["png_image"].apply(calc_hash)
    sqcon = sqlite3.connect(TRAIN_DB)
    sqcur = sqcon.cursor()

    sqcur.execute("""PRAGMA page_size=32768""")
    sqcur.execute("""PRAGMA temp_store=2""")
    sqcur.execute("""PRAGMA locking_mode=EXCLUSIVE""")
    sqcur.execute("""PRAGMA cache_size=-65536""")
    # sqcur.execute("""PRAGMA synchronous=OFF""")
    # sqcur.execute("""PRAGMA journal_mode=MEMORY""")
    # sqcur.execute(
    #     """CREATE TABLE IF NOT EXISTS training_data(
    #                     image_id INTEGER PRIMARY KEY,
    #                     left_eye_center_x REAL,
    #                     left_eye_center_y REAL,
    #                     right_eye_center_x REAL,
    #                     right_eye_center_y REAL,
    #                     left_eye_inner_corner_x REAL,
    #                     left_eye_inner_corner_y REAL,
    #                     left_eye_outer_corner_x REAL,
    #                     left_eye_outer_corner_y REAL,
    #                     right_eye_inner_corner_x REAL,
    #                     right_eye_inner_corner_y REAL,
    #                     right_eye_outer_corner_x REAL,
    #                     right_eye_outer_corner_y REAL,
    #                     left_eyebrow_inner_end_x REAL,
    #                     left_eyebrow_inner_end_y REAL,
    #                     left_eyebrow_outer_end_x REAL,
    #                     left_eyebrow_outer_end_y REAL,
    #                     right_eyebrow_inner_end_x REAL,
    #                     right_eyebrow_inner_end_y REAL,
    #                     right_eyebrow_outer_end_x REAL,
    #                     right_eyebrow_outer_end_y REAL,
    #                     nose_tip_x REAL,
    #                     nose_tip_y REAL,
    #                     mouth_left_corner_x REAL,
    #                     mouth_left_corner_y REAL,
    #                     mouth_right_corner_x REAL,
    #                     mouth_right_corner_y REAL,
    #                     mouth_center_top_lip_x REAL,
    #                     mouth_center_top_lip_y REAL,
    #                     mouth_center_bottom_lip_x REAL,
    #                     mouth_center_bottom_lip_y REAL,
    #                     image BLOB)"""
    # )
    df.to_sql("training_data", sqcon, if_exists="replace", index=True)
    sqcon.commit()
    sqcon.close()


if __name__ == "__main__":
    main()
