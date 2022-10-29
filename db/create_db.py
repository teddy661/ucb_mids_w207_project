import argparse
import io
import os
import sqlite3
from hashlib import blake2b
from pathlib import Path

import pandas as pd
from PIL import Image


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


def calc_hash(input_bytes):
    """
    Calculate the blake2b hash for the png image
    """
    my_hash = blake2b()
    my_hash.update(input_bytes)
    return my_hash.hexdigest()


def create_duplicate_image_view(sqcur):
    sqcur.execute(
        """
            CREATE VIEW dup_images AS
            SELECT
                rowid,
                png_hash
            FROM
                image_data
            WHERE
                png_hash IN (
                    SELECT
                        png_hash
                    FROM
                        image_data
                    GROUP BY
                        png_hash
                    HAVING
                        COUNT(png_hash) > 1)
            ORDER BY
                png_hash
    """
    )


def create_db(data_path, db_path):
    df = pd.read_csv(data_path, encoding="utf8")
    df.rename(columns={"Image": "image_raw_pixels"}, inplace=True)
    df["png_image"] = df["image_raw_pixels"].apply(create_image)
    df["png_hash"] = df["png_image"].apply(calc_hash)

    sqcon = sqlite3.connect(db_path)
    sqcur = sqcon.cursor()

    sqcur.execute("""PRAGMA page_size=32768""")
    sqcur.execute("""PRAGMA temp_store=2""")
    sqcur.execute("""PRAGMA locking_mode=EXCLUSIVE""")
    sqcur.execute("""PRAGMA cache_size=-65536""")
    sqcur.execute("""PRAGMA synchronous = 0""")
    df.to_sql("image_data", sqcon, if_exists="replace", index=True)
    create_duplicate_image_view(sqcon)
    sqcur.execute(
        """CREATE INDEX IF NOT EXISTS idx_image_hash on image_data (png_hash)"""
    )
    sqcur.execute("""ANALYZE""")
    sqcon.commit()
    sqcon.close()


def verify_paths(
    ROOT_DIR, TEST_DATA, TRAIN_DATA, TRAIN_DB, TEST_DB, override: bool = False
):
    if not ROOT_DIR.is_dir():
        print(
            "Terminating, root directory does not exist or is not a directory {0}".format(
                ROOT_DIR
            )
        )
        exit()

    if not TEST_DATA.is_file():
        print(
            "Terminating, Test data csv file doe not exist or is not a file {0}".format(
                TEST_DATA
            )
        )
        exit()

    if not TRAIN_DATA.is_file():
        print(
            "Terminating, Training data csv file doe not exist or is not a file {0}".format(
                TRAIN_DATA
            )
        )
        exit()

    verify_db(TRAIN_DB, override)
    verify_db(TEST_DB, override)


def verify_db(db_path: Path, override: bool = False):

    if override:
        try:
            os.remove(db_path)
        except OSError:
            pass
    elif db_path.is_file() and not override:
        print(
            "Database exists, will not overwrite. Use -f flag to force overwrite {0}".format(
                db_path
            )
        )
        exit()


def main():
    """
    Generate sqlite database from a pandas table.
    :return:
    :rtype:
    """
    parser = argparse.ArgumentParser(
        description="Create an sqlite database from csv files and generate png files from raw image pixels "
    )
    parser.add_argument(
        "-f",
        help="overwrite existing database",
        dest="enable_overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).parent.parent).resolve()

    DATA_DIR = ROOT_DIR.joinpath("data")
    TEST_DATA = DATA_DIR.joinpath("test.csv")
    TRAIN_DATA = DATA_DIR.joinpath("training.csv")

    DB_DIR = ROOT_DIR.joinpath("db")
    if not DB_DIR.exists():
        os.mkdir(DB_DIR)

    TRAIN_DB = DB_DIR.joinpath("training.db")
    TEST_DB = DB_DIR.joinpath("test.db")

    verify_paths(
        ROOT_DIR, TEST_DATA, TRAIN_DATA, TRAIN_DB, TEST_DB, args.enable_overwrite
    )

    create_db(TRAIN_DATA, TRAIN_DB)
    create_db(TEST_DATA, TEST_DB)


if __name__ == "__main__":
    main()
