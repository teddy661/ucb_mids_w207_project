import argparse
import os, io
import sqlite3
from pathlib import Path
from PIL import Image

from db.image_loader import load_image_data


def get_paths() -> tuple:

    ROOT_DIR = (Path(__file__).parent.parent).resolve()
    DATA_DIR = ROOT_DIR.joinpath("data")
    TRAIN_DATA = DATA_DIR.joinpath("training.csv")
    TEST_DATA = DATA_DIR.joinpath("test.csv")
    verify_paths(ROOT_DIR, TRAIN_DATA, TEST_DATA)

    DB_DIR = ROOT_DIR.joinpath("db")
    if not DB_DIR.exists():
        os.mkdir(DB_DIR)

    TRAIN_DB = DB_DIR.joinpath("training.db")
    TEST_DB = DB_DIR.joinpath("test.db")

    return TRAIN_DATA, TEST_DATA, TRAIN_DB, TEST_DB


def verify_paths(ROOT_DIR, TRAIN_DATA, TEST_DATA):
    if not ROOT_DIR.is_dir():
        print(
            "Terminating, root directory does not exist or is not a directory {0}".format(
                ROOT_DIR
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

    if not TEST_DATA.is_file():
        print(
            "Terminating, Test data csv file doe not exist or is not a file {0}".format(
                TEST_DATA
            )
        )
        exit()


def display_image(png_string):
    # im = Image.open(io.BytesIO(df.iloc[0]['png_image']))
    im = Image.open(io.BytesIO(png_string))
    im.show()
    return


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
    df = load_image_data(data_path)

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

    TRAIN_DATA, TEST_DATA, TRAIN_DB, TEST_DB = get_paths()

    verify_db(TRAIN_DB, args.enable_overwrite)
    create_db(TRAIN_DATA, TRAIN_DB)

    verify_db(TEST_DB, args.enable_overwrite)
    create_db(TEST_DATA, TEST_DB)


if __name__ == "__main__":
    main()
