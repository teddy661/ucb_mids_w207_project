import argparse
import os
import sqlite3
from pathlib import Path

from data.path_utils import get_paths
from db.image_utils import load_image_data


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

    TRAIN_DATA_PATH, TEST_DATA_PATH, TRAIN_DB_PATH, TEST_DB_PATH, _ = get_paths()

    verify_db(TRAIN_DB_PATH, args.enable_overwrite)
    create_db(TRAIN_DATA_PATH, TRAIN_DB_PATH)

    verify_db(TEST_DB_PATH, args.enable_overwrite)
    create_db(TEST_DATA_PATH, TEST_DB_PATH)


if __name__ == "__main__":
    main()
