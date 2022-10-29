import pandas as pd
import numpy as np

from db.create_db import get_paths
import db.db_access as dba


def load_image():
    """
    Load image from the database from the csv file.

    return:
    X_train, X_val, y_train, y_val
    """

    TRAIN_DATA, TEST_DATA, TRAIN_DB, TEST_DB = get_paths()
    sqcon, sqcur = dba.get_db_and_cursor(TRAIN_DB)
    X_train, y_train = dba.get_training_data_as_numpy(sqcur)

    del sqcur
    sqcon.close()

    sqcon, sqcur = dba.get_db_and_cursor(TEST_DB)
    X_test = dba.get_test_data_as_numpy(sqcur)

    del sqcur
    sqcon.close()


load_image()
