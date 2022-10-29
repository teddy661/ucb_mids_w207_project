import pandas as pd
import numpy as np
import os

print(os.getcwd())

from db.create_db import get_paths
from db.db_access import *


def load_image():
    """
    Load image from the database from the csv file.

    return:
    X_train, X_val, y_train, y_val
    """

    TRAIN_DATA, TEST_DATA, TRAIN_DB, TEST_DB = get_paths()
    sqcon, sqcur = get_db_and_cursor(TRAIN_DB)
    X_train, y_train = get_training_data_as_numpy(sqcur)

    del sqcur
    sqcon.close()

    sqcon, sqcur = get_db_and_cursor(TEST_DB)
    X_test = get_test_data_as_numpy(sqcur)
    
    del sqcur
    sqcon.close()


load_image()
