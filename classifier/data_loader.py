from json import load
import pandas as pd
import numpy as np

from db.create_db import get_paths
import db.db_access as dba

COLOR_SCALE = 255.0


def load_data(
    train_val_split=0.8,
) -> tuple:

    """
    Load image from the database and transforms into standard numpy arrays.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    _, _, TRAIN_DB, TEST_DB = get_paths()
    np.random.seed(1234)

    sqcon, sqcur = dba.get_con_and_cursor(TRAIN_DB)
    X, y = dba.get_training_data_as_numpy(sqcur)

    shuffled_indices = np.random.permutation(range(len(y)))
    train_indices = shuffled_indices[: int(len(y) * train_val_split)]
    val_indices = shuffled_indices[int(len(y) * train_val_split) :]

    X_train = X[train_indices] / COLOR_SCALE
    y_train = y[train_indices]
    X_val = X[val_indices] / COLOR_SCALE
    y_val = y[val_indices]

    dba.dispose(sqcon, sqcur)

    sqcon, sqcur = dba.get_con_and_cursor(TEST_DB)
    X_test = dba.get_test_data_as_numpy(sqcur) / COLOR_SCALE

    dba.dispose(sqcon, sqcur)

    return X_train, X_val, y_train, y_val, X_test
