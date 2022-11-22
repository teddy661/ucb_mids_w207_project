import numpy as np
import pandas as pd

import db.db_access as dba
from data.path_utils import get_paths

COLOR_SCALE = 1.0
# this turned out not necessary, and having decimals would mess up the calculations later
# as the precision cause 0 to be calculated as very small negative numbers sometimes


def load_data(
    train_val_split=0.8,
    get_clean=False,
    remove_outliers=False,
) -> tuple:

    """
    Load image from the database and transforms into standard numpy arrays.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    _, _, TRAIN_DB_PATH, TEST_DB_PATH, _ = get_paths(remove_outliers)

    np.random.seed(1234)

    sqcon, sqcur = dba.get_con_and_cursor(TRAIN_DB_PATH)
    X, y = dba.get_training_data_as_numpy(sqcur)
    if get_clean:
        X, y = get_clean_data(X, y)

    shuffled_indices = np.random.permutation(range(len(y)))
    train_indices = shuffled_indices[: int(len(y) * train_val_split)]
    val_indices = shuffled_indices[int(len(y) * train_val_split) :]

    X_train = X[train_indices] / COLOR_SCALE
    y_train = y[train_indices]
    X_val = X[val_indices] / COLOR_SCALE
    y_val = y[val_indices]

    dba.dispose(sqcon, sqcur)

    sqcon, sqcur = dba.get_con_and_cursor(TEST_DB_PATH)
    X_test = dba.get_test_data_as_numpy(sqcur) / COLOR_SCALE

    dba.dispose(sqcon, sqcur)

    return X_train, X_val, y_train, y_val, X_test


def get_clean_data(X, y):
    """Remove all images don't have all 30 labels"""

    clean = ~np.isnan(y).any(axis=1)
    return X[clean, :], y[clean, :]
