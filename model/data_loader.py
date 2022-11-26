from pathlib import Path

import numpy as np
import pandas as pd

import data.path_utils as path_utils
import db.db_access as dba

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


def load_data_from_db(
    train_val_split=0.8,
) -> tuple:

    """
    Load image from the database and transforms into standard numpy arrays.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    TRAIN_DB_PATH, TEST_DB_PATH, _ = path_utils.get_db_paths()
    np.random.seed(1234)

    sqcon, sqcur = dba.get_con_and_cursor(TRAIN_DB_PATH)
    X, y = dba.get_training_data_as_numpy(sqcur)

    shuffled_indices = np.random.permutation(range(len(y)))
    train_indices = shuffled_indices[: int(len(y) * train_val_split)]
    val_indices = shuffled_indices[int(len(y) * train_val_split) :]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    dba.dispose(sqcon, sqcur)

    sqcon, sqcur = dba.get_con_and_cursor(TEST_DB_PATH)
    X_test = dba.get_test_data_as_numpy(sqcur)

    dba.dispose(sqcon, sqcur)

    return X_train, X_val, y_train, y_val, X_test


def load_data_from_file(
    train_data_path: Path,
    test_data_path: Path,
    y_columns: list[str],
    train_val_split=0.8,
) -> tuple:

    """
    Load image from the given csv and transforms into standard numpy arrays.
    Only return the Y values that are in labels_to_include.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    train_raw = pd.read_csv(train_data_path, encoding="utf8")
    train_clean = clean_up_data(train_raw, y_columns)

    # todo: better way of doing this
    train_x_array = []
    for idx, r in train_clean.iterrows():
        train_x_array.append(
            np.array(r["Image"].split())
            .astype(np.int64)
            .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        )

    train_x_array = np.array(train_x_array)
    train_y_array = np.array(train_clean[y_columns].values)

    # shuffle and split
    shuffled_indices = np.random.permutation(range(len(train_y_array)))
    train_indices = shuffled_indices[: int(len(train_y_array) * train_val_split)]
    val_indices = shuffled_indices[int(len(train_y_array) * train_val_split) :]

    X_train = train_x_array[train_indices]
    y_train = train_y_array[train_indices]
    X_val = train_x_array[val_indices]
    y_val = train_y_array[val_indices]

    test_raw = pd.read_csv(test_data_path, encoding="utf8")
    test_clean = clean_up_data(test_raw, y_columns)

    # todo: better way of doing this
    test_x_array = []
    for idx, r in test_clean.iterrows():
        test_x_array.append(
            np.array(r["Image"].split())
            .astype(np.int64)
            .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        )
    X_test = np.array(test_x_array)

    return X_train, X_val, y_train, y_val, X_test


def clean_up_data(raw: pd.DataFrame, y_columns: list[str]) -> pd.DataFrame:
    """
    Filter only for columns we care about, then drop NA
    """

    columns_to_include = ["Image"] + y_columns
    raw_filtered = raw[raw.columns.intersection(columns_to_include)]
    clean = raw_filtered.dropna()

    return clean
