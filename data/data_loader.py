import numpy as np
import pandas as pd

import data.data_augment as data_augment
import data.path_utils as path_utils
import db.db_access as dba

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96


def load_train_data_from_file(
    y_columns: list,
    train_val_split=0.8,
) -> tuple:

    """
    Load image from the given csv and transforms into standard numpy arrays.
    Only return the Y values that are in labels_to_include.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    TRAIN_DATA_PATH, _, _ = path_utils.get_data_paths()
    train_raw = pd.read_csv(TRAIN_DATA_PATH, encoding="utf8")
    train_clean = clean_up_data(train_raw, y_columns)

    train_x_array = np.stack(
        [
            np.asarray(p.split(), dtype="float32").reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
            for p in train_clean["Image"]
        ]
    )
    train_y_array = np.asarray(train_clean[y_columns].values)

    # augment data
    train_x_array, train_y_array = data_augment.augment_data(
        train_x_array, train_y_array
    )

    # shuffle and split
    shuffled_indices = np.random.permutation(range(len(train_y_array)))
    train_indices = shuffled_indices[: int(len(train_y_array) * train_val_split)]
    val_indices = shuffled_indices[int(len(train_y_array) * train_val_split) :]

    X_train = train_x_array[train_indices]
    y_train = train_y_array[train_indices]
    X_val = train_x_array[val_indices]
    y_val = train_y_array[val_indices]

    return X_train, X_val, y_train, y_val


def load_test_data_from_file() -> tuple:
    """
    Load test image from the given csv and transforms into standard numpy arrays.
    """

    _, TEST_DB_PATH, _ = path_utils.get_data_paths()
    test_raw = pd.read_csv(TEST_DB_PATH, encoding="utf8")
    X_test = np.stack(
        [
            np.asarray(p.split(), dtype="float32").reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
            for p in test_raw["Image"]
        ]
    )

    return X_test


def clean_up_data(raw: pd.DataFrame, y_columns: list) -> pd.DataFrame:
    """
    Filter only for columns we care about, then drop NA
    """

    columns_to_include = y_columns + ["Image"]
    raw_filtered = raw[raw.columns.intersection(columns_to_include)]
    clean = raw_filtered.dropna()

    return clean


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
