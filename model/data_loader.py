from pathlib import Path

import numpy as np

import data.path_utils as path_utils
import db.db_access as dba

SCALE = 1.0


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

    X_train = X[train_indices] / SCALE
    y_train = y[train_indices]
    X_val = X[val_indices] / SCALE
    y_val = y[val_indices]

    dba.dispose(sqcon, sqcur)

    sqcon, sqcur = dba.get_con_and_cursor(TEST_DB_PATH)
    X_test = dba.get_test_data_as_numpy(sqcur) / SCALE

    dba.dispose(sqcon, sqcur)

    return X_train, X_val, y_train, y_val, X_test


def load_data_from_file(
    train_data_path: Path,
    test_data_path: Path,
    labels_to_include: list[str],
    train_val_split=0.8,
) -> tuple:

    """
    Load image from the given csv and transforms into standard numpy arrays.
    Only return the Y values that are in labels_to_include.

    return:
    X_train, X_val, y_train, y_val, X_test
    """

    train = pd.read_csv(TRAIN_CSV, encoding="utf8")

    imgs_all = []
    for idx, r in train_only_all_points.iterrows():
        imgs_all.append(
            np.array(r["Image"].split())
            .astype(np.int64)
            .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
        )
    imgs_all = np.array(imgs_all)
    y_all = np.array(train_only_all_points[classes])

    X, y = dba.get_training_data_as_numpy(sqcur)

    shuffled_indices = np.random.permutation(range(len(y)))
    train_indices = shuffled_indices[: int(len(y) * train_val_split)]
    val_indices = shuffled_indices[int(len(y) * train_val_split) :]

    X_train = X[train_indices] / SCALE
    y_train = y[train_indices]
    X_val = X[val_indices] / SCALE
    y_val = y[val_indices]

    X_test = dba.get_test_data_as_numpy(sqcur) / SCALE
    return X_train, X_val, y_train, y_val, X_test

    # TODO: make this work with Y names
    def get_clean_data(X, y):
        clean = ~np.isnan(y).any(axis=1)
        return X[clean, :], y[clean, :]
