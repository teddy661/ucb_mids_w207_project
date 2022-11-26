import os
from pathlib import Path


def get_data_paths() -> tuple[Path, Path, Path]:

    ROOT_DIR = (Path(__file__).parent.parent).resolve()
    DATA_DIR = ROOT_DIR.joinpath("data")

    TRAIN_DATA_PATH = DATA_DIR.joinpath("training.csv")
    TEST_DATA_PATH = DATA_DIR.joinpath("test.csv")
    verify_paths(ROOT_DIR, TRAIN_DATA_PATH, TEST_DATA_PATH)

    MODEL_PATH = ROOT_DIR.joinpath("model_saves").resolve()
    if not MODEL_PATH.is_dir():
        MODEL_PATH.mkdir(parents=True, exist_ok=True)

    return TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH


def get_db_paths() -> tuple[Path, Path]:
    ROOT_DIR = (Path(__file__).parent.parent).resolve()

    DB_DIR = ROOT_DIR.joinpath("db")
    if not DB_DIR.exists():
        os.mkdir(DB_DIR)
    TRAIN_DB_PATH = DB_DIR.joinpath("training.db")
    TEST_DB_PATH = DB_DIR.joinpath("test.db")

    return TRAIN_DB_PATH, TEST_DB_PATH


def verify_paths(root_dir, train_data_path, test_data_path):
    if not root_dir.is_dir():
        print(
            "Terminating, root directory does not exist or is not a directory {0}".format(
                root_dir
            )
        )
        exit()

    if not train_data_path.is_file():
        print(
            "Terminating, Training data csv file doe not exist or is not a file {0}".format(
                train_data_path
            )
        )
        exit()

    if not test_data_path.is_file():
        print(
            "Terminating, Test data csv file doe not exist or is not a file {0}".format(
                test_data_path
            )
        )
        exit()
