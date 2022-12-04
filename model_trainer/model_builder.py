import json
import os
import pickle

import keras_tuner as kt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

import data.path_utils as path_utils
import model_trainer.data_loader as data_loader
from model_trainer.face_key_point_hyper_model import FaceKeyPointHyperModel

ALL_Y_COLUMNS = [
    "left_eye_center_x",
    "left_eye_center_y",
    "right_eye_center_x",
    "right_eye_center_y",
    "left_eye_inner_corner_x",
    "left_eye_inner_corner_y",
    "left_eye_outer_corner_x",
    "left_eye_outer_corner_y",
    "right_eye_inner_corner_x",
    "right_eye_inner_corner_y",
    "right_eye_outer_corner_x",
    "right_eye_outer_corner_y",
    "left_eyebrow_inner_end_x",
    "left_eyebrow_inner_end_y",
    "left_eyebrow_outer_end_x",
    "left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x",
    "right_eyebrow_inner_end_y",
    "right_eyebrow_outer_end_x",
    "right_eyebrow_outer_end_y",
    "nose_tip_x",
    "nose_tip_y",
    "mouth_left_corner_x",
    "mouth_left_corner_y",
    "mouth_right_corner_x",
    "mouth_right_corner_y",
    "mouth_center_top_lip_x",
    "mouth_center_top_lip_y",
    "mouth_center_bottom_lip_x",
    "mouth_center_bottom_lip_y",
]


def tune_model(labels_to_include, model_name) -> tf.keras.Model:
    """
    Tune a modle using the given labels
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("INFO")
    tf.random.set_seed(1234)

    hm = FaceKeyPointHyperModel(labels=labels_to_include, name=model_name)

    # Data Preprocessing
    _, _, MODEL_PATH = path_utils.get_data_paths()
    current_path = MODEL_PATH.joinpath(model_name)

    X_train, X_val, y_train, y_val = data_loader.load_train_data_from_file(
        y_columns=hm.get_column_names()
    )
    y_train = hm.convert_y_to_outputs(y_train)
    y_val = hm.convert_y_to_outputs(y_val)

    tuner = kt.Hyperband(
        hm,
        objective="val_loss",
        seed=1234,
        max_epochs=100,  # Hyperband automatically picks the best num of epochs, but this is the max you allow
        executions_per_trial=2,  # avergae out the results of 2 trials
        overwrite=True,
        directory=current_path,
        project_name=model_name + "_tuning",
    )

    tuner.search_space_summary()

    tuner.search(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        verbose=1,
    )

    # get the best hyperparameters, and re-train the model
    best_hp = next(iter(tuner.get_best_hyperparameters()), None)

    if best_hp is not None:
        with open(
            current_path.joinpath(model_name + "_best_hp.json"), "w"
        ) as best_hp_file:
            json.dump(best_hp.values, best_hp_file)
    else:
        best_hp = kt.HyperParameter("default")

    model: tf.keras.Model = tuner.hypermodel.build(best_hp)
    history: tf.keras.callbacks.History = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,  # best_hp.values["batch_size"],
        epochs=200,
        validation_data=(X_val, y_val),
        callbacks=hm.get_callbacks(hp=best_hp),
    )

    with open(current_path.joinpath(model_name + "_history"), "wb") as best_hp_file:
        pickle.dump(history.history, best_hp_file, protocol=pickle.HIGHEST_PROTOCOL)

    plot_model_history(model_name, history.history)

    model.save(current_path.joinpath(model_name), overwrite=True)
    model.summary()

    return model


def save_results_and_submit(results: np.ndarray, model_name: str):
    """
    save the model to the model directory and submit it to kaggle
    """

    _, TEST_DATA_PATH, MODEL_PATH = path_utils.get_data_paths()
    current_model_path = MODEL_PATH.joinpath(model_name)

    np_results = np.transpose(np.array(results))
    np_results = np.squeeze(np_results)
    np_results = np_results.reshape((-1, 30), order="F")

    results_df = pd.DataFrame(np_results, columns=ALL_Y_COLUMNS)
    results_df.index += 1
    # results_df.to_csv(
    #     current_model_path.joinpath(f"{model_name}_test_results.csv"),
    #     index=False,
    #     encoding="utf-8",
    # )

    reformatted_results = []
    for index, row in results_df.iterrows():
        row_df = row.to_frame()
        row_df.rename(columns={index: "Location"}, inplace=True)
        row_df.reset_index(inplace=True)
        row_df.rename(columns={"index": "FeatureName"}, inplace=True)
        row_df.insert(0, "ImageId", index)
        reformatted_results.append(row_df)
    reformatted_results_df = pd.concat(reformatted_results, ignore_index=True)

    ID_LOOKUP_TABLE = TEST_DATA_PATH.parent.joinpath("IdLookupTable.csv")
    id_lookup_df = pd.read_csv(ID_LOOKUP_TABLE, encoding="utf8")

    submission_df = pd.merge(
        id_lookup_df, reformatted_results_df, how="left", on=["ImageId", "FeatureName"]
    )
    submission_df.drop(columns=["ImageId", "FeatureName", "Location_x"], inplace=True)
    submission_df.rename(columns={"Location_y": "Location"}, inplace=True)
    submission_df.to_csv(
        current_model_path.joinpath(model_name + "_submission.csv"),
        index=False,
        encoding="utf-8",
    )


def plot_model_history(model_name: str, history: dict = None):
    """
    Plot the given history
    """

    if history is None:
        _, _, MODEL_PATH = path_utils.get_data_paths()
        current_model_path = MODEL_PATH.joinpath(model_name)
        with open(
            current_model_path.joinpath(model_name + "_history"), "rb"
        ) as history_file:
            history = pickle.load(history_file)

    x_arr = np.arange(len(history["loss"])) + 1

    fig = plt.figure(figsize=(12, 4))
    fig.suptitle(model_name)
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, history["loss"][:-30], "-o", label="Train loss")
    ax.plot(x_arr, history["val_loss"][:-30], "--<", label="Validation loss")
    ax.legend(fontsize=15)
    ax.set_xlabel("Epoch", size=15)
    ax.set_ylabel("Loss", size=15)

    print(history["val_loss"][:-10])
    plt.show()

    print("done")
