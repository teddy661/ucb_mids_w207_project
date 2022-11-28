import os
import pickle

import keras_tuner as kt
import tensorflow as tf

import data.path_utils as path_utils
import model_trainer.data_loader as data_loader
from model_trainer.face_key_point_hyper_model import FaceKeyPointHyperModel


def tune_model(labels_to_include, model_name) -> tf.keras.Model:
    """
    Tune a modle using the given labels
    """

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    tf.get_logger().setLevel("INFO")
    tf.random.set_seed(1234)

    y_columns_to_include = []
    for lable in labels_to_include:
        y_columns_to_include.append(lable + "_x")
        y_columns_to_include.append(lable + "_y")

    # Data Preprocessing
    TRAIN_DATA_PATH, _, MODEL_PATH = path_utils.get_data_paths()
    X_train, X_val, y_train, y_val = data_loader.load_train_data_from_file(
        TRAIN_DATA_PATH,
        y_columns=y_columns_to_include,
    )

    # Tune the model
    tuner = kt.Hyperband(
        FaceKeyPointHyperModel(labels=labels_to_include, name=model_name),
        objective="val_loss",
        seed=1234,
        max_epochs=100,  # Hyperband automatically picks the best num of epochs
        executions_per_trial=1,  # avergae out the results of 2 trials
        overwrite=True,
        directory=MODEL_PATH.joinpath("tuning"),
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
    best_hp = tuner.get_best_hyperparameters()[0]
    print(best_hp)

    model = tuner.hypermodel.build(best_hp)
    history: tf.keras.callbacks.History = tuner.hypermodel.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        epochs=100,
    )

    # model: tf.keras.Model = tuner.get_best_models(num_models=1)[0]  # this is the best model
    # model.summary()

    with open(MODEL_PATH.joinpath(model_name + "_history"), "wb") as history_file:
        pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

    model.save(MODEL_PATH.joinpath(model_name), overwrite=True)

    return model


def test_and_submit(model: tf.keras.Model, model_name: str):
    """
    save the model to the model directory and submit it to kaggle
    """

    _, TEST_DATA_PATH, MODEL_PATH = path_utils.get_data_paths()
    X_test = data_loader.load_test_data_from_file(TEST_DATA_PATH)

    raise NotImplementedError("TODO: implement this")
