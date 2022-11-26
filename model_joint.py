import os
import pickle

import keras_tuner as kt
import tensorflow as tf

import data.path_utils as path_utils
import model.data_loader as data_loader
from model.model_tuner import FaceKeyPointModelTuner

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("INFO")
tf.random.set_seed(1234)

MODEL_NAME = "model_joint"
"""
This is a model that predicts all 15 keypoints using one model. 
"""
LABELS_TO_INCLUDE = [
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
    "mouth_center_bottom_lip",
]

_, _, MODEL_PATH = path_utils.get_data_paths()

## Data Preprocessing

TRAIN_DATA_PATH, TEST_DATA_PATH, MODEL_PATH = path_utils.get_data_paths()

X_train, X_val, y_train, y_val, X_test = data_loader.load_data_from_file(
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    labels_to_include=LABELS_TO_INCLUDE,
    get_clean=True,
)

tuner = kt.Hyperband(
    FaceKeyPointModelTuner(labels=LABELS_TO_INCLUDE, name=MODEL_NAME),
    objective="val_loss",
    max_trials=5,
    executions_per_trial=2,
    overwrite=True,
    directory=MODEL_PATH.joinpath("tuning"),
    project_name=MODEL_NAME + "_tuning",
)

tuner.search_space_summary()

tuner.search(
    X_train,
    y_train,
    epochs=1,
    validation_data=(X_val, y_val),
    verbose=2,
)

tuner.results_summary()

# get the best hyperparameters, and re-train the model
best_hp = tuner.get_best_hyperparameters()[0].values
model = tuner.hypermodel.build(best_hp)
history: tf.keras.callbacks.History = tuner.hypermodel.fit(
    X_train, y_train, epochs=100, validation_data=(X_val, y_val)
)

# model: tf.keras.Model = tuner.get_best_models(num_models=1)[0]  # this is the best model
# model.summary()

with open(MODEL_PATH.joinpath(MODEL_NAME + "_history"), "wb") as history_file:
    pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

model.save(MODEL_PATH.joinpath(MODEL_NAME), overwrite=True)
