import numpy as np
import tensorflow as tf

import data.data_loader as data_loader
import data.path_utils as path_utils
import model_trainer.model_builder as model_builder
from model_trainer.face_key_point_hyper_model import ALL_LABELS

force_train = False

_, _, MODEL_PATH = path_utils.get_data_paths()
"""
This is a model that predicts 4 keypoints using one model, and then use the outputs to predict the other 11. 
"""

# train or load the stage one model
model_name_stage_one = "model_stage_one"
labels_first_stage = [
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "mouth_center_bottom_lip",
]
stage_one_path = MODEL_PATH.joinpath(model_name_stage_one).joinpath(
    model_name_stage_one
)
if not force_train and stage_one_path.is_dir():  # load the trained model
    model_stage_one: tf.keras.Model = tf.keras.models.load_model(stage_one_path)
else:
    model_stage_one: tf.keras.Model = model_builder.tune_model(
        labels_to_include=labels_first_stage, model_name=model_name_stage_one
    )

labels_second_stage = [
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
]

MODEL_NAME = "model_stage_two"
stage_two_path = MODEL_PATH.joinpath(MODEL_NAME).joinpath(MODEL_NAME)
if False and not force_train and stage_two_path.is_dir():  # load the trained model
    model_stage_two: tf.keras.Model = tf.keras.models.load_model(stage_two_path)
else:
    model_stage_two = model_builder.tune_two_stage_model(
        labels_to_include=labels_second_stage,
        model_name=MODEL_NAME,
        model_stage_one=model_stage_one,
    )

X_test = data_loader.load_test_data_from_file()
results_stage_1 = model_stage_one.predict(X_test, batch_size=32, verbose=1)
array_stage_1 = np.asarray(results_stage_1).reshape(8, -1).transpose()
results_stage_2 = model_stage_two.predict(
    [array_stage_1, X_test], batch_size=32, verbose=1
)

results = []
j1 = 0
j2 = 0
for i, label in enumerate(ALL_LABELS):
    if label in labels_first_stage:
        for j in range(2):
            results.append(results_stage_1[j1])
            j1 += 1
    else:
        for j in range(2):
            results.append(results_stage_2[j2])
            j2 += 1

model_builder.save_results_and_submit(results, model_name=MODEL_NAME)
