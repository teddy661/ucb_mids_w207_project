import os

import tensorflow as tf

import data.data_loader as data_loader
import data.path_utils as path_utils
import model_trainer.model_builder as model_builder
from model_trainer.face_key_point_hyper_model import ALL_LABELS

"""
This is a model that predicts all 15 keypoints using one model. 
"""
MODEL_NAME = "model_assembled"
_, _, MODEL_PATH = path_utils.get_data_paths()
force_train = True

models = {}
for label in ALL_LABELS:
    individual_model_name = f"model_{label}"
    individual_model_path = MODEL_PATH.joinpath(individual_model_name).joinpath(
        individual_model_name
    )
    if not force_train and MODEL_PATH.is_dir(): # load the trained model
        model: tf.keras.Model = tf.keras.models.load_model(individual_model_path)
    else:
        model: tf.keras.Model = model_builder.tune_model(
            labels_to_include=[label], model_name=individual_model_name
        )

    for layer in model.layers:
        layer._name = individual_model_name + "_" + layer.name

    models[label] = model

all_inputs = []
all_outputs = []
for label, model in models.items():
    all_inputs.append(model.input)
    all_outputs.append(model.output)

assembled_model = tf.keras.Model(
    inputs=all_inputs, outputs=all_outputs, name=MODEL_NAME
)

assembled_model.summary()


# test the model with test data
_, TEST_DATA_PATH, MODEL_PATH = path_utils.get_data_paths()
X_test = data_loader.load_test_data_from_file()


# have to use CPU for prediction, given the size of the model
with tf.device("/cpu:0"):
    results = assembled_model.predict(
        [X_test] * 15,
        batch_size=200,
        verbose=1,
    )

model_builder.save_results_and_submit(results, MODEL_NAME)
