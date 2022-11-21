import pickle
from pathlib import Path

import keras_tuner
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from data.path_utils import get_paths

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = 64

TRAIN_DATA_PATH, TEST_DATA_PATH, _, _, MODEL_PATH = get_paths()
FINAL_MODEL_NAME = "final-model"
Y_COLUMN_NAMES = [
    "left_eye_center_X",
    "left_eye_center_Y",
    "right_eye_center_X",
    "right_eye_center_Y",
    "left_eye_inner_corner_X",
    "left_eye_inner_corner_Y",
    "left_eye_outer_corner_X",
    "left_eye_outer_corner_Y",
    "right_eye_inner_corner_X",
    "right_eye_inner_corner_Y",
    "right_eye_outer_corner_X",
    "right_eye_outer_corner_Y",
    "left_eyebrow_inner_end_X",
    "left_eyebrow_inner_end_Y",
    "left_eyebrow_outer_end_X",
    "left_eyebrow_outer_end_Y",
    "right_eyebrow_inner_end_X",
    "right_eyebrow_inner_end_Y",
    "right_eyebrow_outer_end_X",
    "right_eyebrow_outer_end_Y",
    "nose_tip_X",
    "nose_tip_Y",
    "mouth_left_corner_X",
    "mouth_left_corner_Y",
    "mouth_right_corner_X",
    "mouth_right_corner_Y",
    "mouth_center_top_lip_X",
    "mouth_center_top_lip_Y",
    "mouth_center_bottom_lip_X",
    "mouth_center_bottom_lip_Y",
]


def convert_y_to_dictonary(y_nd):
    """Converts the y array to a dictionary"""

    y_dict = {}
    for i, col in enumerate(Y_COLUMN_NAMES):
        y_dict[col] = y_nd[:, i]

    return y_dict


train_pd = pd.read_csv(TRAIN_DATA_PATH, encoding="utf8")
classes = train_pd.select_dtypes(include=[np.number]).columns
num_classes = len(classes)

## Data Preprocessing
train_only_all_points = train_pd.dropna()

imgs_all = []
for idx, r in train_only_all_points.iterrows():
    imgs_all.append(
        np.array(r["Image"].split())
        .astype(np.int64)
        .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    )
imgs_all = np.array(imgs_all)
y_all = np.array(train_only_all_points[classes])

tf.random.set_seed(1234)
np.random.seed(1234)
shuffle = np.random.permutation(np.arange(imgs_all.shape[0]))
imgs_all_sh, y_all_sh = imgs_all[shuffle], y_all[shuffle]

split = (0.8, 0.2)
splits = np.multiply(len(imgs_all_sh), split).astype(int)
y_train_nd, y_val_nd = np.split(y_all_sh, [splits[0]])
X_train, X_val = np.split(imgs_all_sh, [splits[0]])
# put y_train and y_val into a dictionary

y_train = convert_y_to_dictonary(y_train_nd)
y_val = convert_y_to_dictonary(y_val_nd)


def build_model(hp: keras_tuner.HyperParameters) -> keras.Model:

    tf.keras.backend.clear_session()
    input_layer = keras.layers.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
    )
    rescale = keras.layers.Rescaling(
        1.0 / 255, name="rescaling", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
    )(input_layer)

    ## Begin Convolutional Layers
    prev_layer = rescale
    for cur_con_layer in range(hp.Int("num_conv_layers", 1, 2)):
        conv_1 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            name="conv_1_" + str(cur_con_layer),
            padding="same",
            kernel_initializer="he_uniform",
            activation="relu",
        )(prev_layer)
        conv_2 = keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=(1, 1),
            name="conv_2_" + str(cur_con_layer),
            padding="same",
            kernel_initializer="he_uniform",
            activation="relu",
        )(conv_1)
        maxp = keras.layers.MaxPooling2D(
            pool_size=(2, 2), padding="same", name="pool_" + str(cur_con_layer)
        )(conv_2)
        # drop_1 = keras.layers.Dropout(0.25, name="Dropout_1")(maxp_1)
        norm = keras.layers.BatchNormalization(name="norm_" + str(cur_con_layer))(maxp)

        prev_layer = norm

    ##
    ## Begin Fully Connected layers
    ##
    flat_1 = keras.layers.Flatten()(prev_layer)
    dense_1 = keras.layers.Dense(
        1024, name="fc_1", kernel_initializer="he_uniform", activation="relu"
    )(flat_1)
    # drop_1 = keras.layers.Dropout(0.20, name="Dropout_1")(dense_1)
    norm_100 = keras.layers.BatchNormalization(name="norm_100")(dense_1)

    dense_2 = keras.layers.Dense(
        512, name="fc_2", kernel_initializer="he_uniform", activation="relu"
    )(norm_100)
    # drop_2 = keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
    norm_101 = keras.layers.BatchNormalization(name="norm_101")(dense_2)

    ##
    ## End Fully Connected Layers
    ##

    ## Construct Output Layers, loss and metrics
    output_layers = []
    loss_dict = {}
    metrics_dict = {}
    for i, col in enumerate(Y_COLUMN_NAMES):
        output_layers.append(
            keras.layers.Dense(units=1, activation=None, name=col)(norm_101)
        )
        loss_dict[col] = "mse"
        metrics_dict[col] = "mse"

    model = tf.keras.Model(
        inputs=[input_layer],
        outputs=output_layers,
        name="FacialKeypoints",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),
        loss=loss_dict,
        metrics=metrics_dict,
    )

    return model


## define the call back functions
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True,
)

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_weights_only=False,
    save_best_only=False,
)

##
tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=5,
    executions_per_trial=2,
    directory="tuning",
    project_name="facial_keypoints",
)
tuner.search_space_summary()
tuner.search(
    X_train,
    y_train,
    epochs=1,
    validation_data=(X_val, y_val),
    # callbacks=[early_stopping, model_checkpoint],
    callbacks=[early_stopping],
)

model: tf.keras.Model = tuner.get_best_models(num_models=1)[0]  # this is the best model
model.summary()
model.save(MODEL_PATH.joinpath(FINAL_MODEL_NAME), overwrite=True)
