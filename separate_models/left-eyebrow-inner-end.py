import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = 32
ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("processed_training.csv")

MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "facial-keypoints-left-eyebrow-inner-end"

Y_COLUMN_NAMES = [
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

if not TRAIN_CSV.is_file():
    print(
        "Terminating, Training data csv file doe not exist or is not a file {0}".format(
            TRAIN_CSV
        )
    )
    exit()


if not MODEL_DIR.is_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


train = pd.read_csv(TRAIN_CSV, encoding="utf8")
classes = train.select_dtypes(include=[np.number]).columns

num_classes = len(classes)

# for col in feature_cols:
#    col_zscore = col + "_zscore"
#    train[col_zscore] = (train[col] - train[col].mean())/train[col].std(ddof=0)

###

train_only_all_points = train[train["left_eyebrow_inner_end_x"].notna()]

imgs_all = []
for idx, r in train_only_all_points.iterrows():
    imgs_all.append(
        np.array(r["Image"].split())
        .astype(np.int64)
        .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    )
imgs_all = np.array(imgs_all)
y_all = np.array(train_only_all_points[classes])


###
tf.random.set_seed(1234)
np.random.seed(1234)
shuffle = np.random.permutation(np.arange(imgs_all.shape[0]))
imgs_all_sh, y_all_sh = imgs_all[shuffle], y_all[shuffle]

split = (0.8, 0.2)
splits = np.multiply(len(imgs_all_sh), split).astype(int)
y_train, y_val = np.split(y_all_sh, [splits[0]])
X_train, X_val = np.split(imgs_all_sh, [splits[0]])


tf.keras.backend.clear_session()

input_layer = keras.layers.Input(
    shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
)
rescale = keras.layers.Rescaling(
    1.0 / 255, name="rescaling", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
)(input_layer)

##
## Begin Convolutional Layers
##

conv_1 = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_1",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(rescale)
conv_2 = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_2",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_1)
maxp_1 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_1")(
    conv_2
)
# drop_1 = keras.layers.Dropout(0.25, name="Dropout_1")(maxp_1)
norm_1 = keras.layers.BatchNormalization(name="norm_1")(maxp_1)

conv_3 = keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_3",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_1)
conv_4 = keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_4",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_3)
maxp_2 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_2")(
    conv_4
)
# drop_2 = keras.layers.Dropout(0.25, name="Dropout_2")(maxp_2)
norm_2 = keras.layers.BatchNormalization(name="norm_2")(maxp_2)

conv_5 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_5",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_2)
conv_6 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_6",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_5)
maxp_3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_3")(
    conv_6
)
# drop_3 = keras.layers.Dropout(0.25, name="Dropout_3")(maxp_2)
norm_3 = keras.layers.BatchNormalization(name="norm_3")(maxp_3)


##
## Begin Fully Connected layers
##

flat_1 = keras.layers.Flatten()(norm_3)
dense_1 = keras.layers.Dense(
    1024, name="fc_1", kernel_initializer="he_uniform", activation="relu"
)(flat_1)
# drop_1 = keras.layers.Dropout(0.20, name="Dropout_1")(dense_1)
norm_4 = keras.layers.BatchNormalization(name="norm_4")(dense_1)

dense_2 = keras.layers.Dense(
    1024, name="fc_2", kernel_initializer="he_uniform", activation="relu"
)(norm_4)
# drop_2 = keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
norm_5 = keras.layers.BatchNormalization(name="norm_5")(dense_2)

##
## End Fully Connected Layers
##

##
## Begin Output Layers
##

left_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_X"
)(norm_5)
left_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_Y"
)(norm_5)


model = tf.keras.Model(
    inputs=[input_layer],
    outputs=[
        left_eyebrow_inner_end_x,
        left_eyebrow_inner_end_y,
    ],
    name=FINAL_MODEL_NAME,
)

model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),
    loss={
        "Left_Eyebrow_Inner_End_X": "mse",
        "Left_Eyebrow_Inner_End_Y": "mse",
    },
    metrics={
        "Left_Eyebrow_Inner_End_X": "mse",
        "Left_Eyebrow_Inner_End_Y": "mse",
    },
)
model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True,
)

model_checkpoint = ModelCheckpoint(
    "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
    monitor="val_loss",
    mode="min",
    verbose=1,
    save_weights_only=False,
    save_best_only=False,
)

history = model.fit(
    x=X_train,
    y={
        "Left_Eyebrow_Inner_End_X": y_train[:, 12],
        "Left_Eyebrow_Inner_End_Y": y_train[:, 13],
    },
    epochs=500,
    batch_size=BATCH_SIZE,
    validation_data=(
        X_val,
        {
            "Left_Eyebrow_Inner_End_X": y_val[:, 12],
            "Left_Eyebrow_Inner_End_Y": y_val[:, 13],
        },
    ),
    verbose=2,
    # callbacks=[early_stopping, model_checkpoint],
    callbacks=[early_stopping],
)


with open(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"), "wb") as history_file:
    pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

model.save(MODEL_DIR.joinpath(FINAL_MODEL_NAME), overwrite=True)
