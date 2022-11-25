import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = 32
ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("processed_training.csv")
MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "final-model"

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

train_only_all_points = train.dropna()
mouth_left_corner_points = train[train["mouth_left_corner_x"].notna()]

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
conv_100 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_100",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_6)
maxp_3 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_3")(
    conv_100
)
# drop_3 = keras.layers.Dropout(0.25, name="Dropout_3")(maxp_2)
norm_3 = keras.layers.BatchNormalization(name="norm_3")(maxp_3)

conv_301 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_301",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_3)
conv_302 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_302",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_301)
conv_303 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_303",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_302)
maxp_301 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_301")(
    conv_303
)
# drop_3 = keras.layers.Dropout(0.25, name="Dropout_3")(maxp_2)
norm_301 = keras.layers.BatchNormalization(name="norm_301")(maxp_301)

conv_401 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_401",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_301)
conv_402 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_402",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_401)
conv_403 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_403",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(conv_402)
maxp_401 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="pool_401")(
    conv_403
)
# drop_3 = keras.layers.Dropout(0.25, name="Dropout_3")(maxp_2)
norm_401 = keras.layers.BatchNormalization(name="norm_401")(maxp_401)


##
## Begin Fully Connected layers
##

flat_1 = keras.layers.Flatten()(norm_401)
dense_1 = keras.layers.Dense(
    2048, name="fc_1", kernel_initializer="he_uniform", activation="relu"
)(flat_1)
# drop_1 = keras.layers.Dropout(0.20, name="Dropout_1")(dense_1)
norm_4 = keras.layers.BatchNormalization(name="norm_4")(dense_1)

dense_15 = keras.layers.Dense(
    2048, name="fc_15", kernel_initializer="he_uniform", activation="relu"
)(norm_4)
# drop_2 = keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
norm_15 = keras.layers.BatchNormalization(name="norm_15")(dense_15)

dense_2 = keras.layers.Dense(
    2048, name="fc_2", kernel_initializer="he_uniform", activation="relu"
)(norm_15)
# drop_2 = keras.layers.Dropout(0.20, name="Dropout_2")(dense_2)
norm_5 = keras.layers.BatchNormalization(name="norm_5")(dense_2)

##
## End Fully Connected Layers
##

##
## Begin Output Layers
##

left_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_X"
)(norm_5)
left_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_Y"
)(norm_5)

right_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_X"
)(norm_5)
right_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_Y"
)(norm_5)

left_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_X"
)(norm_5)
left_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_Y"
)(norm_5)

left_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_X"
)(norm_5)
left_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_Y"
)(norm_5)

right_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_X"
)(norm_5)
right_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_Y"
)(norm_5)

right_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_X"
)(norm_5)
right_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_Y"
)(norm_5)

left_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_X"
)(norm_5)
left_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_Y"
)(norm_5)

left_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_X"
)(norm_5)
left_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_Y"
)(norm_5)

right_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_X"
)(norm_5)
right_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_Y"
)(norm_5)

right_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_X"
)(norm_5)
right_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_Y"
)(norm_5)

nose_tip_x = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_X")(norm_5)
nose_tip_y = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_Y")(norm_5)

mouth_left_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_X"
)(norm_5)
mouth_left_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_Y"
)(norm_5)

mouth_right_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_X"
)(norm_5)
mouth_right_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_Y"
)(norm_5)

mouth_center_top_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_X"
)(norm_5)
mouth_center_top_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_Y"
)(norm_5)

mouth_center_bottom_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_X"
)(norm_5)
mouth_center_bottom_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_Y"
)(norm_5)

model = tf.keras.Model(
    inputs=[input_layer],
    outputs=[
        left_eye_center_x,
        left_eye_center_y,
        right_eye_center_x,
        right_eye_center_y,
        left_eye_inner_corner_x,
        left_eye_inner_corner_y,
        left_eye_outer_corner_x,
        left_eye_outer_corner_y,
        right_eye_inner_corner_x,
        right_eye_inner_corner_y,
        right_eye_outer_corner_x,
        right_eye_outer_corner_y,
        left_eyebrow_inner_end_x,
        left_eyebrow_inner_end_y,
        left_eyebrow_outer_end_x,
        left_eyebrow_outer_end_y,
        right_eyebrow_inner_end_x,
        right_eyebrow_inner_end_y,
        right_eyebrow_outer_end_x,
        right_eyebrow_outer_end_y,
        nose_tip_x,
        nose_tip_y,
        mouth_left_corner_x,
        mouth_left_corner_y,
        mouth_right_corner_x,
        mouth_right_corner_y,
        mouth_center_top_lip_x,
        mouth_center_top_lip_y,
        mouth_center_bottom_lip_x,
        mouth_center_bottom_lip_y,
    ],
    name="FacialKeypoints",
)

model.compile(
    optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0001),
    loss={
        "Left_Eye_Center_X": "mse",
        "Left_Eye_Center_Y": "mse",
        "Right_Eye_Center_X": "mse",
        "Right_Eye_Center_Y": "mse",
        "Left_Eye_Inner_Corner_X": "mse",
        "Left_Eye_Inner_Corner_Y": "mse",
        "Left_Eye_Outer_Corner_X": "mse",
        "Left_Eye_Outer_Corner_Y": "mse",
        "Right_Eye_Inner_Corner_X": "mse",
        "Right_Eye_Inner_Corner_Y": "mse",
        "Right_Eye_Outer_Corner_X": "mse",
        "Right_Eye_Outer_Corner_Y": "mse",
        "Left_Eyebrow_Inner_End_X": "mse",
        "Left_Eyebrow_Inner_End_Y": "mse",
        "Left_Eyebrow_Outer_End_X": "mse",
        "Left_Eyebrow_Outer_End_Y": "mse",
        "Right_Eyebrow_Inner_End_X": "mse",
        "Right_Eyebrow_Inner_End_Y": "mse",
        "Right_Eyebrow_Outer_End_X": "mse",
        "Right_Eyebrow_Outer_End_Y": "mse",
        "Nose_Tip_X": "mse",
        "Nose_Tip_Y": "mse",
        "Mouth_Left_Corner_X": "mse",
        "Mouth_Left_Corner_Y": "mse",
        "Mouth_Right_Corner_X": "mse",
        "Mouth_Right_Corner_Y": "mse",
        "Mouth_Center_Top_Lip_X": "mse",
        "Mouth_Center_Top_Lip_Y": "mse",
        "Mouth_Center_Bottom_Lip_X": "mse",
        "Mouth_Center_Bottom_Lip_Y": "mse",
    },
    metrics={
        "Left_Eye_Center_X": "mse",
        "Left_Eye_Center_Y": "mse",
        "Right_Eye_Center_X": "mse",
        "Right_Eye_Center_Y": "mse",
        "Left_Eye_Inner_Corner_X": "mse",
        "Left_Eye_Inner_Corner_Y": "mse",
        "Left_Eye_Outer_Corner_X": "mse",
        "Left_Eye_Outer_Corner_Y": "mse",
        "Right_Eye_Inner_Corner_X": "mse",
        "Right_Eye_Inner_Corner_Y": "mse",
        "Right_Eye_Outer_Corner_X": "mse",
        "Right_Eye_Outer_Corner_Y": "mse",
        "Left_Eyebrow_Inner_End_X": "mse",
        "Left_Eyebrow_Inner_End_Y": "mse",
        "Left_Eyebrow_Outer_End_X": "mse",
        "Left_Eyebrow_Outer_End_Y": "mse",
        "Right_Eyebrow_Inner_End_X": "mse",
        "Right_Eyebrow_Inner_End_Y": "mse",
        "Right_Eyebrow_Outer_End_X": "mse",
        "Right_Eyebrow_Outer_End_Y": "mse",
        "Nose_Tip_X": "mse",
        "Nose_Tip_Y": "mse",
        "Mouth_Left_Corner_X": "mse",
        "Mouth_Left_Corner_Y": "mse",
        "Mouth_Right_Corner_X": "mse",
        "Mouth_Right_Corner_Y": "mse",
        "Mouth_Center_Top_Lip_X": "mse",
        "Mouth_Center_Top_Lip_Y": "mse",
        "Mouth_Center_Bottom_Lip_X": "mse",
        "Mouth_Center_Bottom_Lip_Y": "mse",
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
        "Left_Eye_Center_X": y_train[:, 0],
        "Left_Eye_Center_Y": y_train[:, 1],
        "Right_Eye_Center_X": y_train[:, 2],
        "Right_Eye_Center_Y": y_train[:, 3],
        "Left_Eye_Inner_Corner_X": y_train[:, 4],
        "Left_Eye_Inner_Corner_Y": y_train[:, 5],
        "Left_Eye_Outer_Corner_X": y_train[:, 6],
        "Left_Eye_Outer_Corner_Y": y_train[:, 7],
        "Right_Eye_Inner_Corner_X": y_train[:, 8],
        "Right_Eye_Inner_Corner_Y": y_train[:, 9],
        "Right_Eye_Outer_Corner_X": y_train[:, 10],
        "Right_Eye_Outer_Corner_Y": y_train[:, 11],
        "Left_Eyebrow_Inner_End_X": y_train[:, 12],
        "Left_Eyebrow_Inner_End_Y": y_train[:, 13],
        "Left_Eyebrow_Outer_End_X": y_train[:, 14],
        "Left_Eyebrow_Outer_End_Y": y_train[:, 15],
        "Right_Eyebrow_Inner_End_X": y_train[:, 16],
        "Right_Eyebrow_Inner_End_Y": y_train[:, 17],
        "Right_Eyebrow_Outer_End_X": y_train[:, 18],
        "Right_Eyebrow_Outer_End_Y": y_train[:, 19],
        "Nose_Tip_X": y_train[:, 20],
        "Nose_Tip_Y": y_train[:, 21],
        "Mouth_Left_Corner_X": y_train[:, 22],
        "Mouth_Left_Corner_Y": y_train[:, 23],
        "Mouth_Right_Corner_X": y_train[:, 24],
        "Mouth_Right_Corner_Y": y_train[:, 25],
        "Mouth_Center_Top_Lip_X": y_train[:, 26],
        "Mouth_Center_Top_Lip_Y": y_train[:, 27],
        "Mouth_Center_Bottom_Lip_X": y_train[:, 28],
        "Mouth_Center_Bottom_Lip_Y": y_train[:, 29],
    },
    epochs=1000,
    batch_size=BATCH_SIZE,
    validation_data=(
        X_val,
        {
            "Left_Eye_Center_X": y_val[:, 0],
            "Left_Eye_Center_Y": y_val[:, 1],
            "Right_Eye_Center_X": y_val[:, 2],
            "Right_Eye_Center_Y": y_val[:, 3],
            "Left_Eye_Inner_Corner_X": y_val[:, 4],
            "Left_Eye_Inner_Corner_Y": y_val[:, 5],
            "Left_Eye_Outer_Corner_X": y_val[:, 6],
            "Left_Eye_Outer_Corner_Y": y_val[:, 7],
            "Right_Eye_Inner_Corner_X": y_val[:, 8],
            "Right_Eye_Inner_Corner_Y": y_val[:, 9],
            "Right_Eye_Outer_Corner_X": y_val[:, 10],
            "Right_Eye_Outer_Corner_Y": y_val[:, 11],
            "Left_Eyebrow_Inner_End_X": y_val[:, 12],
            "Left_Eyebrow_Inner_End_Y": y_val[:, 13],
            "Left_Eyebrow_Outer_End_X": y_val[:, 14],
            "Left_Eyebrow_Outer_End_Y": y_val[:, 15],
            "Right_Eyebrow_Inner_End_X": y_val[:, 16],
            "Right_Eyebrow_Inner_End_Y": y_val[:, 17],
            "Right_Eyebrow_Outer_End_X": y_val[:, 18],
            "Right_Eyebrow_Outer_End_Y": y_val[:, 19],
            "Nose_Tip_X": y_val[:, 20],
            "Nose_Tip_Y": y_val[:, 21],
            "Mouth_Left_Corner_X": y_val[:, 22],
            "Mouth_Left_Corner_Y": y_val[:, 23],
            "Mouth_Right_Corner_X": y_val[:, 24],
            "Mouth_Right_Corner_Y": y_val[:, 25],
            "Mouth_Center_Top_Lip_X": y_val[:, 26],
            "Mouth_Center_Top_Lip_Y": y_val[:, 27],
            "Mouth_Center_Bottom_Lip_X": y_val[:, 28],
            "Mouth_Center_Bottom_Lip_Y": y_val[:, 29],
        },
    ),
    verbose=2,
    # callbacks=[early_stopping, model_checkpoint],
    callbacks=[early_stopping],
)


with open(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"), "wb") as history_file:
    pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

model.save(MODEL_DIR.joinpath(FINAL_MODEL_NAME), overwrite=True)
