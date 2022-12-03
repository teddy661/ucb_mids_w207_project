import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = 32
ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("processed_training.csv")
MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME = "vgg19-reduced"

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

train_only_all_points = train.dropna()
y_all = np.array(train_only_all_points[classes])

imgs_all = []
for idx, r in train_only_all_points.iterrows():
    imgs_all.append(
        np.fromstring(r["Image"], dtype=np.uint8, sep=" ").reshape(
            IMAGE_WIDTH, IMAGE_HEIGHT, 1
        )
    )
imgs_all = np.array(imgs_all)


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

conv_101 = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_101",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(rescale)
norm_101 = keras.layers.BatchNormalization(name="norm_101")(conv_101)
conv_102 = keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_102",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_101)
norm_102 = keras.layers.BatchNormalization(name="norm_102")(conv_102)
maxp_101 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="same", name="maxp_101")(
    norm_102
)

conv_201 = keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_201",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(maxp_101)
norm_201 = keras.layers.BatchNormalization(name="norm_201")(conv_201)
conv_202 = keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_202",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_201)
norm_202 = keras.layers.BatchNormalization(name="norm_202")(conv_202)
maxp_201 = keras.layers.MaxPooling2D(pool_size=(2, 2),  padding="same", name="maxp_201")(
    norm_202
)

conv_301 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_301",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(maxp_201)
norm_301 = keras.layers.BatchNormalization(name="norm_301")(conv_301)
conv_302 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_302",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_301)
norm_302 = keras.layers.BatchNormalization(name="norm_302")(conv_302)
conv_303 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_303",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_302)
norm_303 = keras.layers.BatchNormalization(name="norm_303")(conv_303)
conv_304 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_304",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_303)
norm_304 = keras.layers.BatchNormalization(name="norm_304")(conv_304)
maxp_301 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxp_301")(
    norm_304
)

conv_401 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_401",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(maxp_301)
norm_401 = keras.layers.BatchNormalization(name="norm_401")(conv_401)
conv_402 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_402",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_401)
norm_402 = keras.layers.BatchNormalization(name="norm_402")(conv_402)
conv_403 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_403",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_402)
norm_403 = keras.layers.BatchNormalization(name="norm_403")(conv_403)
conv_404 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_404",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_403)
norm_404 = keras.layers.BatchNormalization(name="norm_404")(conv_404)
maxp_401 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxp_401")(
    norm_404
)

conv_501 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_501",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(maxp_401)
norm_501 = keras.layers.BatchNormalization(name="norm_501")(conv_501)
conv_502 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_502",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_501)
norm_502 = keras.layers.BatchNormalization(name="norm_502")(conv_502)
conv_503 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_503",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_502)
norm_503 = keras.layers.BatchNormalization(name="norm_503")(conv_503)
conv_504 = keras.layers.Conv2D(
    filters=256,
    kernel_size=(3, 3),
    strides=(1, 1),
    name="conv_504",
    padding="same",
    kernel_initializer="he_uniform",
    activation="relu",
)(norm_503)
norm_504 = keras.layers.BatchNormalization(name="norm_504")(conv_504)
maxp_501 = keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same", name="maxp_501")(
    norm_504
)



##
## Begin Fully Connected layers
##

flat_1 = keras.layers.Flatten()(maxp_501)
dense_1 = keras.layers.Dense(
    2048, name="fc_1", kernel_initializer="he_uniform", activation="relu"
)(flat_1)
norm_fc1 = keras.layers.BatchNormalization(name="norm_fc1")(dense_1)

dense_2 = keras.layers.Dense(
    2048, name="fc_2", kernel_initializer="he_uniform", activation="relu"
)(norm_fc1)
norm_fc2 = keras.layers.BatchNormalization(name="norm_fc2")(dense_2)


##
## Begin Output Layers
##

left_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_X"
)(norm_fc2)
left_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_Y"
)(norm_fc2)

right_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_X"
)(norm_fc2)
right_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_Y"
)(norm_fc2)

left_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_X"
)(norm_fc2)
left_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_Y"
)(norm_fc2)

left_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_X"
)(norm_fc2)
left_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_Y"
)(norm_fc2)

right_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_X"
)(norm_fc2)
right_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_Y"
)(norm_fc2)

right_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_X"
)(norm_fc2)
right_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_Y"
)(norm_fc2)

left_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_X"
)(norm_fc2)
left_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_Y"
)(norm_fc2)

left_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_X"
)(norm_fc2)
left_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_Y"
)(norm_fc2)

right_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_X"
)(norm_fc2)
right_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_Y"
)(norm_fc2)

right_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_X"
)(norm_fc2)
right_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_Y"
)(norm_fc2)

nose_tip_x = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_X")(norm_fc2)
nose_tip_y = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_Y")(norm_fc2)

mouth_left_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_X"
)(norm_fc2)
mouth_left_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_Y"
)(norm_fc2)

mouth_right_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_X"
)(norm_fc2)
mouth_right_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_Y"
)(norm_fc2)

mouth_center_top_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_X"
)(norm_fc2)
mouth_center_top_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_Y"
)(norm_fc2)

mouth_center_bottom_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_X"
)(norm_fc2)
mouth_center_bottom_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_Y"
)(norm_fc2)

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
    optimizer=tf.keras.optimizers.Adam(),
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

tf.keras.utils.plot_model(model, to_file="model_ed7.png", show_shapes=True)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=40,
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

reduce_lr_on_plateau = ReduceLROnPlateau(
    monitor='val_loss',
    patience=4,
    verbose=1,
    factor=0.3,
    min_lr=0.00000001)

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
    epochs=400,
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
	# callbacks=[early_stopping],
    # callbacks=[early_stopping, model_checkpoint],
    callbacks=[early_stopping,reduce_lr_on_plateau],
)


with open(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"), "wb") as history_file:
    pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

model.save(MODEL_DIR.joinpath(FINAL_MODEL_NAME), overwrite=True)
