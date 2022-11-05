import pandas as pd
import io
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image
from io import BytesIO

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96
BATCH_SIZE = 10
ROOT_DIR = Path(
    r"C:\Users\teddy\Documents\01-Berkeley\W207\facial-keypoints-detection"
).resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TRAIN_CSV = DATA_DIR.joinpath("training.csv")
TEST_CSV = DATA_DIR.joinpath("test.csv")

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

if not TEST_CSV.is_file():
    print(
        "Terminating, Test data csv file doe not exist or is not a file {0}".format(
            TEST_CSV
        )
    )
    exit()


def create_image_from_pixels(pixels) -> Image.Image:
    temp_image = Image.new("L", (IMAGE_WIDTH, IMAGE_HEIGHT))
    temp_image.putdata([int(x) for x in pixels.split()])

    return temp_image


def create_png(pixel_string):
    """
    Create Images from the integer text lists in the csv file
    """

    temp_image = create_image_from_pixels(pixel_string)

    buf = io.BytesIO()
    temp_image.save(buf, format="PNG")
    png_image = buf.getvalue()
    return png_image


train = pd.read_csv(TRAIN_CSV, encoding="utf8")
train["png"] = train["Image"].apply(create_png)
train.dropna(inplace=True)
classes = train.select_dtypes(include=[np.number]).columns
num_classes = len(classes)

# for col in feature_cols:
#    col_zscore = col + "_zscore"
#    train[col_zscore] = (train[col] - train[col].mean())/train[col].std(ddof=0)

###
imgs_all = []
np.random.seed(1234)
for idx, r in train.iterrows():
    img = tf.keras.preprocessing.image.load_img(
        BytesIO(r["png"]),
        color_mode="grayscale",
        target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
    )
    img = tf.keras.preprocessing.image.img_to_array(img)
    imgs_all.append(img)

imgs_all = np.stack(imgs_all)
y_all = np.array(train[classes])
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
tf.random.set_seed(1234)

input_layer = keras.layers.Input(
    shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1), name="InputLayer"
)
rescale = keras.layers.Rescaling(
    1.0 / 255, name="rescaling", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)
)(input_layer)

conv_1 = keras.layers.Conv2D(
    filters=32,
    kernel_size=(5, 5),
    strides=(1, 1),
    name="conv_1",
    padding="same",
    activation="relu",
)(rescale)
maxp_1 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv_1)

conv_2 = keras.layers.Conv2D(
    filters=64,
    kernel_size=(5, 5),
    strides=(1, 1),
    name="conv_2",
    padding="same",
    activation="relu",
)(maxp_1)
maxp_2 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv_2)

conv_3 = keras.layers.Conv2D(
    filters=128,
    kernel_size=(5, 5),
    strides=(1, 1),
    name="conv_3",
    padding="same",
    activation="relu",
)(maxp_2)
maxp_3 = keras.layers.MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv_3)

flat_1 = keras.layers.Flatten()(maxp_3)
dense_1 = keras.layers.Dense(1024, name="fc_1", activation="relu")(flat_1)

left_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_X"
)(dense_1)
left_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Center_Y"
)(dense_1)

right_eye_center_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_X"
)(dense_1)
right_eye_center_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Center_Y"
)(dense_1)

left_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_X"
)(dense_1)
left_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Inner_Corner_Y"
)(dense_1)

left_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_X"
)(dense_1)
left_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eye_Outer_Corner_Y"
)(dense_1)

right_eye_inner_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_X"
)(dense_1)
right_eye_inner_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Inner_Corner_Y"
)(dense_1)

right_eye_outer_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_X"
)(dense_1)
right_eye_outer_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eye_Outer_Corner_Y"
)(dense_1)

left_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_X"
)(dense_1)
left_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Inner_End_Y"
)(dense_1)

left_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_X"
)(dense_1)
left_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Left_Eyebrow_Outer_End_Y"
)(dense_1)

right_eyebrow_inner_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_X"
)(dense_1)
right_eyebrow_inner_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Inner_End_Y"
)(dense_1)

right_eyebrow_outer_end_x = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_X"
)(dense_1)
right_eyebrow_outer_end_y = keras.layers.Dense(
    units=1, activation=None, name="Right_Eyebrow_Outer_End_Y"
)(dense_1)

nose_tip_x = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_X")(dense_1)
nose_tip_y = keras.layers.Dense(units=1, activation=None, name="Nose_Tip_Y")(dense_1)

mouth_left_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_X"
)(dense_1)
mouth_left_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Left_Corner_Y"
)(dense_1)

mouth_right_corner_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_X"
)(dense_1)
mouth_right_corner_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Right_Corner_Y"
)(dense_1)

mouth_center_top_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_X"
)(dense_1)
mouth_center_top_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Top_Lip_Y"
)(dense_1)

mouth_center_bottom_lip_x = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_X"
)(dense_1)
mouth_center_bottom_lip_y = keras.layers.Dense(
    units=1, activation=None, name="Mouth_Center_Bottom_Lip_Y"
)(dense_1)

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
        "Left_Eye_Center_X": "mean_squared_error",
        "Left_Eye_Center_Y": "mean_squared_error",
        "Right_Eye_Center_X": "mean_squared_error",
        "Right_Eye_Center_Y": "mean_squared_error",
        "Left_Eye_Inner_Corner_X": "mean_squared_error",
        "Left_Eye_Inner_Corner_Y": "mean_squared_error",
        "Left_Eye_Outer_Corner_X": "mean_squared_error",
        "Left_Eye_Outer_Corner_Y": "mean_squared_error",
        "Right_Eye_Inner_Corner_X": "mean_squared_error",
        "Right_Eye_Inner_Corner_Y": "mean_squared_error",
        "Right_Eye_Outer_Corner_X": "mean_squared_error",
        "Right_Eye_Outer_Corner_Y": "mean_squared_error",
        "Left_Eyebrow_Inner_End_X": "mean_squared_error",
        "Left_Eyebrow_Inner_End_Y": "mean_squared_error",
        "Left_Eyebrow_Outer_End_X": "mean_squared_error",
        "Left_Eyebrow_Outer_End_Y": "mean_squared_error",
        "Right_Eyebrow_Inner_End_X": "mean_squared_error",
        "Right_Eyebrow_Inner_End_Y": "mean_squared_error",
        "Right_Eyebrow_Outer_End_X": "mean_squared_error",
        "Right_Eyebrow_Outer_End_Y": "mean_squared_error",
        "Nose_Tip_X": "mean_squared_error",
        "Nose_Tip_Y": "mean_squared_error",
        "Mouth_Left_Corner_X": "mean_squared_error",
        "Mouth_Left_Corner_Y": "mean_squared_error",
        "Mouth_Right_Corner_X": "mean_squared_error",
        "Mouth_Right_Corner_Y": "mean_squared_error",
        "Mouth_Center_Top_Lip_X": "mean_squared_error",
        "Mouth_Center_Top_Lip_Y": "mean_squared_error",
        "Mouth_Center_Bottom_Lip_X": "mean_squared_error",
        "Mouth_Center_Bottom_Lip_Y": "mean_squared_error",
    },
    metrics={
        "Left_Eye_Center_X": "mean_squared_error",
        "Left_Eye_Center_Y": "mean_squared_error",
        "Right_Eye_Center_X": "mean_squared_error",
        "Right_Eye_Center_Y": "mean_squared_error",
        "Left_Eye_Inner_Corner_X": "mean_squared_error",
        "Left_Eye_Inner_Corner_Y": "mean_squared_error",
        "Left_Eye_Outer_Corner_X": "mean_squared_error",
        "Left_Eye_Outer_Corner_Y": "mean_squared_error",
        "Right_Eye_Inner_Corner_X": "mean_squared_error",
        "Right_Eye_Inner_Corner_Y": "mean_squared_error",
        "Right_Eye_Outer_Corner_X": "mean_squared_error",
        "Right_Eye_Outer_Corner_Y": "mean_squared_error",
        "Left_Eyebrow_Inner_End_X": "mean_squared_error",
        "Left_Eyebrow_Inner_End_Y": "mean_squared_error",
        "Left_Eyebrow_Outer_End_X": "mean_squared_error",
        "Left_Eyebrow_Outer_End_Y": "mean_squared_error",
        "Right_Eyebrow_Inner_End_X": "mean_squared_error",
        "Right_Eyebrow_Inner_End_Y": "mean_squared_error",
        "Right_Eyebrow_Outer_End_X": "mean_squared_error",
        "Right_Eyebrow_Outer_End_Y": "mean_squared_error",
        "Nose_Tip_X": "mean_squared_error",
        "Nose_Tip_Y": "mean_squared_error",
        "Mouth_Left_Corner_X": "mean_squared_error",
        "Mouth_Left_Corner_Y": "mean_squared_error",
        "Mouth_Right_Corner_X": "mean_squared_error",
        "Mouth_Right_Corner_Y": "mean_squared_error",
        "Mouth_Center_Top_Lip_X": "mean_squared_error",
        "Mouth_Center_Top_Lip_Y": "mean_squared_error",
        "Mouth_Center_Bottom_Lip_X": "mean_squared_error",
        "Mouth_Center_Bottom_Lip_Y": "mean_squared_error",
    },
)
model.summary()

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
    batch_size=100,
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
    verbose=True,
)
model.save("good_project_model", overwrite=True)
