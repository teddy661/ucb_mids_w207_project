from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

IMAGE_HEIGHT = 96
IMAGE_WIDTH = 96

OUTPUTS = [
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

ROOT_DIR = Path(r"../facial-keypoints-detection").resolve()
DATA_DIR = ROOT_DIR.joinpath("data")
TEST_CSV = DATA_DIR.joinpath("test.csv")
ID_LOOKUP_TABLE = DATA_DIR.joinpath("IdLookupTable.csv")
MODEL_DIR = Path("model_saves").resolve()

LEFT_EYE_CENTER = "facial-keypoints-left-eye-center"
RIGHT_EYE_CENTER = "facial-keypoints-right-eye-center"
LEFT_EYE_INNER_CORNER = "facial-keypoints-left-eye-inner-corner"
LEFT_EYE_OUTER_CORNER = "facial-keypoints-left-eye-outer-corner"
RIGHT_EYE_INNER_CORNER = "facial-keypoints-right-eye-inner-corner"
RIGHT_EYE_OUTER_CORNER = "facial-keypoints-right-eye-outer-corner"
LEFT_EYEBROW_INNER_END = "facial-keypoints-left-eyebrow-inner-end"
LEFT_EYEBROW_OUTER_END = "facial-keypoints-left-eyebrow-outer-end"
RIGHT_EYEBROW_INNER_END = "facial-keypoints-right-eyebrow-inner-end"
RIGHT_EYEBROW_OUTER_END = "facial-keypoints-right-eyebrow-outer-end"
NOSE_TIP = "facial-keypoints-nose-tip"
MOUTH_LEFT_CORNER = "facial-keypoints-mouth-left-corner"
MOUTH_RIGHT_CORNER = "facial-keypoints-mouth-right-corner"
MOUTH_CENTER_TOP_LIP = "facial-keypoints-mouth-center-top-lip"
MOUTH_CENTER_BOTTOM_LIP = "facial-keypoints-mouth-center-bottom-lip"

left_eye_center_model = tf.keras.models.load_model(MODEL_DIR.joinpath(LEFT_EYE_CENTER))
left_eye_center_model.trainable = False

right_eye_center_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(RIGHT_EYE_CENTER)
)
right_eye_center_model.trainable = False

left_eye_inner_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(LEFT_EYE_INNER_CORNER)
)
left_eye_inner_corner_model.trainable = False

left_eye_outer_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(LEFT_EYE_OUTER_CORNER)
)
left_eye_outer_corner_model.trainable = False

right_eye_inner_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(RIGHT_EYE_INNER_CORNER)
)
right_eye_inner_corner_model.trainable = False

right_eye_outer_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(RIGHT_EYE_OUTER_CORNER)
)
right_eye_outer_corner_model.trainable = False

left_eyebrow_inner_end_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(LEFT_EYEBROW_INNER_END)
)
left_eyebrow_inner_end_model.trainable = False

left_eyebrow_outer_end_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(LEFT_EYEBROW_OUTER_END)
)
left_eyebrow_outer_end_model.trainable = False

right_eyebrow_inner_end_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(RIGHT_EYEBROW_INNER_END)
)
right_eyebrow_inner_end_model.trainable = False

right_eyebrow_outer_end_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(RIGHT_EYEBROW_OUTER_END)
)
right_eyebrow_outer_end_model.trainable = False

nose_tip_model = tf.keras.models.load_model(MODEL_DIR.joinpath(NOSE_TIP))
nose_tip_model.trainable = False

mouth_left_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(MOUTH_LEFT_CORNER)
)
mouth_left_corner_model.trainable = False

mouth_right_corner_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(MOUTH_RIGHT_CORNER)
)
mouth_right_corner_model.trainable = False

mouth_center_top_lip_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(MOUTH_CENTER_TOP_LIP)
)
mouth_center_top_lip_model.trainable = False

mouth_center_bottom_lip_model = tf.keras.models.load_model(
    MODEL_DIR.joinpath(MOUTH_CENTER_BOTTOM_LIP)
)
mouth_center_bottom_lip_model.trainable = False

# merged = tf.keras.Model(
#    inputs=[left_eye_center_model.get_layer("InputLayer")],
#    outputs=[left_eye_center_model.get_layer("Left_Eye_Center_X")],
# )

# inputlayer=right_eye_center_model.get_layer("InputLayer")
# inputlayer._name="Left_Eye_Center_Input"
for layer in right_eye_center_model.layers:
    layer._name = layer.name + str("_2")
for layer in left_eye_inner_corner_model.layers:
    layer._name = layer.name + str("_3")
for layer in left_eye_outer_corner_model.layers:
    layer._name = layer.name + str("_4")
for layer in right_eye_inner_corner_model.layers:
    layer._name = layer.name + str("_5")
for layer in right_eye_outer_corner_model.layers:
    layer._name = layer.name + str("_6")
for layer in left_eyebrow_inner_end_model.layers:
    layer._name = layer.name + str("_7")
for layer in left_eyebrow_outer_end_model.layers:
    layer._name = layer.name + str("_8")
for layer in right_eyebrow_inner_end_model.layers:
    layer._name = layer.name + str("_9")
for layer in right_eyebrow_outer_end_model.layers:
    layer._name = layer.name + str("_10")
for layer in nose_tip_model.layers:
    layer._name = layer.name + str("_11")
for layer in mouth_left_corner_model.layers:
    layer._name = layer.name + str("_12")
for layer in mouth_right_corner_model.layers:
    layer._name = layer.name + str("_13")
for layer in mouth_center_top_lip_model.layers:
    layer._name = layer.name + str("_14")
for layer in mouth_center_bottom_lip_model.layers:
    layer._name = layer.name + str("_15")

complete_model = tf.keras.Model(
    inputs=[
        left_eye_center_model.input,
        right_eye_center_model.input,
        left_eye_inner_corner_model.input,
        left_eye_outer_corner_model.input,
        right_eye_inner_corner_model.input,
        right_eye_outer_corner_model.input,
        left_eyebrow_inner_end_model.input,
        left_eyebrow_outer_end_model.input,
        right_eyebrow_inner_end_model.input,
        right_eyebrow_outer_end_model.input,
        nose_tip_model.input,
        mouth_left_corner_model.input,
        mouth_right_corner_model.input,
        mouth_center_top_lip_model.input,
        mouth_center_bottom_lip_model.input,
    ],
    outputs=[
        left_eye_center_model.output,
        right_eye_center_model.output,
        left_eye_inner_corner_model.output,
        left_eye_outer_corner_model.output,
        right_eye_inner_corner_model.output,
        right_eye_outer_corner_model.output,
        left_eyebrow_inner_end_model.output,
        left_eyebrow_outer_end_model.output,
        right_eyebrow_inner_end_model.output,
        right_eyebrow_outer_end_model.output,
        nose_tip_model.output,
        mouth_left_corner_model.output,
        mouth_right_corner_model.output,
        mouth_center_top_lip_model.output,
        mouth_center_bottom_lip_model.output,
    ],
)
complete_model.summary()

test_df = pd.read_csv(TEST_CSV, encoding="utf8")

imgs_all = []
for idx, r in test_df.iterrows():
    imgs_all.append(
        np.array(r["Image"].split())
        .astype(np.uint8)
        .reshape(IMAGE_WIDTH, IMAGE_HEIGHT, 1)
    )

processed_images = []
clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
for cimage in imgs_all:
    step_1 = cv2.fastNlMeansDenoising(cimage)
    step_2 = clahe.apply(step_1)
    step_3 = step_2.reshape(96,96,1)
    processed_images.append(step_3)

# This is used below choose either processed_images or imgs_all for original
imgs_all = np.array(processed_images)

test_np_data = np.array(imgs_all)

results = complete_model.predict(
    [
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
        test_np_data,
    ],
    batch_size=200,
    verbose=2,
)

np_results = np.transpose(np.array(results))
np_results = np.squeeze(np_results)
np_results = np_results.reshape((-1, 30), order="F")
# np_results = np_results.reshape(np_results.shape[0],np_results.shape[1]*np_results.shape[2])
results_df = pd.DataFrame(np_results, columns=OUTPUTS)
results_df.index += 1
results_df.to_csv("test_assembled_results.csv", index=False, encoding="utf-8")

reformatted_results = []
for index, row in results_df.iterrows():
    row_df = row.to_frame()
    row_df.rename(columns={index: "Location"}, inplace=True)
    row_df.reset_index(inplace=True)
    row_df.rename(columns={"index": "FeatureName"}, inplace=True)
    row_df.insert(0, "ImageId", index)
    reformatted_results.append(row_df)
reformatted_results_df = pd.concat(reformatted_results, ignore_index=True)

id_lookup_df = pd.read_csv(ID_LOOKUP_TABLE, encoding="utf8")

submission_df = pd.merge(
    id_lookup_df, reformatted_results_df, how="left", on=["ImageId", "FeatureName"]
)
submission_df.drop(columns=["ImageId", "FeatureName", "Location_x"], inplace=True)
submission_df.rename(columns={"Location_y": "Location"}, inplace=True)
submission_df.to_csv("submission-assembled.csv", index=False, encoding="utf-8")
