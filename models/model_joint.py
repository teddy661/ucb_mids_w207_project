import model_trainer.model_builder as model_builder

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

model = model_builder.tune_model(
    labels_to_include=LABELS_TO_INCLUDE, model_name=MODEL_NAME
)
model_builder.test_and_submit(model, model_name=MODEL_NAME)
