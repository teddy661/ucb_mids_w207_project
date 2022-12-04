import model_trainer.data_loader as data_loader
import model_trainer.model_builder as model_builder

MODEL_NAME = "model_joint"
"""
This is a model that predicts all 15 keypoints using one model. 
"""
LABELS_TO_INCLUDE = [
    "left_eye_center",
    "right_eye_center",
    "left_eye_inner_corner",
    "left_eye_outer_corner",
    "right_eye_inner_corner",
    "right_eye_outer_corner",
    "left_eyebrow_inner_end",
    "left_eyebrow_outer_end",
    "right_eyebrow_inner_end",
    "right_eyebrow_outer_end",
    "nose_tip",
    "mouth_left_corner",
    "mouth_right_corner",
    "mouth_center_top_lip",
    "mouth_center_bottom_lip",
]

model = model_builder.tune_model(
    labels_to_include=LABELS_TO_INCLUDE, model_name=MODEL_NAME
)

X_test = data_loader.load_test_data_from_file()
results = model.predict(X_test, batch_size=32, verbose=1)

model_builder.save_results_and_submit(results, model_name=MODEL_NAME)
