import data.data_loader as data_loader
import model_trainer.model_builder as model_builder

MODEL_NAME = "model_two_stage"
"""
This is a model that predicts 4 keypoints using one model, and then use the outputs to predict the other 11. 
"""

model_name_stage_one = "model_stage_one"
labels_first_stage = [
    "left_eye_center",
    "right_eye_center",
    "nose_tip",
    "mouth_center_bottom_lip",
]
model_one = model_builder.tune_model(
    labels_to_include=labels_first_stage, model_name=model_name_stage_one
)

X_test = data_loader.load_test_data_from_file()
results = model_one.predict(X_test, batch_size=32, verbose=1)

model_name_stage_two = "model_stage_two"
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

model_two = model_builder.tune_model(
    labels_to_include=labels_second_stage, model_name=model_name_stage_two
)

results = model_two.predict(X_test, batch_size=32, verbose=1)


model_builder.save_results_and_submit(results, model_name=MODEL_NAME)
