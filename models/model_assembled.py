import tensorflow as tf

import data.path_utils as path_utils
import model_trainer.data_loader as data_loader
import model_trainer.model_builder as model_builder

"""
This is a model that predicts all 15 keypoints using one model. 
"""
INDIVIDUAL_LABLES = [
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

models = {}

for label in INDIVIDUAL_LABLES:
    model_name = f"model_{label}"
    model: tf.keras.Model = model_builder.tune_model(
        labels_to_include=[label], model_name=model_name
    )
    models[label] = model

all_inputs = []
all_outputs = []
for label, model in models.items():
    for layer in model.layers:  # rename all layers
        layer._name = model.name + ": " + layer.name

    all_inputs.append(model.input)
    all_outputs.append(model.output)


complete_model = tf.keras.Model(
    inputs=all_inputs, outputs=all_outputs, name="model_complete"
)

complete_model.summary()


# test the model with test data
_, TEST_DATA_PATH, MODEL_PATH = path_utils.get_data_paths()
X_test = data_loader.load_test_data_from_file(TEST_DATA_PATH)

results = complete_model.predict(
    [X_test * 10],
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
