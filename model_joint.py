import pickle

import keras_tuner as kt
import tensorflow as tf

from data.path_utils import get_paths
from model.data_loader import load_data
from model.model_tuner import HyperModelTuner

MODEL_NAME = "model_joint"

_, _, _, _, MODEL_PATH = get_paths()

## Data Preprocessing
# TODO: put this part into the model.fit() function

tf.random.set_seed(1234)
X_train, X_val, y_train, y_val, X_test = load_data(get_clean=True)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}")


tuner = kt.Hyperband(
    HyperModelTuner(),
    objective="val_loss",
    max_trials=5,
    executions_per_trial=2,
    overwrite=True,
    directory=MODEL_PATH.joinpath("tuning"),
    project_name=MODEL_NAME + "_tuning",
)

tuner.search_space_summary()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=2,
    patience=50,
    min_delta=0.0001,
    restore_best_weights=True,
)

# model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
#     "model_checkpoints/{epoch:04d}-{val_loss:.2f}",
#     monitor="val_loss",
#     mode="min",
#     verbose=1,
#     save_weights_only=False,
#     save_best_only=False,
# )

tuner.search(
    X_train,
    y_train,
    epochs=1,
    validation_data=(X_val, y_val),
    verbose=2,
    callbacks=[early_stopping],
)

model: tf.keras.Model = tuner.get_best_models(num_models=1)[0]  # this is the best model
model.summary()

# with open(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"), "wb") as history_file:
#     pickle.dump(history.history, history_file, protocol=pickle.HIGHEST_PROTOCOL)

model.save(MODEL_PATH.joinpath(MODEL_NAME), overwrite=True)
