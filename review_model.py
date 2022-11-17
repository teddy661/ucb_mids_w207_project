import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
MODEL_DIR = Path("./model_saves").resolve()
FINAL_MODEL_NAME="final-model"

model = tf.keras.models.load_model(MODEL_DIR.joinpath(FINAL_MODEL_NAME))
model.summary()

print(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"))
history = pickle.load(open(MODEL_DIR.joinpath(FINAL_MODEL_NAME + "_history"), 'rb'))

hist = history
x_arr = np.arange(len(hist["loss"])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist["loss"], "-o", label="Train loss")
ax.plot(x_arr, hist["val_loss"], "--<", label="Validation loss")
ax.legend(fontsize=15)
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Loss", size=15)

print(hist['val_loss'])
plt.show()