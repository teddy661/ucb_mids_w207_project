import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_NAME = "model_stage_one"
MODEL_DIR = Path("./model_saves/" + MODEL_NAME).resolve()

print(MODEL_DIR.joinpath(MODEL_NAME + "_history"))
history = pickle.load(open(MODEL_DIR.joinpath(MODEL_NAME + "_history"), "rb"))


hist = history
x_arr = np.arange(len(hist["loss"][-30:])) + 1

fig = plt.figure(figsize=(12, 4))
ax = fig.add_subplot(1, 2, 1)
ax.plot(x_arr, hist["loss"][-30:], "-o", label="Train loss")
ax.plot(x_arr, hist["val_loss"][-30:], "--<", label="Validation loss")
ax.legend(fontsize=15)
ax.set_xlabel("Epoch", size=15)
ax.set_ylabel("Loss", size=15)

print(hist["val_loss"][-30:])
plt.show()

print("done")
