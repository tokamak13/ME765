import h5py, numpy as np
from tensorflow import keras

from pathlib import Path
Path("layers").mkdir(parents=True, exist_ok=True)

model = keras.models.load_model("nut_model.keras")
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:
        np.save(f"layers/layer{i}_W.npy", w[0])
        np.save(f"layers/layer{i}_b.npy", w[1])