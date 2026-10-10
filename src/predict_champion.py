"""Langkah 7b: muat model via models:/<nama>@champion lalu prediksi."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import mlflow
import numpy as np

from common import CLASS_NAMES, MODEL_NAME, load_data, setup_mlflow

if __name__ == "__main__":
    setup_mlflow()
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
    data = load_data()
    x = (data["x_test"].astype(np.float32) / 255.0)[:, None, :, :]
    logits = np.asarray(model.predict(x[:10]))
    pred = logits.argmax(1)
    for i, (p, t) in enumerate(zip(pred, data["y_test"][:10])):
        print(f"[{i}] prediksi={CLASS_NAMES[p]:12s} | label asli={CLASS_NAMES[t]}")
