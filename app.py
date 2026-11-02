import io
import threading
from urllib.parse import urlparse

import numpy as np
import pillow_heif
import requests
from ai_edge_litert.interpreter import Interpreter
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

CLASS_NAMES = ["bin", "clean", "litter"]   # same order as training

interp = Interpreter(model_path="litter_model.tflite")
interp.allocate_tensors()
IN = interp.get_input_details()[0]["index"]
OUT = interp.get_output_details()[0]["index"]
lock = threading.Lock()

app = FastAPI(title="EcoQuest classifier")


def passes(step: int, probs: dict) -> bool:
    # Provisional thresholds from the 18-photo phone test; tune here, no app update needed.
    if step == 1:
        return probs["litter"] >= 0.6
    if step == 2:
        return probs["bin"] >= 0.25
    if step == 3:
        return probs["litter"] < 0.6
    return False


@app.get("/")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict(url: str, step: int = 0):
    u = urlparse(url)
    if u.scheme != "https" or u.hostname != "res.cloudinary.com":
        raise HTTPException(400, "Only https://res.cloudinary.com URLs are allowed")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(r.content))).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not download or read the image")

    im = ImageOps.fit(im, (224, 224))
    x = np.expand_dims(np.array(im).astype("float32"), 0)
    with lock:
        interp.set_tensor(IN, x)
        interp.invoke()
        p = interp.get_tensor(OUT)[0]

    probs = {c: round(float(v), 3) for c, v in zip(CLASS_NAMES, p)}
    return {
        "label": CLASS_NAMES[int(p.argmax())],
        "confidence": round(float(p.max()), 3),
        "ok": passes(step, probs),
        **probs,
    }
