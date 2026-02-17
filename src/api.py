import io
import os
import time
import logging

import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

# =====================
# LOGGING & METRICS
# =====================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

REQUEST_COUNT = 0
TOTAL_LATENCY = 0

# =====================
# MODEL LOADING (CI SAFE)
# =====================

MODEL_PATH = "model.keras"
model = None

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
else:
    logger.warning("Model not found — running in CI mode")

# =====================
# APP SETUP
# =====================

IMG_SIZE = 224
app = FastAPI(title="Cats vs Dogs Classifier")

# =====================
# IMAGE PREPROCESS
# =====================

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype("float32")
    return image

# =====================
# PREDICTION ENDPOINT
# =====================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global REQUEST_COUNT, TOTAL_LATENCY

    start = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = preprocess_image(image)

    # CI safety
    if model is None:
        return {"error": "Model not loaded (CI mode)"}

    prediction = model.predict(img_tensor)[0][0]

    latency = time.time() - start
    REQUEST_COUNT += 1
    TOTAL_LATENCY += latency

    logger.info(f"latency={latency:.3f}s requests={REQUEST_COUNT}")

    label = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction) if label == "dog" else float(1 - prediction)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }

# =====================
# HEALTH CHECK
# =====================

@app.get("/health")
def health():
    return {"status": "ok"}

# =====================
# METRICS
# =====================

@app.get("/metrics")
def metrics():
    avg_latency = TOTAL_LATENCY / REQUEST_COUNT if REQUEST_COUNT else 0
    return {
        "requests": REQUEST_COUNT,
        "avg_latency": round(avg_latency, 4)
    }