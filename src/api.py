import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference")

REQUEST_COUNT = 0
TOTAL_LATENCY = 0

MODEL_PATH = "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224
app = FastAPI(title="Cats vs Dogs Classifier")

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype("float32")
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global REQUEST_COUNT, TOTAL_LATENCY

    start = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = preprocess_image(image)

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    avg_latency = TOTAL_LATENCY / REQUEST_COUNT if REQUEST_COUNT else 0
    return {
        "requests": REQUEST_COUNT,
        "avg_latency": round(avg_latency, 4)
    }