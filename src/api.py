import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf

# =====================
# LOAD MODEL
# =====================

MODEL_PATH = "model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

app = FastAPI(title="Cats vs Dogs Classifier")

# =====================
# UTIL: IMAGE PREPROCESS
# =====================

def preprocess_image(image: Image.Image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype("float32")
    return image

# =====================
# ENDPOINTS
# =====================

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img_tensor = preprocess_image(image)
    prediction = model.predict(img_tensor)[0][0]

    label = "dog" if prediction > 0.5 else "cat"
    confidence = float(prediction) if label == "dog" else float(1 - prediction)

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }
