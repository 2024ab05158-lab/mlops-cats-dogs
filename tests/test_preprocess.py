import numpy as np
from PIL import Image
from src.api import preprocess_image

def test_preprocess_shape():
    img = Image.fromarray(np.zeros((300, 300, 3), dtype=np.uint8))
    output = preprocess_image(img)

    assert output.shape == (1, 224, 224, 3)