import os
import cv2
import random
from sklearn.model_selection import train_test_split

# =====================
# CONFIG
# =====================

RAW_DIR = "data/raw/extracted"
OUTPUT_DIR = "data/processed"
IMG_SIZE = 224

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

# =====================
# LOAD IMAGE PATHS
# =====================

def load_image_paths():
    base = RAW_DIR

    # If PetImages exists inside extracted, use it
    if os.path.exists(os.path.join(base, "PetImages")):
        base = os.path.join(base, "PetImages")

    cats_dir = os.path.join(base, "Cat")
    dogs_dir = os.path.join(base, "Dog")

    if not os.path.exists(cats_dir) or not os.path.exists(dogs_dir):
        raise FileNotFoundError("Could not find Cat and Dog folders in extracted dataset")

    cats = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
    dogs = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

    print(f"Cats found: {len(cats)}")
    print(f"Dogs found: {len(dogs)}")

    X = cats + dogs
    y = ["cat"] * len(cats) + ["dog"] * len(dogs)

    combined = list(zip(X, y))
    random.shuffle(combined)

    X, y = zip(*combined)
    return list(X), list(y)

    base = RAW_DIR

    # If PetImages exists inside extracted, use it
    if os.path.exists(os.path.join(base, "PetImages")):
        base = os.path.join(base, "PetImages")

    cats_dir = os.path.join(base, "Cat")
    dogs_dir = os.path.join(base, "Dog")

    if not os.path.exists(cats_dir) or not os.path.exists(dogs_dir):
        raise FileNotFoundError("Could not find Cat and Dog folders in extracted dataset")

    cats = [os.path.join(cats_dir, f) for f in os.listdir(cats_dir)]
    dogs = [os.path.join(dogs_dir, f) for f in os.listdir(dogs_dir)]

    print(f"Cats found: {len(cats)}")
    print(f"Dogs found: {len(dogs)}")

    X = cats + dogs
    y = ["cat"] * len(cats) + ["dog"] * len(dogs)

    combined = list(zip(X, y))
    random.shuffle(combined)

    X, y = zip(*combined)
    return list(X), list(y)

# =====================
# SAVE SPLIT SAFELY
# =====================

def save_split(X, y, split_name):
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_dir, exist_ok=True)

    skipped = 0
    saved = 0

    for i, (img_path, label) in enumerate(zip(X, y)):
        img = cv2.imread(img_path)

        # Skip corrupted images
        if img is None:
            skipped += 1
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        class_dir = os.path.join(split_dir, label)
        os.makedirs(class_dir, exist_ok=True)

        cv2.imwrite(os.path.join(class_dir, f"{saved}.jpg"), img)
        saved += 1

    print(f"{split_name} split done — saved {saved}, skipped {skipped}")

# =====================
# MAIN PIPELINE
# =====================

if __name__ == "__main__":
    print("Loading dataset...")
    X, y = load_image_paths()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    print("Saving splits...")

    save_split(X_train, y_train, "train")
    save_split(X_val, y_val, "val")
    save_split(X_test, y_test, "test")

    print("Preprocessing complete ✅")