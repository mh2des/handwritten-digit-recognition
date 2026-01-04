import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder


# Automatically detect project directory 
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(PROJECT_DIR, "test")
TRAIN_DIR = os.path.join(PROJECT_DIR, "train")
MODEL_PATH = os.path.join(PROJECT_DIR, "model.keras")
CLASSES_PATH = os.path.join(PROJECT_DIR, "classes.npy")
RESULT_CSV = os.path.join(PROJECT_DIR, "result.csv")

# Match train.py settings
PATCH = 300
STRIP_H = 300
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Voting strength (match train.py's multi_patch_predict)
N_PATCHES = 35    # reduced from 180 for faster inference
N_EVEN    = 15
SEED      = 123


# HELPERS

def ensure_u8(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return np.ascontiguousarray(img)

def resize_keep_height(gray_u8, target_h=PATCH):
    h, w = gray_u8.shape
    if h == 0 or w == 0:
        return np.zeros((target_h, target_h), dtype=np.uint8)
    scale = target_h / float(h)
    new_w = max(target_h, int(round(w * scale)))
    return cv2.resize(gray_u8, (new_w, target_h), interpolation=cv2.INTER_AREA)

def preprocess_for_model(gray_u8):
    """Match train.py's preprocess_for_backbone function."""
    g = ensure_u8(gray_u8)
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g = resize_keep_height(g, PATCH)               # (300, W>=300)
    x = g.astype(np.float32)
    x = np.repeat(x[..., None], 3, axis=-1)        # (300, W, 3) in [0..255]
    return x

def load_classes():
    if os.path.exists(CLASSES_PATH):
        return np.load(CLASSES_PATH, allow_pickle=True)

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Missing {CLASSES_PATH} and train dir not found: {TRAIN_DIR}")

    files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    classes = sorted({f[:2] for f in files})
    arr = np.array(classes, dtype=object)
    np.save(CLASSES_PATH, arr)
    return arr

def load_model_robust():
    # MobileNetV2 preprocessing (matches train.py)
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except (ImportError, ModuleNotFoundError):
        from keras.applications.mobilenet_v2 import preprocess_input

    # Keras 3 safe-mode can block Lambda; try enabling unsafe deserialization.
    try:
        import keras
        if hasattr(keras, "config") and hasattr(keras.config, "enable_unsafe_deserialization"):
            keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    try:
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            safe_mode=False
        )
    except TypeError:
        # older TF/Keras may not support safe_mode=
        return tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"preprocess_input": preprocess_input},
            compile=False
        )

def predict_one_image(model, gray_u8):
    """Top+bottom strip voting + even+random patch voting."""
    rng = np.random.default_rng(SEED)

    h, w = gray_u8.shape
    strips = []
    if h > STRIP_H:
        strips.append(gray_u8[0:STRIP_H, :])
        strips.append(gray_u8[h-STRIP_H:h, :])
    else:
        strips.append(gray_u8)

    strip_probs = []
    for s in strips:
        xprep = preprocess_for_model(s)  # (300, W, 3)
        W = xprep.shape[1]
        if W <= PATCH:
            strip_probs.append(model.predict(xprep[None, ...], verbose=0)[0])
            continue

        max_x = W - PATCH
        xs = np.linspace(0, max_x, num=min(N_EVEN, max_x + 1), dtype=int).tolist()
        while len(xs) < N_PATCHES:
            xs.append(int(rng.integers(0, max_x + 1)))

        patches = np.stack([xprep[:, x0:x0+PATCH, :] for x0 in xs], axis=0).astype(np.float32)
        probs = model.predict(patches, verbose=0).mean(axis=0)
        strip_probs.append(probs)

    return np.mean(strip_probs, axis=0)


# MAIN

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test folder not found: {TEST_DIR}")

    model = load_model_robust()

    classes = load_classes()
    enc = LabelEncoder()
    enc.fit(classes)

    test_files = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    if not test_files:
        raise RuntimeError(f"No images found in: {TEST_DIR}")

    rows = []
    correct = 0
    total = 0

    print(f"Testing {len(test_files)} images...")
    for idx, f in enumerate(test_files):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing {idx+1}/{len(test_files)}...")
        
        true_lab = f[:2]
        true_i = int(enc.transform([true_lab])[0])

        img = cv2.imread(os.path.join(TEST_DIR, f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        probs = predict_one_image(model, img)
        pred_i = int(np.argmax(probs))
        pred_lab = enc.inverse_transform([pred_i])[0]

        correct += int(pred_i == true_i)
        total += 1
        rows.append([f, true_lab, pred_lab])

    acc = correct / max(1, total)
    print(f"TEST accuracy (N_PATCHES={N_PATCHES}): {acc*100:.2f}% ({correct}/{total})")

    pd.DataFrame(rows, columns=["filename", "actual", "predicted"]).to_csv(RESULT_CSV, index=False)
    print("Saved:", RESULT_CSV)

if __name__ == "__main__":
    main()
