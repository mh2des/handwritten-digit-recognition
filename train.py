import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# 0) SETUP

cv2.setNumThreads(0)
np.random.seed(42)
tf.random.set_seed(42)

# Automatically detect project directory (works on any device)
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(PROJECT_DIR, 'train')
TEST_DIR = os.path.join(PROJECT_DIR, 'test')

MODEL_PATH = os.path.join(PROJECT_DIR, 'model.keras')
CLASSES_PATH = os.path.join(PROJECT_DIR, 'classes.npy')
RESULT_CSV = os.path.join(PROJECT_DIR, 'result.csv')

assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"


# 1) CONFIG

PATCH = 300
STRIP_H = 300
STRIDE_Y = 80

BATCH_SIZE = 32
STEPS_PER_EPOCH = 250  # raise to 350 if you can wait longer
EPOCHS_HEAD = 20
EPOCHS_FT =  30

LABEL_SMOOTH = 0.05

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
preprocess_fn = tf.keras.applications.mobilenet_v2.preprocess_input


def ensure_u8(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8, copy=False)
    return np.ascontiguousarray(img)


def resize_keep_height(gray_u8, target_h=PATCH):
    h, w = gray_u8.shape
    scale = target_h / float(h)
    new_w = max(target_h, int(round(w * scale)))
    return cv2.resize(gray_u8, (new_w, target_h), interpolation=cv2.INTER_AREA)


def preprocess_for_backbone(gray_u8):
    """Return float32 [0..255], shape (224, W>=224, 3)"""
    g = ensure_u8(gray_u8)
    g = clahe.apply(g)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    g = resize_keep_height(g, PATCH)
    x = g.astype(np.float32)
    x = np.repeat(x[..., None], 3, axis=-1)
    return x



# 2) LOAD TRAIN PAGES

print("Loading training data...")
train_files = sorted([f for f in os.listdir(TRAIN_DIR) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
assert len(train_files) > 0, "No training images found."

labels = [f[:2] for f in train_files]  # first two chars
enc = LabelEncoder()
enc.fit(labels)
num_classes = len(enc.classes_)
np.save(CLASSES_PATH, enc.classes_)
print("Classes:", num_classes)

pages = []
y_int = []
for f in train_files:
    img = cv2.imread(os.path.join(TRAIN_DIR, f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    pages.append(ensure_u8(img))
    y_int.append(int(enc.transform([f[:2]])[0]))
y_int = np.array(y_int, dtype=np.int32)
print("Loaded pages:", len(pages))


# 3) DEFINE TRAIN/VAL STRIP ZONES PER PAGE

train_ypos = []
val_ypos = []

for img in pages:
    h, w = img.shape
    if h <= STRIP_H:
        ys = [0, 0, 0]
    else:
        ys = list(range(0, h - STRIP_H + 1, STRIDE_Y))
        if len(ys) < 3:
            ys = ys + ys + ys
    val = ys[-2:]  # last two strips = validation zone
    trn = ys[:-2]  # rest = training zone
    if len(trn) == 0:
        trn = [ys[0]]
    train_ypos.append(trn)
    val_ypos.append(val)


# 4) BUILD VALIDATION PATCH SET (fixed)

def build_val_set():
    Xv, yv = [], []
    for img, lab, ys in zip(pages, y_int, val_ypos):
        for y0 in ys:
            strip = img[y0:y0 + STRIP_H, :] if img.shape[0] > STRIP_H else img
            xprep = preprocess_for_backbone(strip)  # (224, W, 3)
            W = xprep.shape[1]
            if W <= PATCH:
                Xv.append(xprep)
                yv.append(lab)
            else:
                xs = np.linspace(0, W - PATCH, num=6, dtype=int).tolist()
                for x0 in xs:
                    Xv.append(xprep[:, x0:x0 + PATCH, :])
                    yv.append(lab)
    Xv = np.stack(Xv).astype(np.float32)
    yv = tf.keras.utils.to_categorical(np.array(yv, dtype=np.int32), 
                                       num_classes=num_classes).astype(np.float32)
    return Xv, yv


print("Building validation set...")
X_val, y_val = build_val_set()
print("Val patches:", X_val.shape, y_val.shape)



# 5) TRAIN PATCH GENERATOR

class PatchSequence(tf.keras.utils.Sequence):
    def __init__(self, pages, labels_int, train_ypos, batch_size, steps, num_classes):
        super().__init__()
        self.pages = pages
        self.labels = labels_int
        self.train_ypos = train_ypos
        self.bs = batch_size
        self.steps = steps
        self.C = num_classes

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.bs, PATCH, PATCH, 3), dtype=np.float32)
        y = np.zeros((self.bs, self.C), dtype=np.float32)

        for i in range(self.bs):
            j = np.random.randint(0, len(self.pages))
            page = self.pages[j]
            lab = int(self.labels[j])

            # pick strip only from training y-positions
            ys = self.train_ypos[j]
            y0 = ys[np.random.randint(0, len(ys))] if len(ys) else 0
            strip = page[y0:y0 + STRIP_H, :] if page.shape[0] > STRIP_H else page

            xprep = preprocess_for_backbone(strip)  # (224, W, 3)
            W = xprep.shape[1]
            if W <= PATCH:
                patch = xprep
            else:
                x0 = np.random.randint(0, W - PATCH + 1)
                patch = xprep[:, x0:x0 + PATCH, :]

            X[i] = patch
            y[i, lab] = 1.0

        return X, y


train_seq = PatchSequence(pages, y_int, train_ypos, BATCH_SIZE, STEPS_PER_EPOCH, num_classes)


# 6) MODEL (transfer learning)

print("\nBuilding model...")
aug = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.02),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.05),
    tf.keras.layers.RandomContrast(0.2),
], name="augment")

base = tf.keras.applications.MobileNetV2(
    input_shape=(PATCH, PATCH, 3),
    include_top=False,
    weights="imagenet"
)
base.trainable = False

inp = tf.keras.Input(shape=(PATCH, PATCH, 3))
x = aug(inp)
x = tf.keras.layers.Lambda(preprocess_fn)(x)  # [0..255] -> [-1..1]
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.5)(x)
out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
    metrics=["accuracy"]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6)
]

print("\n--- Train head (frozen backbone) ---")
model.fit(train_seq, epochs=EPOCHS_HEAD, validation_data=(X_val, y_val), 
          callbacks=callbacks, verbose=1)

# Fine-tune last layers (keep BatchNorm frozen)
print("\n--- Fine-tune last backbone layers ---")
for layer in base.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
for layer in base.layers[-40:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.03),
    metrics=["accuracy"]
)

model.fit(train_seq, epochs=EPOCHS_FT, validation_data=(X_val, y_val), 
          callbacks=callbacks, verbose=1)

model.save(MODEL_PATH)
print("\nSaved model:", MODEL_PATH)
print("Saved classes:", CLASSES_PATH)



# 7) TEST EVAL

def multi_patch_predict(model, strip_u8, n_patches=35):
    xprep = preprocess_for_backbone(strip_u8)  # (224, W, 3)
    W = xprep.shape[1]
    if W <= PATCH:
        return model.predict(xprep[None, ...], verbose=0)[0]

    # evenly spaced + random positions
    xs = np.linspace(0, W - PATCH, num=min(15, W - PATCH + 1), dtype=int).tolist()
    while len(xs) < n_patches:
        xs.append(np.random.randint(0, W - PATCH + 1))

    patches = np.stack([xprep[:, x0:x0 + PATCH, :] for x0 in xs], axis=0).astype(np.float32)
    probs = model.predict(patches, verbose=0)
    return probs.mean(axis=0)


print("\n--- Evaluating on test set ---")
test_files = sorted([f for f in os.listdir(TEST_DIR) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
rows = []
correct = 0
total = 0

for f in test_files:
    true_lab = f[:2]
    true_i = int(enc.transform([true_lab])[0])

    img = cv2.imread(os.path.join(TEST_DIR, f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = ensure_u8(img)

    probs = multi_patch_predict(model, img, n_patches=35)
    pred_i = int(np.argmax(probs))
    pred_lab = enc.inverse_transform([pred_i])[0]

    correct += int(pred_i == true_i)
    total += 1
    rows.append([f, true_lab, pred_lab])

acc = correct / max(1, total)
print(f"\nTEST accuracy: {acc:.4f} ({correct}/{total})")

pd.DataFrame(rows, columns=["filename", "actual", "predicted"]).to_csv(RESULT_CSV, index=False)
print("Saved:", RESULT_CSV)

print("\nTraining complete!")
