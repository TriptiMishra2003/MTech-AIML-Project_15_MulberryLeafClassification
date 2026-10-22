import os
import glob
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42

def set_seed(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed()
print("TensorFlow version:", tf.__version__)

train_path = r"C:\Users\Tripti Mishra\Downloads\Mulberry Leaf Dataset\Mulberry Leaf Dataset\Training set"
test_path  = r"C:\Users\Tripti Mishra\Downloads\Mulberry Leaf Dataset\Mulberry Leaf Dataset\Test set"

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def list_images(folder):
    return [f for f in glob.glob(os.path.join(folder, "*"))
            if f.lower().endswith(IMG_EXTS)]

classes = sorted([
    folder for folder in os.listdir(train_path)
    if os.path.isdir(os.path.join(train_path, folder))
])

data = []
for class_name in classes:
    train_count = len(list_images(os.path.join(train_path, class_name)))
    test_count  = len(list_images(os.path.join(test_path, class_name)))
    data.append([class_name, train_count, test_count])

df = pd.DataFrame(data, columns=["Class", "Training Images", "Testing Images"])
print(df.to_string(index=False))

print("\nTotal training images:", df["Training Images"].sum())
print("Total testing images:", df["Testing Images"].sum())
print("Total images:", df["Training Images"].sum() + df["Testing Images"].sum())

# EfficientNetB0 in Keras has built-in rescaling + normalization,
# so we feed raw 0–255 pixels (no rescale=1/255).
# Both generators use the same validation_split; Keras splits files
# deterministically, so train and validation never overlap.

train_datagen = ImageDataGenerator(
    validation_split=0.20,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

validation_datagen = ImageDataGenerator(validation_split=0.20)

train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32,
    class_mode="categorical", subset="training",
    shuffle=True, seed=SEED
)

validation_generator = validation_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=32,
    class_mode="categorical", subset="validation",
    shuffle=False, seed=SEED
)

print("Training images:", train_generator.samples)
print("Validation images:", validation_generator.samples)
print("Classes:", train_generator.class_indices)

class_names = list(train_generator.class_indices.keys())
num_classes = len(class_names)

train_counts = np.bincount(train_generator.classes, minlength=num_classes)
val_counts   = np.bincount(validation_generator.classes, minlength=num_classes)
test_counts  = [len(list_images(os.path.join(test_path, c))) for c in class_names]

df_split = pd.DataFrame({
    "Class": class_names,
    "Training": train_counts,
    "Validation": val_counts,
    "Testing": test_counts
})

print(df_split.to_string(index=False))
print("\nTotal Training Images:", df_split["Training"].sum())
print("Total Validation Images:", df_split["Validation"].sum())
print("Total Testing Images:", df_split["Testing"].sum())
print("Total Images:", df_split[["Training", "Validation", "Testing"]].sum().sum())

train_images = []
for c in classes:
    train_images.extend(list_images(os.path.join(train_path, c)))

image = Image.open(train_images[0])
print("Image size:", image.size)
print("Image mode:", image.mode)

# Same 80/20 split as before, but no augmentation and shuffle=False,
# so every image keeps a fixed index. Needed for the curriculum ordering.
plain_datagen = ImageDataGenerator(validation_split=0.20)
plain_kw = dict(target_size=(224, 224), batch_size=32,
                class_mode="sparse", shuffle=False)

tr_plain = plain_datagen.flow_from_directory(train_path, subset="training", **plain_kw)
va_plain = plain_datagen.flow_from_directory(train_path, subset="validation", **plain_kw)
te_plain = ImageDataGenerator().flow_from_directory(test_path, **plain_kw)

# Sanity checks: same split and same class mapping as the baseline
assert tr_plain.filenames == train_generator.filenames
assert va_plain.filenames == validation_generator.filenames
assert te_plain.class_indices == train_generator.class_indices
print("Split and class mapping match the baseline.")

extractor = EfficientNetB0(weights="imagenet", include_top=False,
                           pooling="avg", input_shape=(224, 224, 3))

t0 = time.time()
X_tr = extractor.predict(tr_plain, verbose=1); y_tr = tr_plain.classes
X_va = extractor.predict(va_plain, verbose=1); y_va = va_plain.classes
X_te = extractor.predict(te_plain, verbose=1); y_te = te_plain.classes
print(f"Feature extraction time: {(time.time() - t0) / 60:.2f} minutes")

np.savez("b0_features.npz",
         X_tr=X_tr, y_tr=y_tr, X_va=X_va, y_va=y_va, X_te=X_te, y_te=y_te,
         tr_files=np.array(tr_plain.filepaths))

print("Train:", X_tr.shape, " Val:", X_va.shape, " Test:", X_te.shape)

data = np.load("b0_features.npz")
X_tr, y_tr = data["X_tr"], data["y_tr"]
X_va, y_va = data["X_va"], data["y_va"]
X_te, y_te = data["X_te"], data["y_te"]
tr_files = data["tr_files"]
print("Loaded:", X_tr.shape, X_va.shape, X_te.shape)

def l2_normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)

Z_tr = l2_normalize(X_tr)

# Class prototypes (centres), computed from training data only
centroids = l2_normalize(np.stack([Z_tr[y_tr == c].mean(axis=0)
                                   for c in range(num_classes)]))

sims = Z_tr @ centroids.T                          # cosine similarity to every centre
idx = np.arange(len(y_tr))
own_sim = sims[idx, y_tr]

other = sims.copy()
other[idx, y_tr] = -np.inf
closest_other_sim = other.max(axis=1)
closest_other_cls = other.argmax(axis=1)

margin = own_sim - closest_other_sim               # higher = easier
difficulty = -margin                               # higher = harder
order_easy = np.argsort(difficulty)                # easiest first

print(f"Difficulty range: {difficulty.min():.4f} to {difficulty.max():.4f}")
print(f"Samples closer to another class's centre (margin < 0): "
      f"{(margin < 0).sum()} / {len(margin)} ({(margin < 0).mean() * 100:.1f}%)")

Z_va = l2_normalize(X_va)
sims_va = Z_va @ centroids.T
iv = np.arange(len(y_va))
own_va = sims_va[iv, y_va]
oth_va = sims_va.copy(); oth_va[iv, y_va] = -np.inf
difficulty_va = -(own_va - oth_va.max(axis=1))

# Percentile of each sample's difficulty within its own class (0 = easiest, 1 = hardest)
within_pct = np.zeros(len(y_tr))
for c in range(num_classes):
    m = np.where(y_tr == c)[0]
    ranks = difficulty[m].argsort().argsort()
    within_pct[m] = (ranks + 0.5) / len(m)

# Primary key: within-class percentile; tie-break: raw difficulty
order_easy_bal = np.lexsort((difficulty, within_pct))
print("Balanced ordering created:", order_easy_bal.shape)
