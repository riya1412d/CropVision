import os
import json
import numpy as np

from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/cropvision_model.keras"
VALIDATION_DIR = "dataset/validation"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


print("\nModel classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\nLoading validation dataset...")

validation_ds = keras.utils.image_dataset_from_directory(

    VALIDATION_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


print("\nFolder classes detected by Keras:")

for i, name in enumerate(validation_ds.class_names):
    print(f"{i}: {name}")


# ============================================================
# CHECK CLASS ORDER
# ============================================================

if class_names != validation_ds.class_names:

    print("\nWARNING!")
    print("Model class order and folder class order are different.")

    print("\nModel:")
    print(class_names)

    print("\nFolders:")
    print(validation_ds.class_names)

else:

    print("\n✅ Class order matches.")


# ============================================================
# PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in validation_ds:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_labels = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(
        labels.numpy()
    )

    y_pred.extend(
        predicted_labels
    )


# ============================================================
# ACCURACY
# ============================================================

y_true = np.array(y_true)
y_pred = np.array(y_pred)

accuracy = np.mean(
    y_true == y_pred
)


print("\n==========================================")
print("CURRENT MODEL VERIFICATION")
print("==========================================")

print(
    f"\nTotal validation images: {len(y_true)}"
)

print(
    f"Correct predictions: "
    f"{np.sum(y_true == y_pred)}"
)

print(
    f"Incorrect predictions: "
    f"{np.sum(y_true != y_pred)}"
)

print(
    f"\nAccuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nRows = Actual")
print("Columns = Predicted\n")

print(cm)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n==========================================")
print("PER-CLASS RESULTS")
print("==========================================")

for i, class_name in enumerate(class_names):

    class_indices = (
        y_true == i
    )

    total = np.sum(class_indices)

    correct = np.sum(
        y_pred[class_indices] == i
    )

    if total > 0:
        class_accuracy = (
            correct / total
        ) * 100
    else:
        class_accuracy = 0

    print(
        f"{class_name}: "
        f"{correct}/{total} "
        f"({class_accuracy:.2f}%)"
    )


# ============================================================
# FINISHED
# ============================================================

print("\n==========================================")
print("✅ VERIFICATION COMPLETE")
print("==========================================")