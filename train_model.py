import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

EPOCHS = 15

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "cropvision_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    MODEL_DIR,
    "class_names.json"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "training_history.png"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not os.path.exists(VALIDATION_DIR):

    raise FileNotFoundError(
        f"Validation folder not found: {VALIDATION_DIR}"
    )


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("\n==========================================")
print("Loading training dataset...")
print("==========================================")

train_ds = keras.utils.image_dataset_from_directory(

    TRAIN_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=42
)


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

print("\n==========================================")
print("Loading validation dataset...")
print("==========================================")

validation_ds = keras.utils.image_dataset_from_directory(

    VALIDATION_DIR,

    labels="inferred",

    label_mode="int",

    image_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = train_ds.class_names

print("\n==========================================")
print("CLASSES")
print("==========================================")

for index, name in enumerate(class_names):

    print(f"{index}: {name}")


with open(
    CLASS_NAMES_PATH,
    "w"
) as file:

    json.dump(
        class_names,
        file,
        indent=4
    )


# ============================================================
# PERFORMANCE OPTIMIZATION
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    buffer_size=AUTOTUNE
)

validation_ds = validation_ds.prefetch(
    buffer_size=AUTOTUNE
)


# ============================================================
# DATA AUGMENTATION
# ============================================================

data_augmentation = keras.Sequential(

    [
        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.1
        ),

        layers.RandomZoom(
            0.1
        ),

        layers.RandomContrast(
            0.1
        )
    ],

    name="data_augmentation"
)


# ============================================================
# LOAD PRETRAINED MODEL
# ============================================================

print("\n==========================================")
print("Loading MobileNetV2...")
print("==========================================")

base_model = MobileNetV2(

    input_shape=(224, 224, 3),

    include_top=False,

    weights="imagenet"
)


# Freeze pretrained layers
base_model.trainable = False


# ============================================================
# BUILD MODEL
# ============================================================

inputs = keras.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(
    inputs
)

x = preprocess_input(
    x
)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(
    0.35
)(x)

outputs = layers.Dense(

    len(class_names),

    activation="softmax"
)(x)


model = keras.Model(
    inputs,
    outputs
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(
        learning_rate=0.0001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


# ============================================================
# DISPLAY MODEL
# ============================================================

print("\n==========================================")
print("MODEL SUMMARY")
print("==========================================")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=4,

    restore_best_weights=True
)


checkpoint = keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================\n")


history = model.fit(

    train_ds,

    validation_data=validation_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ============================================================
# LOAD BEST SAVED MODEL
# ============================================================

print("\nLoading best saved model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Best model loaded successfully.")


# ============================================================
# TRAINING GRAPHS
# ============================================================

plt.figure(figsize=(10, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "CropVision Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    HISTORY_PATH
)

plt.close()


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n==========================================")
print("EVALUATING MODEL")
print("==========================================")

loss, accuracy = model.evaluate(
    validation_ds,
    verbose=1
)

print(
    f"\nValidation Loss: {loss:.4f}"
)

print(
    f"Validation Accuracy: {accuracy:.4f}"
)

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


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

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(9, 7)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "CropVision Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    tick_marks,
    class_names
)


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )


plt.ylabel(
    "True Label"
)

plt.xlabel(
    "Predicted Label"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n==========================================")
print("✅ CROPVISION TRAINING COMPLETE!")
print("==========================================")

print(
    f"\nModel saved to:"
)

print(
    os.path.abspath(
        MODEL_PATH
    )
)

print(
    "\nTraining graph:"
)

print(
    os.path.abspath(
        HISTORY_PATH
    )
)

print(
    "\nConfusion matrix:"
)

print(
    os.path.abspath(
        CONFUSION_MATRIX_PATH
    )
)