import os
import json

import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/cropvision_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

IMAGE_PATH = "test_images/Potato_Late_blight.jpg"

OUTPUT_PATH = "models/gradcam_result.png"

IMAGE_SIZE = (224, 224)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading CropVision model...")

model = keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")


# ============================================================
# LOAD IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

original_image = Image.open(
    IMAGE_PATH
).convert("RGB")

resized_image = original_image.resize(
    IMAGE_SIZE
)

image_array = np.array(
    resized_image,
    dtype=np.float32
)

input_image = np.expand_dims(
    image_array,
    axis=0
)


# ============================================================
# FIND IMPORTANT LAYERS
# ============================================================

base_model = None
augmentation_layer = None
pooling_layer = None
dropout_layer = None
classifier_layer = None


for layer in model.layers:

    layer_name = layer.name.lower()

    # MobileNetV2
    if isinstance(layer, keras.Model):
        if "mobilenet" in layer_name:
            base_model = layer

    # Data augmentation
    if layer_name == "data_augmentation":
        augmentation_layer = layer

    # Global Average Pooling
    if isinstance(
        layer,
        keras.layers.GlobalAveragePooling2D
    ):
        pooling_layer = layer

    # Dropout
    if isinstance(
        layer,
        keras.layers.Dropout
    ):
        dropout_layer = layer

    # Final classifier
    if isinstance(
        layer,
        keras.layers.Dense
    ):
        classifier_layer = layer


if base_model is None:
    raise RuntimeError(
        "MobileNetV2 base model was not found."
    )

if pooling_layer is None:
    raise RuntimeError(
        "GlobalAveragePooling2D layer was not found."
    )

if classifier_layer is None:
    raise RuntimeError(
        "Final Dense classifier layer was not found."
    )


print("\nUsing layers:")
print("Base model:", base_model.name)
print("Pooling:", pooling_layer.name)

if dropout_layer is not None:
    print("Dropout:", dropout_layer.name)

print("Classifier:", classifier_layer.name)


# ============================================================
# BUILD GRAD-CAM GRAPH
# ============================================================

grad_input = keras.Input(
    shape=(224, 224, 3)
)

x = grad_input


# Same augmentation layer used during training.
# In inference mode, random augmentation is disabled.
if augmentation_layer is not None:
    x = augmentation_layer(
        x,
        training=False
    )


# Same MobileNetV2 preprocessing
x = preprocess_input(x)


# Get final convolutional feature map
conv_outputs = base_model(
    x,
    training=False
)


# Recreate classifier head
x = pooling_layer(
    conv_outputs
)

if dropout_layer is not None:
    x = dropout_layer(
        x,
        training=False
    )

predictions = classifier_layer(
    x
)


grad_model = keras.Model(
    inputs=grad_input,
    outputs=[
        conv_outputs,
        predictions
    ]
)


# ============================================================
# CALCULATE GRADIENTS
# ============================================================

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(
        input_image,
        training=False
    )

    predicted_index = tf.argmax(
        predictions[0]
    )

    predicted_score = predictions[
        0,
        predicted_index
    ]


grads = tape.gradient(
    predicted_score,
    conv_outputs
)


if grads is None:
    raise RuntimeError(
        "Gradients could not be calculated."
    )


# ============================================================
# GLOBAL AVERAGE POOLING
# ============================================================

pooled_grads = tf.reduce_mean(
    grads,
    axis=(0, 1, 2)
)


# ============================================================
# GENERATE HEATMAP
# ============================================================

conv_outputs = conv_outputs[0]

heatmap = tf.reduce_sum(
    conv_outputs * pooled_grads,
    axis=-1
)

heatmap = tf.maximum(
    heatmap,
    0
)

max_heatmap = tf.reduce_max(
    heatmap
)

if max_heatmap > 0:
    heatmap = heatmap / max_heatmap

heatmap = heatmap.numpy()


# ============================================================
# RESIZE HEATMAP
# ============================================================

heatmap_image = Image.fromarray(
    np.uint8(heatmap * 255)
)

heatmap_image = heatmap_image.resize(
    original_image.size
)

heatmap = np.array(
    heatmap_image
) / 255.0


# ============================================================
# PREDICTION DETAILS
# ============================================================

prediction_index = int(
    predicted_index.numpy()
)

confidence = float(
    predictions[
        0,
        prediction_index
    ].numpy()
)

display_name = class_names[
    prediction_index
].replace(
    "_",
    " "
)


# ============================================================
# SAVE GRAD-CAM IMAGE
# ============================================================

plt.figure(
    figsize=(12, 5)
)

# ------------------------------------------------------------
# Original
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    1
)

plt.imshow(
    original_image
)

plt.title(
    "Original Image"
)

plt.axis("off")


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------

plt.subplot(
    1,
    2,
    2
)

plt.imshow(
    original_image
)

plt.imshow(
    heatmap,
    cmap="jet",
    alpha=0.30
)

plt.title(
    f"Grad-CAM\n"
    f"{display_name}\n"
    f"{confidence * 100:.1f}% confidence"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# RESULT
# ============================================================

print("\n==========================================")
print("GRAD-CAM COMPLETE")
print("==========================================")

print(
    f"Prediction : {display_name}"
)

print(
    f"Confidence : {confidence * 100:.2f}%"
)

print(
    f"Saved to   : {OUTPUT_PATH}"
)

print("==========================================\n")