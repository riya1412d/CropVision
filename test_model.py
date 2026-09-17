import json
import os
import shutil

import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = "models/cropvision_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"

VALIDATION_DIR = "dataset/validation"
TEST_DIR = "test_images"

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

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8"
) as file:

    class_names = json.load(file)


# ============================================================
# CREATE TEST FOLDER
# ============================================================

os.makedirs(
    TEST_DIR,
    exist_ok=True
)


# ============================================================
# FIND FIRST IMAGE IN EACH CLASS
# ============================================================

results = []


print("\n==========================================")
print("TESTING ONE IMAGE FROM EACH CLASS")
print("==========================================\n")


for class_name in class_names:

    class_folder = os.path.join(
        VALIDATION_DIR,
        class_name
    )

    if not os.path.exists(class_folder):

        print(
            f"Folder not found: {class_folder}"
        )

        continue


    # Find first image
    image_files = [
        file_name
        for file_name in os.listdir(class_folder)
        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]


    if not image_files:

        print(
            f"No image found for: {class_name}"
        )

        continue


    image_files.sort()

    source_file = os.path.join(
        class_folder,
        image_files[0]
    )


    # --------------------------------------------------------
    # COPY IMAGE TO TEST FOLDER
    # --------------------------------------------------------

    safe_name = (
        class_name
        .replace(",", "")
        .replace(",", "")
        .replace("___", "_")
    )

    destination_file = os.path.join(
        TEST_DIR,
        f"{safe_name}.jpg"
    )

    shutil.copy2(
        source_file,
        destination_file
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = Image.open(
        source_file
    ).convert("RGB")


    resized = image.resize(
        IMAGE_SIZE
    )


    image_array = np.array(
        resized,
        dtype=np.float32
    )


    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    image_array = preprocess_input(
        image_array
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]


    predicted_index = int(
        np.argmax(predictions)
    )


    predicted_class = class_names[
        predicted_index
    ]


    confidence = (
        float(
            predictions[predicted_index]
        )
        * 100
    )


    # --------------------------------------------------------
    # CHECK
    # --------------------------------------------------------

    correct = (
        class_name == predicted_class
    )


    results.append(
        {
            "actual": class_name,
            "predicted": predicted_class,
            "confidence": confidence,
            "correct": correct
        }
    )


    # --------------------------------------------------------
    # PRINT RESULT
    # --------------------------------------------------------

    print(
        "Actual    :",
        class_name
    )

    print(
        "Predicted :",
        predicted_class
    )

    print(
        f"Confidence: {confidence:.2f}%"
    )

    print(
        "Result    :",
        "✅ CORRECT" if correct else "❌ INCORRECT"
    )

    print("-" * 55)


# ============================================================
# SUMMARY
# ============================================================

print("\n==========================================")
print("TEST SUMMARY")
print("==========================================\n")


correct_count = sum(
    result["correct"]
    for result in results
)

total_count = len(results)


for result in results:

    print(
        f"{result['actual']}"
        f" → "
        f"{result['predicted']}"
        f" "
        f"({result['confidence']:.2f}%)"
    )


print("\n------------------------------------------")


if total_count > 0:

    print(
        f"Correct: {correct_count}/{total_count}"
    )

    print(
        f"Test accuracy: "
        f"{(correct_count / total_count) * 100:.2f}%"
    )


print("------------------------------------------")

print(
    "\nTest images copied to:"
)

print(
    os.path.abspath(TEST_DIR)
)

print(
    "\nCropVision test completed."
)