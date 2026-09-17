import os
import shutil
from collections import defaultdict

from datasets import load_dataset


# ============================================================
# SETTINGS
# ============================================================

DATASET_NAME = "geraldmc/plantvillage-full"
DATASET_REVISION = "v0.1.0"

OUTPUT_DIR = "dataset"

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VALIDATION_DIR = os.path.join(OUTPUT_DIR, "validation")


# Exact class labels confirmed from the dataset
SELECTED_CLASSES = [
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]


MAX_TRAIN_PER_CLASS = 800
MAX_VALIDATION_PER_CLASS = 200


# ============================================================
# CLEAN OLD PREPARED DATA
# ============================================================

print("\nCleaning previous prepared dataset folders...")

if os.path.exists(TRAIN_DIR):
    shutil.rmtree(TRAIN_DIR)

if os.path.exists(VALIDATION_DIR):
    shutil.rmtree(VALIDATION_DIR)


# ============================================================
# CREATE NEW FOLDERS
# ============================================================

for class_name in SELECTED_CLASSES:

    os.makedirs(
        os.path.join(TRAIN_DIR, class_name),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(VALIDATION_DIR, class_name),
        exist_ok=True
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\n==========================================")
print("Loading PlantVillage dataset...")
print("==========================================\n")

dataset = load_dataset(
    DATASET_NAME,
    revision=DATASET_REVISION
)

data = dataset["train"]

print("Dataset loaded successfully.")
print("Total images:", len(data))

print("\nColumns:")
print(data.column_names)


# ============================================================
# CHECK EXACT CLASSES
# ============================================================

available_classes = set(data["class_label"])

print("\nChecking selected classes:")

for class_name in SELECTED_CLASSES:

    if class_name in available_classes:
        print(f"FOUND  : {class_name}")
    else:
        print(f"MISSING: {class_name}")


# ============================================================
# CHECK SPLITS
# ============================================================

print("\nAvailable split values:")

available_splits = sorted(
    set(data["split"])
)

for split in available_splits:
    print("-", split)


# ============================================================
# COUNTERS
# ============================================================

train_counts = defaultdict(int)
validation_counts = defaultdict(int)


# ============================================================
# SAVE IMAGES
# ============================================================

print("\n==========================================")
print("Preparing images...")
print("==========================================\n")


for item in data:

    class_name = item["class_label"]
    split = item["split"]

    # Ignore classes outside our project
    if class_name not in SELECTED_CLASSES:
        continue


    # --------------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------------

    if split == "train":

        if train_counts[class_name] >= MAX_TRAIN_PER_CLASS:
            continue

        output_folder = os.path.join(
            TRAIN_DIR,
            class_name
        )

        image_number = train_counts[class_name]

        filename = f"image_{image_number:04d}.jpg"

        output_path = os.path.join(
            output_folder,
            filename
        )

        item["image"].convert("RGB").save(
            output_path,
            quality=95
        )

        train_counts[class_name] += 1


    # --------------------------------------------------------
    # VALIDATION DATA
    # --------------------------------------------------------

    elif split == "test":

        if validation_counts[class_name] >= MAX_VALIDATION_PER_CLASS:
            continue

        output_folder = os.path.join(
            VALIDATION_DIR,
            class_name
        )

        image_number = validation_counts[class_name]

        filename = f"image_{image_number:04d}.jpg"

        output_path = os.path.join(
            output_folder,
            filename
        )

        item["image"].convert("RGB").save(
            output_path,
            quality=95
        )

        validation_counts[class_name] += 1


# ============================================================
# DISPLAY TRAINING COUNTS
# ============================================================

print("\n==========================================")
print("TRAINING IMAGES")
print("==========================================")

for class_name in SELECTED_CLASSES:

    print(
        f"{class_name}: "
        f"{train_counts[class_name]}"
    )


# ============================================================
# DISPLAY VALIDATION COUNTS
# ============================================================

print("\n==========================================")
print("VALIDATION IMAGES")
print("==========================================")

for class_name in SELECTED_CLASSES:

    print(
        f"{class_name}: "
        f"{validation_counts[class_name]}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n==========================================")
print("FINAL DATASET CHECK")
print("==========================================")

problem = False


for class_name in SELECTED_CLASSES:

    train_count = train_counts[class_name]
    validation_count = validation_counts[class_name]

    if train_count == 0:

        print(
            f"ERROR: No training images for "
            f"{class_name}"
        )

        problem = True


    if validation_count == 0:

        print(
            f"ERROR: No validation images for "
            f"{class_name}"
        )

        problem = True


# ============================================================
# FINAL MESSAGE
# ============================================================

if not problem:

    print("\n==========================================")
    print("✅ DATASET PREPARATION COMPLETE!")
    print("==========================================")

    print("\nDataset location:")
    print(
        os.path.abspath(OUTPUT_DIR)
    )

else:

    print("\n==========================================")
    print("⚠ DATASET PREPARATION FINISHED WITH ERRORS")
    print("==========================================")