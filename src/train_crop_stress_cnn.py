import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# AeroCrop Drone AI - CNN Crop Stress & Pest Classification
# ============================================================

DATA_DIR = "data/drone_image_dataset"
MODEL_PATH = "models/crop_stress_cnn.keras"
CLASS_PATH = "models/crop_stress_classes.json"
CONFUSION_PATH = "models/cnn_confusion_matrix.png"
HISTORY_PATH = "models/cnn_training_history.png"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
RANDOM_SEED = 42

print("=" * 70)
print("AeroCrop Drone AI - CNN Drone Image Intelligence")
print("=" * 70)

# ------------------------------------------------------------
# 1. Verify Dataset
# ------------------------------------------------------------

print("\nChecking drone image dataset...")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_DIR}"
    )

classes = sorted([
    folder
    for folder in os.listdir(DATA_DIR)
    if os.path.isdir(
        os.path.join(DATA_DIR, folder)
    )
])

print(f"Classes detected: {classes}")

if len(classes) != 3:
    raise ValueError(
        f"Expected 3 classes, found {len(classes)}"
    )

# ------------------------------------------------------------
# 2. Image Data Generator
# ------------------------------------------------------------

print("\nPreparing image data pipeline...")

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    validation_split=0.20,
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.10,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=RANDOM_SEED
)

validation_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=RANDOM_SEED
)

print(
    f"\nTraining images: "
    f"{train_generator.samples}"
)

print(
    f"Validation images: "
    f"{validation_generator.samples}"
)

print(
    f"Class mapping: "
    f"{train_generator.class_indices}"
)

# ------------------------------------------------------------
# 3. Build CNN Model
# ------------------------------------------------------------

print("\nBuilding CNN model...")

model = Sequential([
    Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            3
        )
    ),

    # Block 1
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(
        (2, 2)
    ),

    # Block 2
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(
        (2, 2)
    ),

    # Block 3
    Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(
        (2, 2)
    ),

    # Block 4
    Conv2D(
        256,
        (3, 3),
        activation="relu",
        padding="same"
    ),
    BatchNormalization(),
    MaxPooling2D(
        (2, 2)
    ),

    Flatten(),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(0.40),

    Dense(
        3,
        activation="softmax"
    )
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------------------------------------
# 4. Callbacks
# ------------------------------------------------------------

os.makedirs(
    "models",
    exist_ok=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

# ------------------------------------------------------------
# 5. Train CNN
# ------------------------------------------------------------

print("\nTraining CNN model...")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ],
    verbose=1
)

print("\nCNN training completed.")

# ------------------------------------------------------------
# 6. Evaluate Model
# ------------------------------------------------------------

print("\nEvaluating CNN model...")

validation_generator.reset()

loss, accuracy = model.evaluate(
    validation_generator,
    verbose=0
)

print(
    f"Validation Loss: "
    f"{loss:.6f}"
)

print(
    f"Validation Accuracy: "
    f"{accuracy * 100:.2f}%"
)

# ------------------------------------------------------------
# 7. Predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

validation_generator.reset()

probabilities = model.predict(
    validation_generator,
    verbose=1
)

predicted_classes = np.argmax(
    probabilities,
    axis=1
)

true_classes = validation_generator.classes

class_names = list(
    validation_generator.class_indices.keys()
)

# ------------------------------------------------------------
# 8. Classification Report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CNN CLASSIFICATION REPORT")
print("=" * 70)

report = classification_report(
    true_classes,
    predicted_classes,
    target_names=class_names,
    digits=4
)

print(report)

# ------------------------------------------------------------
# 9. Confusion Matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

plt.figure(
    figsize=(8, 6)
)

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "AeroCrop CNN Confusion Matrix"
)

plt.colorbar()

tick_marks = np.arange(
    len(class_names)
)

plt.xticks(
    tick_marks,
    class_names,
    rotation=45
)

plt.yticks(
    tick_marks,
    class_names
)

threshold = cm.max() / 2.0

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white"
            if cm[i, j] > threshold
            else "black"
        )

plt.ylabel(
    "Actual Class"
)

plt.xlabel(
    "Predicted Class"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nConfusion matrix saved to:"
    f"\n{CONFUSION_PATH}"
)

# ------------------------------------------------------------
# 10. Training History Plot
# ------------------------------------------------------------

print("\nGenerating training history plot...")

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "AeroCrop CNN Training Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    HISTORY_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(
    f"Training history saved to:"
    f"\n{HISTORY_PATH}"
)

# ------------------------------------------------------------
# 11. Save Class Mapping
# ------------------------------------------------------------

class_mapping = {
    str(index): class_name
    for class_name, index
    in train_generator.class_indices.items()
}

with open(
    CLASS_PATH,
    "w"
) as file:

    json.dump(
        class_mapping,
        file,
        indent=4
    )

print(
    f"\nClass mapping saved to:"
    f"\n{CLASS_PATH}"
)

# ------------------------------------------------------------
# 12. Final Status
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CNN DRONE IMAGE INTELLIGENCE READY")
print("=" * 70)

print("\nGenerated files:")

print(
    f"1. {MODEL_PATH}"
)

print(
    f"2. {CLASS_PATH}"
)

print(
    f"3. {CONFUSION_PATH}"
)

print(
    f"4. {HISTORY_PATH}"
)

print("\nDetected classes:")

for index, class_name in enumerate(class_names):
    print(
        f"{index}: {class_name}"
    )

print("\nPhase 6 CNN training completed successfully.")