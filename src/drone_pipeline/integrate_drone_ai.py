import os
import json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


# ============================================================
# AeroCrop AI
# Multispectral + Thermal → Segmentation → CNN Integration
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MULTISPECTRAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "multispectral"
)

SEGMENTATION_DIR = os.path.join(
    BASE_DIR,
    "data",
    "segmentation"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_stress_cnn.keras"
)

CLASSES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_stress_classes.json"
)

OUTPUT_PATH = os.path.join(
    SEGMENTATION_DIR,
    "drone_ai_zone_analysis.csv"
)


# ============================================================
# Load CNN Classes
# ============================================================

with open(
    CLASSES_PATH,
    "r",
    encoding="utf-8"
) as file:
    class_mapping = json.load(file)

class_names = [
    class_mapping[str(i)]
    for i in range(len(class_mapping))
]


# ============================================================
# Load CNN
# ============================================================

print("=" * 60)
print("AeroCrop AI - Drone AI Integration")
print("=" * 60)

print("\n[1/5] Loading CNN model...")

cnn_model = tf.keras.models.load_model(
    MODEL_PATH
)

print("CNN model loaded successfully.")

print(
    "Classes:",
    ", ".join(class_names)
)


# ============================================================
# Load Multispectral Products
# ============================================================

print("\n[2/5] Loading multispectral + thermal products...")

scene_path = os.path.join(
    MULTISPECTRAL_DIR,
    "aerocrop_multispectral_thermal_scene.npz"
)

scene = np.load(scene_path)

ndvi = scene["ndvi"]
ndre = scene["ndre"]
thermal = scene["thermal"]

print("Multispectral data loaded.")
print("Thermal data loaded.")
print("NDVI loaded.")
print("NDRE loaded.")


# ============================================================
# Load Segmentation
# ============================================================

print("\n[3/5] Loading field-zone segmentation...")

zone_mask_path = os.path.join(
    SEGMENTATION_DIR,
    "field_zone_segmentation.png"
)

zone_mask = cv2.imread(
    zone_mask_path,
    cv2.IMREAD_GRAYSCALE
)

if zone_mask is None:
    raise FileNotFoundError(
        "Field segmentation mask not found."
    )

unique_values = np.unique(
    zone_mask
)

print(
    f"Segmentation mask loaded."
)

print(
    f"Detected mask values: "
    f"{len(unique_values)}"
)


# ============================================================
# Load RGB Composite
# ============================================================

rgb_path = os.path.join(
    MULTISPECTRAL_DIR,
    "drone_rgb_composite.png"
)

rgb = cv2.imread(
    rgb_path
)

if rgb is None:
    raise FileNotFoundError(
        "Drone RGB composite not found."
    )

rgb = cv2.cvtColor(
    rgb,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# CNN Prediction Function
# ============================================================

def predict_crop_condition(image):

    resized = cv2.resize(
        image,
        (128, 128)
    )

    input_image = (
        resized.astype(np.float32)
        / 255.0
    )

    input_image = np.expand_dims(
        input_image,
        axis=0
    )

    probabilities = cnn_model.predict(
        input_image,
        verbose=0
    )[0]

    class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[class_index]
    )

    return (
        class_names[class_index],
        confidence,
        probabilities
    )


# ============================================================
# Zone Analysis
# ============================================================

print("\n[4/5] Running zone-level AI analysis...")

records = []

zone_ids = np.unique(
    zone_mask
)

zone_counter = 0

for raw_zone_id in zone_ids:

    if raw_zone_id == 0:
        continue

    mask = (
        zone_mask == raw_zone_id
    )

    pixel_count = int(
        mask.sum()
    )

    if pixel_count < 20:
        continue

    ys, xs = np.where(mask)

    if len(xs) == 0:
        continue

    x_min = max(
        int(xs.min()) - 4,
        0
    )

    x_max = min(
        int(xs.max()) + 5,
        rgb.shape[1]
    )

    y_min = max(
        int(ys.min()) - 4,
        0
    )

    y_max = min(
        int(ys.max()) + 5,
        rgb.shape[0]
    )

    crop = rgb[
        y_min:y_max,
        x_min:x_max
    ]

    if crop.size == 0:
        continue

    prediction, confidence, probabilities = (
        predict_crop_condition(crop)
    )

    zone_ndvi = float(
        ndvi[mask].mean()
    )

    zone_ndre = float(
        ndre[mask].mean()
    )

    zone_temperature = float(
        thermal[mask].mean()
    )

    if zone_ndvi < 0.25:
        vegetation_status = "Severe Stress"

    elif zone_ndvi < 0.45:
        vegetation_status = "Crop Stress"

    else:
        vegetation_status = "Healthy"

    zone_counter += 1

    records.append(
        {
            "zone_id": (
                f"DRONE-ZONE-{zone_counter:03d}"
            ),
            "source_zone": int(
                raw_zone_id
            ),
            "pixel_count": pixel_count,
            "mean_ndvi": round(
                zone_ndvi,
                4
            ),
            "mean_ndre": round(
                zone_ndre,
                4
            ),
            "mean_thermal_temperature": round(
                zone_temperature,
                2
            ),
            "vegetation_status": (
                vegetation_status
            ),
            "cnn_prediction": prediction,
            "cnn_confidence": round(
                confidence,
                4
            ),
            "healthy_probability": round(
                float(probabilities[
                    class_names.index("healthy")
                ]),
                4
            ),
            "crop_stress_probability": round(
                float(probabilities[
                    class_names.index("crop_stress")
                ]),
                4
            ),
            "pest_zone_probability": round(
                float(probabilities[
                    class_names.index("pest_zone")
                ]),
                4
            )
        }
    )


# ============================================================
# Save Results
# ============================================================

print("\n[5/5] Saving integrated AI results...")

results_df = pd.DataFrame(
    records
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"Zone analyses generated: "
    f"{len(results_df)}"
)

print(
    f"Saved to:\n{OUTPUT_PATH}"
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("DRONE AI INTEGRATION COMPLETED")
print("=" * 60)

if not results_df.empty:

    print("\nCNN Prediction Distribution:")

    print(
        results_df[
            "cnn_prediction"
        ].value_counts()
    )

    print(
        "\nAverage CNN confidence: "
        f"{results_df['cnn_confidence'].mean():.2%}"
    )

    print(
        "\nAverage NDVI: "
        f"{results_df['mean_ndvi'].mean():.4f}"
    )

    print(
        "Average NDRE: "
        f"{results_df['mean_ndre'].mean():.4f}"
    )

    print(
        "Average Thermal Temperature: "
        f"{results_df['mean_thermal_temperature'].mean():.2f}"
    )

else:

    print(
        "No valid segmented zones were available."
    )