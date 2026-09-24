import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

FIELD_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "aerocrop_field_zones.csv"
)

DRONE_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "segmentation",
    "drone_ai_zone_analysis.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spray_priority_xgboost.pkl"
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "spray_priority_features.pkl"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "segmentation",
    "drone_spray_priority_analysis.csv"
)


print("=" * 60)
print("AeroCrop AI - Drone Spray Priority Integration")
print("=" * 60)


# ---------------------------------------------------------
# 1. Load existing field intelligence data
# ---------------------------------------------------------

print("\n[1/5] Loading field intelligence dataset...")

field_df = pd.read_csv(FIELD_DATA_PATH)

print(
    f"Field records loaded: {len(field_df)}"
)


# ---------------------------------------------------------
# 2. Load drone-zone analysis
# ---------------------------------------------------------

print("\n[2/5] Loading segmented drone-zone analysis...")

drone_df = pd.read_csv(DRONE_DATA_PATH)

print(
    f"Drone zones loaded: {len(drone_df)}"
)


# ---------------------------------------------------------
# 3. Load trained XGBoost model
# ---------------------------------------------------------

print("\n[3/5] Loading XGBoost spray-priority model...")

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURES_PATH)

print(
    f"XGBoost model loaded successfully."
)

print(
    f"Expected features: {len(feature_columns)}"
)


# ---------------------------------------------------------
# 4. Create drone-zone feature integration
# ---------------------------------------------------------

print("\n[4/5] Integrating drone intelligence...")


# We use the existing field dataset as the source
# for environmental / IoT / historical features.
#
# Drone-derived NDVI, NDRE and thermal values replace
# the corresponding field-level values.
#
# CNN probabilities are used to strengthen the
# pest/stress indicators.

field_lookup = field_df.copy()


# Select source field records.
#
# The current synthetic drone scene contains four
# segmented zones. We map them to representative
# field zones for this prototype integration.

source_indices = np.linspace(
    0,
    len(field_lookup) - 1,
    len(drone_df),
    dtype=int
)

selected_field = field_lookup.iloc[
    source_indices
].copy()

selected_field = selected_field.reset_index(
    drop=True
)

drone_df = drone_df.reset_index(
    drop=True
)


# ---------------------------------------------------------
# Build integrated feature dataframe
# ---------------------------------------------------------

integrated = selected_field[
    [
        "zone_id",
        "farm_id",
        "crop_type",
        "blue_band",
        "green_band",
        "red_band",
        "red_edge_band",
        "nir_band",
        "ndvi",
        "ndre",
        "thermal_temperature",
        "soil_moisture",
        "humidity",
        "temperature",
        "rainfall",
        "wind_speed",
        "historical_pest_risk",
        "pest_probability",
        "crop_stress",
        "canopy_anomaly"
    ]
].copy()


# ---------------------------------------------------------
# Replace spectral / thermal values with drone measurements
# ---------------------------------------------------------

integrated["ndvi"] = drone_df["mean_ndvi"]

integrated["ndre"] = drone_df["mean_ndre"]

integrated["thermal_temperature"] = (
    drone_df["mean_thermal_temperature"]
)


# ---------------------------------------------------------
# Use CNN intelligence to refine pest/stress indicators
# ---------------------------------------------------------

cnn_pest = drone_df[
    "pest_zone_probability"
].fillna(0)

cnn_stress = drone_df[
    "crop_stress_probability"
].fillna(0)


# Combine existing field intelligence with
# drone CNN evidence.

integrated["pest_probability"] = np.maximum(
    integrated["pest_probability"].values,
    cnn_pest.values
)

integrated["crop_stress"] = np.maximum(
    integrated["crop_stress"].values,
    cnn_stress.values
)


# ---------------------------------------------------------
# Run XGBoost
# ---------------------------------------------------------

X = integrated[
    feature_columns
].copy()

predictions = model.predict(X)


# Convert numeric prediction to labels.

priority_labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}

integrated["spray_priority"] = [
    priority_labels.get(
        int(prediction),
        "Unknown"
    )
    for prediction in predictions
]


# ---------------------------------------------------------
# Add drone intelligence columns
# ---------------------------------------------------------

integrated.insert(
    0,
    "drone_zone_id",
    drone_df["zone_id"]
)

integrated.insert(
    1,
    "source_zone",
    drone_df["source_zone"]
)

integrated["drone_vegetation_status"] = (
    drone_df["vegetation_status"]
)

integrated["cnn_prediction"] = (
    drone_df["cnn_prediction"]
)

integrated["cnn_confidence"] = (
    drone_df["cnn_confidence"]
)

integrated["healthy_probability"] = (
    drone_df["healthy_probability"]
)

integrated["crop_stress_probability"] = (
    drone_df["crop_stress_probability"]
)

integrated["pest_zone_probability"] = (
    drone_df["pest_zone_probability"]
)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

print("\n[5/5] Saving integrated results...")

integrated.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 60)
print("DRONE SPRAY PRIORITY INTEGRATION COMPLETED")
print("=" * 60)

print(
    f"\nZones analysed: {len(integrated)}"
)

print("\nSpray Priority Distribution:")

print(
    integrated[
        "spray_priority"
    ].value_counts()
)

print("\nDrone Zone Results:")

print(
    integrated[
        [
            "drone_zone_id",
            "ndvi",
            "ndre",
            "thermal_temperature",
            "cnn_prediction",
            "cnn_confidence",
            "spray_priority"
        ]
    ].to_string(index=False)
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 60)