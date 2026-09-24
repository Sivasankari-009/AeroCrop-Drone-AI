import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

DATA_PATH = "data/aerocrop_field_zones.csv"

MODEL_PATH = "models/canopy_autoencoder.keras"
SCALER_PATH = "models/canopy_scaler.pkl"
THRESHOLD_PATH = "models/anomaly_threshold.pkl"

RANDOM_STATE = 42

print("=" * 70)
print("AeroCrop Drone AI - Canopy Anomaly Detection")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape}")

# ---------------------------------------------------------
# 2. Select Canopy Reflectance Features
# ---------------------------------------------------------

features = [
    "blue_band",
    "green_band",
    "red_band",
    "red_edge_band",
    "nir_band",
    "ndvi",
    "ndre",
    "thermal_temperature"
]

X = df[features].copy()

print("\nCanopy features selected:")
for feature in features:
    print(f"- {feature}")

print(f"\nFeature count: {len(features)}")
print(f"Total samples: {len(X)}")

# ---------------------------------------------------------
# 3. Handle Missing Values
# ---------------------------------------------------------

print("\nChecking missing values...")

missing_values = X.isnull().sum().sum()

print(f"Total missing values: {missing_values}")

if missing_values > 0:
    X = X.fillna(X.median())
    print("Missing values filled using feature medians.")
else:
    print("No missing values found.")

# ---------------------------------------------------------
# 4. Standardize Features
# ---------------------------------------------------------

print("\nScaling canopy features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")

# ---------------------------------------------------------
# 5. Train / Validation Split
# ---------------------------------------------------------

X_train, X_test = train_test_split(
    X_scaled,
    test_size=0.20,
    random_state=RANDOM_STATE
)

print("\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# ---------------------------------------------------------
# 6. Build Autoencoder
# ---------------------------------------------------------

print("\nBuilding Autoencoder model...")

input_dim = X_train.shape[1]

model = Sequential([
    Input(shape=(input_dim,)),

    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(4, activation="relu"),

    Dense(8, activation="relu"),
    Dense(16, activation="relu"),
    Dense(input_dim, activation="linear")
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# ---------------------------------------------------------
# 7. Train Autoencoder
# ---------------------------------------------------------

print("\nTraining Autoencoder...")

history = model.fit(
    X_train,
    X_train,
    epochs=50,
    batch_size=32,
    validation_split=0.20,
    shuffle=True,
    verbose=1
)

print("\nAutoencoder training completed.")

# ---------------------------------------------------------
# 8. Calculate Reconstruction Error
# ---------------------------------------------------------

print("\nCalculating reconstruction errors...")

X_test_pred = model.predict(
    X_test,
    verbose=0
)

reconstruction_errors = np.mean(
    np.square(X_test - X_test_pred),
    axis=1
)

print(
    f"Mean reconstruction error: "
    f"{reconstruction_errors.mean():.6f}"
)

print(
    f"Maximum reconstruction error: "
    f"{reconstruction_errors.max():.6f}"
)

# ---------------------------------------------------------
# 9. Calculate Anomaly Threshold
# ---------------------------------------------------------

threshold = np.percentile(
    reconstruction_errors,
    95
)

print("\nAnomaly threshold calculated:")
print(f"Threshold: {threshold:.6f}")

# ---------------------------------------------------------
# 10. Detect Anomalies
# ---------------------------------------------------------

anomaly_flags = (
    reconstruction_errors > threshold
).astype(int)

anomaly_count = anomaly_flags.sum()

normal_count = len(anomaly_flags) - anomaly_count

print("\nAnomaly Detection Results:")
print(f"Normal samples: {normal_count}")
print(f"Anomalous samples: {anomaly_count}")

print(
    f"Anomaly percentage: "
    f"{(anomaly_count / len(anomaly_flags)) * 100:.2f}%"
)

# ---------------------------------------------------------
# 11. Save Model
# ---------------------------------------------------------

print("\nSaving Autoencoder model...")

os.makedirs("models", exist_ok=True)

model.save(MODEL_PATH)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    threshold,
    THRESHOLD_PATH
)

print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
print(f"Threshold saved to: {THRESHOLD_PATH}")

# ---------------------------------------------------------
# 12. Generate Full Dataset Anomaly Scores
# ---------------------------------------------------------

print("\nGenerating anomaly scores for all field zones...")

X_all_pred = model.predict(
    X_scaled,
    verbose=0
)

all_reconstruction_errors = np.mean(
    np.square(X_scaled - X_all_pred),
    axis=1
)

df["anomaly_score"] = all_reconstruction_errors

df["anomaly_detected"] = (
    df["anomaly_score"] > threshold
).astype(int)

df["anomaly_category"] = np.where(
    df["anomaly_detected"] == 1,
    "Anomalous",
    "Normal"
)

# ---------------------------------------------------------
# 13. Save Enhanced Dataset
# ---------------------------------------------------------

OUTPUT_DATASET = "data/aerocrop_anomaly_results.csv"

df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print(f"\nEnhanced dataset saved to:")
print(OUTPUT_DATASET)

# ---------------------------------------------------------
# 14. Display Top Anomalous Zones
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 10 ANOMALOUS FIELD ZONES")
print("=" * 70)

top_anomalies = df.sort_values(
    by="anomaly_score",
    ascending=False
).head(10)

print(
    top_anomalies[
        [
            "zone_id",
            "farm_id",
            "crop_type",
            "ndvi",
            "ndre",
            "thermal_temperature",
            "anomaly_score",
            "anomaly_category"
        ]
    ].to_string(index=False)
)

# ---------------------------------------------------------
# 15. Final Status
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("AUTOENCODER ANOMALY ENGINE READY")
print("=" * 70)

print("\nGenerated files:")
print(f"1. {MODEL_PATH}")
print(f"2. {SCALER_PATH}")
print(f"3. {THRESHOLD_PATH}")
print(f"4. {OUTPUT_DATASET}")

print("\nPhase 4 completed successfully.")