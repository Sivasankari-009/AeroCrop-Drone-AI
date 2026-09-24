import os
import joblib
import pandas as pd


# ============================================================
# AeroCrop AI - Edge Inference Engine
# Local / Limited Connectivity Mode
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
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

DRONE_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "segmentation",
    "drone_spray_priority_analysis.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "data",
    "segmentation",
    "edge_inference_results.csv"
)


print("=" * 65)
print("AeroCrop AI - EDGE INFERENCE ENGINE")
print("=" * 65)

print("\nExecution Mode: LOCAL EDGE")
print("Network Dependency: NONE")
print("Backend Dependency: NONE")


# ------------------------------------------------------------
# 1. Load local model
# ------------------------------------------------------------

print("\n[1/4] Loading local XGBoost model...")

model = joblib.load(
    MODEL_PATH
)

feature_columns = joblib.load(
    FEATURES_PATH
)

print("XGBoost model loaded.")
print(
    f"Model features: {len(feature_columns)}"
)


# ------------------------------------------------------------
# 2. Load local drone-zone intelligence
# ------------------------------------------------------------

print("\n[2/4] Loading local drone-zone data...")

df = pd.read_csv(
    DRONE_DATA_PATH
)

print(
    f"Drone zones loaded: {len(df)}"
)


# ------------------------------------------------------------
# 3. Local inference
# ------------------------------------------------------------

print("\n[3/4] Running local edge inference...")


X = df[
    feature_columns
].copy()


predictions = model.predict(
    X
)


priority_labels = {
    0: "Low",
    1: "Medium",
    2: "High"
}


df["edge_spray_priority"] = [
    priority_labels.get(
        int(prediction),
        "Unknown"
    )
    for prediction in predictions
]


df["inference_mode"] = "EDGE_LOCAL"

df["network_required"] = False

df["backend_required"] = False


# ------------------------------------------------------------
# 4. Save edge results
# ------------------------------------------------------------

print("\n[4/4] Saving edge inference results...")

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print("\n" + "=" * 65)
print("EDGE INFERENCE COMPLETED")
print("=" * 65)

print(
    "\nConnectivity Status:"
)

print(
    "Network: NOT REQUIRED"
)

print(
    "Backend API: NOT REQUIRED"
)

print(
    "Inference Location: LOCAL EDGE NODE"
)

print(
    f"\nZones processed: {len(df)}"
)

print(
    "\nEdge Spray Priority:"
)

print(
    df[
        "edge_spray_priority"
    ].value_counts()
)

print(
    "\nZone Results:"
)

display_columns = [
    "drone_zone_id",
    "edge_spray_priority",
    "inference_mode",
    "network_required"
]

available_columns = [
    column
    for column in display_columns
    if column in df.columns
]

print(
    df[
        available_columns
    ].to_string(index=False)
)

print(
    f"\nSaved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 65)