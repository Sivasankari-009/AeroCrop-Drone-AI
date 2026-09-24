import os
import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. Load Dataset
# ============================================================

DATA_PATH = "data/aerocrop_field_zones.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("AeroCrop Drone AI - Spray Priority Model")
print("=" * 70)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 2. Features
# ============================================================

features = [
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

target = "priority_category"


X = df[features]
y = df[target]


# ============================================================
# 3. Convert Target Labels
# ============================================================

label_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

y = y.map(label_mapping)


# ============================================================
# 4. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Training records: {len(X_train)}")
print(f"Testing records : {len(X_test)}")


# ============================================================
# 5. XGBoost Model
# ============================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)


# ============================================================
# 6. Train
# ============================================================

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 7. Prediction
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. Evaluation
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Medium", "High"]
    )
)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ============================================================
# 9. Feature Importance
# ============================================================

importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(
    by="importance",
    ascending=False
)

print("\n" + "=" * 70)
print("TOP FEATURES")
print("=" * 70)

print(importance_df.to_string(index=False))


# ============================================================
# 10. Save Model
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "spray_priority_xgboost.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)


# Save feature configuration
feature_path = os.path.join(
    MODEL_DIR,
    "spray_priority_features.pkl"
)

joblib.dump(features, feature_path)

print("Feature configuration saved to:")
print(feature_path)

print("\n" + "=" * 70)
print("SPRAY PRIORITY ENGINE READY 🚁🌾")
print("=" * 70)