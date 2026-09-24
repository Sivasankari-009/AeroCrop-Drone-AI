import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

DATA_PATH = "data/aerocrop_field_zones.csv"
MODEL_PATH = "models/spray_priority_xgboost.pkl"
FEATURE_PATH = "models/spray_priority_features.pkl"

OUTPUT_IMPORTANCE = "models/shap_feature_importance.csv"
OUTPUT_PLOT = "models/shap_summary.png"

print("=" * 70)
print("AeroCrop Drone AI - SHAP Explainability")
print("=" * 70)

print("\nLoading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")

print("\nLoading trained XGBoost model...")
model = joblib.load(MODEL_PATH)
print("XGBoost model loaded successfully.")

if os.path.exists(FEATURE_PATH):
    features = joblib.load(FEATURE_PATH)
    print("Feature configuration loaded.")
else:
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
    print("Using default feature configuration.")

X = df[features]

print(f"\nNumber of features: {len(features)}")
print(f"Total rows: {len(X)}")

sample_size = min(500, len(X))

X_sample = X.sample(
    n=sample_size,
    random_state=42
)

print(f"SHAP sample size: {len(X_sample)}")

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

print("SHAP analysis completed.")

print("\nProcessing SHAP values...")

if isinstance(shap_values, list):

    class_importances = []

    for values in shap_values:
        class_importance = abs(values).mean(axis=0)
        class_importances.append(class_importance)

    mean_abs_shap = (
        sum(class_importances) / len(class_importances)
    )

else:

    shap_array = shap_values

    if shap_array.ndim == 3:

        mean_abs_shap = abs(shap_array).mean(
            axis=(0, 2)
        )

    elif shap_array.ndim == 2:

        mean_abs_shap = abs(shap_array).mean(
            axis=0
        )

    else:

        raise ValueError(
            f"Unexpected SHAP output shape: {shap_array.shape}"
        )

mean_abs_shap = mean_abs_shap.ravel()

if len(mean_abs_shap) != len(features):

    raise ValueError(
        f"Feature count mismatch! "
        f"Features = {len(features)}, "
        f"SHAP values = {len(mean_abs_shap)}"
    )

importance_df = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": mean_abs_shap
})

importance_df = importance_df.sort_values(
    by="mean_abs_shap",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("GLOBAL FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)

importance_df.to_csv(
    OUTPUT_IMPORTANCE,
    index=False
)

print("\nSHAP importance saved to:")
print(OUTPUT_IMPORTANCE)

print("\nGenerating SHAP summary plot...")

if isinstance(shap_values, list):

    shap.summary_plot(
        shap_values[2],
        X_sample,
        show=False
    )

else:

    shap_array = shap_values

    if shap_array.ndim == 3:

        shap.summary_plot(
            shap_array[:, :, 2],
            X_sample,
            show=False
        )

    else:

        shap.summary_plot(
            shap_array,
            X_sample,
            show=False
        )

plt.title(
    "AeroCrop AI - Spray Priority SHAP Analysis"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary saved to:")
print(OUTPUT_PLOT)

print("\n" + "=" * 70)
print("TOP 5 AI DECISION DRIVERS")
print("=" * 70)

for index, row in importance_df.head(5).iterrows():

    print(
        f"{index + 1}. "
        f"{row['feature']} "
        f"-> SHAP: {row['mean_abs_shap']:.6f}"
    )

print("\n" + "=" * 70)
print("EXPLAINABLE AI ENGINE READY")
print("=" * 70)

print("\nGenerated files:")
print(f"1. {OUTPUT_IMPORTANCE}")
print(f"2. {OUTPUT_PLOT}")