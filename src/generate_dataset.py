import os
import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Output directory
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of field-zone records
N = 5000

# Zone IDs
zone_id = [f"ZONE-{i:04d}" for i in range(1, N + 1)]

# Farm and crop information
farm_id = np.random.choice(
    ["FARM-001", "FARM-002", "FARM-003", "FARM-004", "FARM-005"],
    N
)

crop_type = np.random.choice(
    ["Rice", "Wheat", "Cotton", "Maize", "Tomato"],
    N
)

# Geographic coordinates
latitude = np.random.uniform(10.80, 11.10, N)
longitude = np.random.uniform(76.80, 77.10, N)

# Multispectral features
blue_band = np.random.uniform(0.10, 0.70, N)
green_band = np.random.uniform(0.15, 0.80, N)
red_band = np.random.uniform(0.10, 0.75, N)
red_edge_band = np.random.uniform(0.20, 0.90, N)
nir_band = np.random.uniform(0.30, 0.95, N)

# Vegetation indices
ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-6)
ndre = (nir_band - red_edge_band) / (nir_band + red_edge_band + 1e-6)

# Environmental / sensor data
thermal_temperature = np.random.uniform(22, 42, N)
soil_moisture = np.random.uniform(15, 80, N)
humidity = np.random.uniform(35, 95, N)
temperature = np.random.uniform(20, 42, N)
rainfall = np.random.uniform(0, 30, N)
wind_speed = np.random.uniform(1, 25, N)

# Historical pest risk
historical_pest_risk = np.random.uniform(0, 1, N)

# Pest probability
pest_probability = (
    0.35 * historical_pest_risk
    + 0.25 * (thermal_temperature - 22) / 20
    + 0.20 * (1 - ndvi)
    + 0.20 * np.random.uniform(0, 1, N)
)

pest_probability = np.clip(pest_probability, 0, 1)

# Crop stress score
crop_stress = (
    0.40 * (1 - ndvi)
    + 0.25 * (thermal_temperature - 22) / 20
    + 0.20 * (1 - soil_moisture / 100)
    + 0.15 * np.random.uniform(0, 1, N)
)

crop_stress = np.clip(crop_stress, 0, 1)

# Canopy anomaly score
canopy_anomaly = (
    0.50 * (1 - ndre)
    + 0.30 * crop_stress
    + 0.20 * np.random.uniform(0, 1, N)
)

canopy_anomaly = np.clip(canopy_anomaly, 0, 1)

# Stress category
stress_category = np.select(
    [
        crop_stress < 0.35,
        crop_stress < 0.65
    ],
    [
        "Healthy",
        "Moderate Stress"
    ],
    default="Severe Stress"
)

# Pest zone
pest_zone = np.where(
    pest_probability >= 0.65,
    "High Risk",
    np.where(
        pest_probability >= 0.35,
        "Moderate Risk",
        "Low Risk"
    )
)

# Irrigation requirement
irrigation_required = np.where(
    soil_moisture < 35,
    1,
    0
)

# Spraying priority
spraying_priority = (
    0.45 * pest_probability
    + 0.30 * crop_stress
    + 0.15 * canopy_anomaly
    + 0.10 * historical_pest_risk
) * 100

spraying_priority = np.clip(spraying_priority, 0, 100)

# Priority category
priority_category = np.select(
    [
        spraying_priority < 35,
        spraying_priority < 65
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)

# Build dataframe
df = pd.DataFrame({
    "zone_id": zone_id,
    "farm_id": farm_id,
    "crop_type": crop_type,
    "latitude": latitude,
    "longitude": longitude,

    "blue_band": blue_band,
    "green_band": green_band,
    "red_band": red_band,
    "red_edge_band": red_edge_band,
    "nir_band": nir_band,

    "ndvi": ndvi,
    "ndre": ndre,

    "thermal_temperature": thermal_temperature,
    "soil_moisture": soil_moisture,
    "humidity": humidity,
    "temperature": temperature,
    "rainfall": rainfall,
    "wind_speed": wind_speed,

    "historical_pest_risk": historical_pest_risk,
    "pest_probability": pest_probability,
    "crop_stress": crop_stress,
    "canopy_anomaly": canopy_anomaly,

    "stress_category": stress_category,
    "pest_zone": pest_zone,
    "irrigation_required": irrigation_required,

    "spraying_priority": spraying_priority,
    "priority_category": priority_category
})

# Save dataset
output_file = os.path.join(OUTPUT_DIR, "aerocrop_field_zones.csv")
df.to_csv(output_file, index=False)

print("=" * 60)
print("AeroCrop Drone AI - Dataset Generator")
print("=" * 60)
print(f"Total records       : {len(df)}")
print(f"Total farms         : {df['farm_id'].nunique()}")
print(f"Crop types          : {df['crop_type'].nunique()}")
print(f"High pest zones     : {(df['pest_zone'] == 'High Risk').sum()}")
print(f"Severe stress zones : {(df['stress_category'] == 'Severe Stress').sum()}")
print(f"Irrigation zones    : {df['irrigation_required'].sum()}")
print(f"High spray priority : {(df['priority_category'] == 'High').sum()}")
print(f"Dataset saved to    : {output_file}")
print("=" * 60)