import os
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model


# ============================================================
# AeroCrop Drone AI - FastAPI Backend
# ============================================================

app = FastAPI(
    title="AeroCrop Drone AI",
    description="Autonomous Drone-Based Precision Agriculture AI Backend",
    version="1.0.0"
)


# ============================================================
# Model Paths
# ============================================================

SPRAY_MODEL_PATH = "models/spray_priority_xgboost.pkl"
SPRAY_FEATURE_PATH = "models/spray_priority_features.pkl"

ANOMALY_MODEL_PATH = "models/canopy_autoencoder.keras"
ANOMALY_SCALER_PATH = "models/canopy_scaler.pkl"
ANOMALY_THRESHOLD_PATH = "models/anomaly_threshold.pkl"

IRRIGATION_MODEL_PATH = "models/irrigation_lstm.keras"
IRRIGATION_SCALER_PATH = "models/irrigation_scaler.pkl"

CNN_MODEL_PATH = "models/crop_stress_cnn.keras"
CNN_CLASS_PATH = "models/crop_stress_classes.json"


# ============================================================
# Load Models
# ============================================================

print("=" * 70)
print("AeroCrop Drone AI - Loading AI Models")
print("=" * 70)


# XGBoost
spray_model = joblib.load(
    SPRAY_MODEL_PATH
)

spray_features = joblib.load(
    SPRAY_FEATURE_PATH
)


# Autoencoder
anomaly_model = load_model(
    ANOMALY_MODEL_PATH
)

anomaly_scaler = joblib.load(
    ANOMALY_SCALER_PATH
)

anomaly_threshold = joblib.load(
    ANOMALY_THRESHOLD_PATH
)


# LSTM
irrigation_model = load_model(
    IRRIGATION_MODEL_PATH
)

irrigation_scaler = joblib.load(
    IRRIGATION_SCALER_PATH
)


# CNN
cnn_model = load_model(
    CNN_MODEL_PATH
)

with open(
    CNN_CLASS_PATH,
    "r"
) as file:
    cnn_classes = json.load(file)


print("\nAll AI models loaded successfully.")

print(f"Spray features: {len(spray_features)}")
print(f"Anomaly threshold: {anomaly_threshold}")
print(f"CNN classes: {cnn_classes}")


# ============================================================
# Request Models
# ============================================================

class SprayRequest(BaseModel):

    blue_band: float
    green_band: float
    red_band: float
    red_edge_band: float
    nir_band: float
    ndvi: float
    ndre: float
    thermal_temperature: float
    soil_moisture: float
    humidity: float
    temperature: float
    rainfall: float
    wind_speed: float
    historical_pest_risk: float
    pest_probability: float
    crop_stress: float
    canopy_anomaly: float


class AnomalyRequest(BaseModel):

    blue_band: float
    green_band: float
    red_band: float
    red_edge_band: float
    nir_band: float
    ndvi: float
    ndre: float
    thermal_temperature: float


class IrrigationRequest(BaseModel):

    sequence: list


class DronePredictionRequest(BaseModel):

    image_features: list


class ApprovalRequest(BaseModel):

    zone_id: str
    intervention_type: str
    approved_by: str
    resource_volume: float = 0.0
    estimated_cost: float = 0.0
    currency: str = "INR"
    reason: str = ""


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():

    return {
        "project": "AeroCrop Drone AI",
        "status": "online",
        "version": "1.0.0",
        "message": "AI backend is running successfully"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models": {
            "spray_priority": "loaded",
            "canopy_anomaly": "loaded",
            "irrigation_forecasting": "loaded",
            "drone_cnn": "loaded"
        }
    }


# ============================================================
# Spray Priority Prediction
# ============================================================

@app.post("/predict/spray-priority")
def predict_spray_priority(
    request: SprayRequest
):

    try:

        input_data = pd.DataFrame(
            [[
                request.blue_band,
                request.green_band,
                request.red_band,
                request.red_edge_band,
                request.nir_band,
                request.ndvi,
                request.ndre,
                request.thermal_temperature,
                request.soil_moisture,
                request.humidity,
                request.temperature,
                request.rainfall,
                request.wind_speed,
                request.historical_pest_risk,
                request.pest_probability,
                request.crop_stress,
                request.canopy_anomaly
            ]],
            columns=spray_features
        )

        prediction = spray_model.predict(
            input_data
        )[0]

        labels = {
            0: "Low",
            1: "Medium",
            2: "High"
        }

        priority = labels.get(
            int(prediction),
            "Unknown"
        )

        return {
            "success": True,
            "spraying_priority": priority,
            "priority_code": int(prediction)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Canopy Anomaly Detection
# ============================================================

@app.post("/detect/anomaly")
def detect_anomaly(
    request: AnomalyRequest
):

    try:

        values = np.array([[
            request.blue_band,
            request.green_band,
            request.red_band,
            request.red_edge_band,
            request.nir_band,
            request.ndvi,
            request.ndre,
            request.thermal_temperature
        ]])

        scaled_values = anomaly_scaler.transform(
            values
        )

        reconstructed = anomaly_model.predict(
            scaled_values,
            verbose=0
        )

        error = np.mean(
            np.square(
                scaled_values - reconstructed
            ),
            axis=1
        )[0]

        detected = (
            error > anomaly_threshold
        )

        return {
            "success": True,
            "anomaly_score": float(error),
            "threshold": float(anomaly_threshold),
            "anomaly_detected": bool(detected),
            "status": (
                "Anomalous"
                if detected
                else "Normal"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Irrigation Forecasting
# ============================================================

@app.post("/predict/irrigation")
def predict_irrigation(
    request: IrrigationRequest
):

    try:

        sequence = np.array(
            request.sequence,
            dtype=float
        )

        if sequence.shape != (7, 4):

            raise HTTPException(
                status_code=400,
                detail=(
                    "Sequence must contain "
                    "7 days with 4 features: "
                    "soil_moisture, temperature, "
                    "humidity, rainfall"
                )
            )

        scaled = irrigation_scaler.transform(
            sequence
        )

        model_input = scaled.reshape(
            1,
            7,
            4
        )

        prediction = irrigation_model.predict(
            model_input,
            verbose=0
        )[0][0]

        if prediction >= 25:
            recommendation = "HIGH"
        elif prediction >= 10:
            recommendation = "MEDIUM"
        else:
            recommendation = "LOW"

        return {
            "success": True,
            "predicted_irrigation_demand": float(
                prediction
            ),
            "recommendation": recommendation
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CNN Drone Image Endpoint
# ============================================================

@app.post("/analyze/drone-image")
def analyze_drone_image(
    request: DronePredictionRequest
):

    try:

        features = np.array(
            request.image_features,
            dtype=float
        )

        if features.size != 128 * 128 * 3:

            raise HTTPException(
                status_code=400,
                detail=(
                    "image_features must contain "
                    "128 x 128 x 3 values"
                )
            )

        image = features.reshape(
            1,
            128,
            128,
            3
        )

        image = image / 255.0

        probabilities = cnn_model.predict(
            image,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_class = cnn_classes[
            str(predicted_index)
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        return {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence,
            "probabilities": {
                cnn_classes[str(i)]: float(
                    probabilities[i]
                )
                for i in range(
                    len(probabilities)
                )
            }
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Human Approval - Simulated Intervention
# ============================================================

@app.post("/intervention/approve")
def approve_intervention(
    request: ApprovalRequest
):

    allowed_types = [
        "spraying",
        "irrigation"
    ]

    if request.intervention_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail=(
                "Intervention type must be "
                "spraying or irrigation"
            )
        )

    # --------------------------------------------------------
    # Create audit record
    # --------------------------------------------------------

    audit_record = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "zone_id": request.zone_id,
        "intervention_type": request.intervention_type,
        "approved_by": request.approved_by,
        "approval_status": "APPROVED",
        "execution_mode": "SIMULATED",
        "resource_volume": request.resource_volume,
        "estimated_cost": request.estimated_cost,
        "currency": request.currency,
        "compliance_status": "COMPLIANT",
        "human_approval_required": True,
        "reason": request.reason
    }

    # --------------------------------------------------------
    # Ensure logs directory exists
    # --------------------------------------------------------

    os.makedirs(
        "logs",
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save audit record
    # --------------------------------------------------------

    if os.path.exists(AUDIT_LOG_PATH):

        audit_df = pd.read_csv(
            AUDIT_LOG_PATH
        )

        new_record_df = pd.DataFrame(
            [audit_record]
        )

        audit_df = pd.concat(
            [
                audit_df,
                new_record_df
            ],
            ignore_index=True
        )

    else:

        audit_df = pd.DataFrame(
            [audit_record]
        )

    audit_df.to_csv(
        AUDIT_LOG_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Return approval response
    # --------------------------------------------------------

    return {
        "success": True,
        "zone_id": request.zone_id,
        "intervention_type": request.intervention_type,
        "approved_by": request.approved_by,
        "resource_volume": request.resource_volume,
        "estimated_cost": request.estimated_cost,
        "currency": request.currency,
        "compliance_status": "COMPLIANT",
        "human_approval_required": True,
        "approval_status": "APPROVED",
        "execution_mode": "SIMULATED",
        "message": (
            "Intervention approved for "
            "simulation only. No physical "
            "actuator was triggered."
        ),
        "reason": request.reason,
        "audit_logged": True,
        "audit_log_path": AUDIT_LOG_PATH
    }

# ============================================================
# Field Zone API
# ============================================================

@app.get("/field-zones")
def get_field_zones():

    try:

        df = pd.read_csv(
            "data/aerocrop_field_zones.csv"
        )

        columns = [
            "zone_id",
            "farm_id",
            "crop_type",
            "latitude",
            "longitude",
            "ndvi",
            "ndre",
            "thermal_temperature",
            "soil_moisture",
            "pest_probability",
            "crop_stress",
            "canopy_anomaly",
            "stress_category",
            "pest_zone",
            "irrigation_required",
            "spraying_priority",
            "priority_category"
        ]

        return {
            "success": True,
            "total_zones": len(df),
            "zones": df[
                columns
            ].to_dict(
                orient="records"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Audit Log
# ============================================================

AUDIT_LOG_PATH = "logs/intervention_audit.csv"


@app.get("/audit-log")
def get_audit_log():

    if not os.path.exists(
        AUDIT_LOG_PATH
    ):

        return {
            "success": True,
            "records": []
        }

    df = pd.read_csv(
        AUDIT_LOG_PATH
    )

    # Convert empty CSV values (NaN)
    # into JSON-safe empty strings
    df = df.fillna("")

    return {
        "success": True,
        "records": df.to_dict(
            orient="records"
        )
    }


# ============================================================
# Run Information
# ============================================================

@app.get("/system-info")
def system_info():

    return {
        "project": "AeroCrop Drone AI",
        "domain": "Precision Agriculture",
        "ai_engines": [
            "XGBoost Spray Priority",
            "SHAP Explainable AI",
            "Autoencoder Canopy Anomaly Detection",
            "LSTM Irrigation Forecasting",
            "CNN Drone Image Intelligence"
        ],
        "intervention_control": (
            "Human approval required"
        ),
        "actuator_mode": "Simulation only"
    }