import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

DATA_PATH = "data/aerocrop_field_zones.csv"

MODEL_PATH = "models/irrigation_lstm.keras"
SCALER_PATH = "models/irrigation_scaler.pkl"
OUTPUT_PATH = "data/aerocrop_irrigation_forecast.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

print("=" * 70)
print("AeroCrop Drone AI - LSTM Irrigation Forecasting")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------

print("\nLoading field-zone dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded: {df.shape}")

# ---------------------------------------------------------
# 2. Generate Synthetic Time-Series
# ---------------------------------------------------------

print("\nGenerating 30-day irrigation history...")

base = df[
    [
        "zone_id",
        "soil_moisture",
        "temperature",
        "humidity",
        "rainfall"
    ]
].copy()

records = []

for _, row in base.iterrows():

    zone_id = row["zone_id"]

    base_moisture = row["soil_moisture"]
    base_temperature = row["temperature"]
    base_humidity = row["humidity"]
    base_rainfall = row["rainfall"]

    for day in range(30):

        seasonal_effect = np.sin(day / 4.0)

        moisture = (
            base_moisture
            + seasonal_effect * 3
            + np.random.normal(0, 1.5)
        )

        temperature = (
            base_temperature
            + np.random.normal(0, 1.2)
        )

        humidity = (
            base_humidity
            + np.random.normal(0, 3)
        )

        rainfall = max(
            0,
            base_rainfall
            + np.random.normal(0, 2)
        )

        # Simulated irrigation demand
        irrigation_demand = (
            0.55 * (100 - moisture)
            + 0.25 * temperature
            - 0.20 * humidity
            - 0.50 * rainfall
        )

        irrigation_demand = max(
            0,
            irrigation_demand
        )

        records.append(
            [
                zone_id,
                day,
                moisture,
                temperature,
                humidity,
                rainfall,
                irrigation_demand
            ]
        )

timeseries_df = pd.DataFrame(
    records,
    columns=[
        "zone_id",
        "day",
        "soil_moisture",
        "temperature",
        "humidity",
        "rainfall",
        "irrigation_demand"
    ]
)

print(
    f"Time-series records generated: "
    f"{len(timeseries_df)}"
)

# ---------------------------------------------------------
# 3. Prepare Features
# ---------------------------------------------------------

features = [
    "soil_moisture",
    "temperature",
    "humidity",
    "rainfall"
]

target = "irrigation_demand"

X_raw = timeseries_df[features].values
y_raw = timeseries_df[target].values.reshape(-1, 1)

print("\nInput features:")
for feature in features:
    print(f"- {feature}")

# ---------------------------------------------------------
# 4. Scale Data
# ---------------------------------------------------------

print("\nScaling time-series data...")

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X_raw)

# ---------------------------------------------------------
# 5. Create LSTM Sequences
# ---------------------------------------------------------

SEQUENCE_LENGTH = 7

X_sequences = []
y_sequences = []

for i in range(SEQUENCE_LENGTH, len(X_scaled)):

    X_sequences.append(
        X_scaled[i - SEQUENCE_LENGTH:i]
    )

    y_sequences.append(
        y_raw[i]
    )

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

print("\nLSTM sequence preparation completed.")

print(f"Sequence length: {SEQUENCE_LENGTH} days")
print(f"X shape: {X_sequences.shape}")
print(f"y shape: {y_sequences.shape}")

# ---------------------------------------------------------
# 6. Train / Test Split
# ---------------------------------------------------------

split_index = int(
    len(X_sequences) * 0.80
)

X_train = X_sequences[:split_index]
X_test = X_sequences[split_index:]

y_train = y_sequences[:split_index]
y_test = y_sequences[split_index:]

print("\nDataset split:")
print(f"Training sequences: {len(X_train)}")
print(f"Testing sequences: {len(X_test)}")

# ---------------------------------------------------------
# 7. Build LSTM Model
# ---------------------------------------------------------

print("\nBuilding LSTM model...")

model = Sequential([
    Input(
        shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    ),

    LSTM(
        64,
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        32
    ),

    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),

    Dense(
        1,
        activation="linear"
    )
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# ---------------------------------------------------------
# 8. Train Model
# ---------------------------------------------------------

print("\nTraining LSTM irrigation forecasting model...")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.10,
    shuffle=False,
    verbose=1
)

print("\nLSTM training completed.")

# ---------------------------------------------------------
# 9. Evaluate Model
# ---------------------------------------------------------

print("\nEvaluating LSTM model...")

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print(f"Test Loss: {loss:.6f}")
print(f"Test MAE: {mae:.6f}")

# ---------------------------------------------------------
# 10. Generate Predictions
# ---------------------------------------------------------

print("\nGenerating irrigation forecasts...")

predictions = model.predict(
    X_test,
    verbose=0
).flatten()

actual = y_test.flatten()

forecast_df = pd.DataFrame({
    "actual_irrigation_demand": actual,
    "predicted_irrigation_demand": predictions
})

# ---------------------------------------------------------
# 11. Irrigation Recommendation
# ---------------------------------------------------------

forecast_df["irrigation_recommendation"] = np.select(
    [
        forecast_df["predicted_irrigation_demand"] >= 25,
        forecast_df["predicted_irrigation_demand"] >= 10
    ],
    [
        "HIGH",
        "MEDIUM"
    ],
    default="LOW"
)

# ---------------------------------------------------------
# 12. Save Model and Scaler
# ---------------------------------------------------------

print("\nSaving LSTM model...")

os.makedirs("models", exist_ok=True)

model.save(
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

print(f"Model saved to: {MODEL_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")

# ---------------------------------------------------------
# 13. Save Forecast Results
# ---------------------------------------------------------

forecast_df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Forecast results saved to: {OUTPUT_PATH}")

# ---------------------------------------------------------
# 14. Display Forecast Preview
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGATION FORECAST PREVIEW")
print("=" * 70)

print(
    forecast_df.head(10).to_string(
        index=False
    )
)

# ---------------------------------------------------------
# 15. Forecast Statistics
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("IRRIGATION RECOMMENDATION SUMMARY")
print("=" * 70)

print(
    forecast_df[
        "irrigation_recommendation"
    ].value_counts()
)

# ---------------------------------------------------------
# 16. Final Status
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("LSTM IRRIGATION FORECASTING ENGINE READY")
print("=" * 70)

print("\nGenerated files:")
print(f"1. {MODEL_PATH}")
print(f"2. {SCALER_PATH}")
print(f"3. {OUTPUT_PATH}")

print("\nPhase 5 completed successfully.")