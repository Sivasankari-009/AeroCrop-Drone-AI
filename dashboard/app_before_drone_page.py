import os
import json
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from PIL import Image
from streamlit_folium import st_folium
import folium


# ============================================================
# AeroCrop Drone AI - Main Dashboard
# ============================================================

st.set_page_config(
    page_title="AeroCrop Drone AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# Configuration
# ============================================================

API_URL = "http://127.0.0.1:8000"

FIELD_DATA = "data/aerocrop_field_zones.csv"
ANOMALY_DATA = "data/aerocrop_anomaly_results.csv"
IRRIGATION_DATA = "data/aerocrop_irrigation_forecast.csv"
SHAP_DATA = "models/shap_feature_importance.csv"


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    '''
    <style>
    :root {
        --ac-bg:#070b0a; --ac-panel:#101a16; --ac-border:rgba(112,255,185,.13);
        --ac-green:#49f2a6; --ac-cyan:#55d9ff; --ac-amber:#ffc857; --ac-red:#ff5c70;
        --ac-text:#f4f8f6; --ac-muted:#8b9b94;
    }
    .stApp {
        background:radial-gradient(circle at 82% 8%,rgba(73,242,166,.08),transparent 24%),
        radial-gradient(circle at 18% 18%,rgba(85,217,255,.055),transparent 22%),
        linear-gradient(135deg,#070b0a 0%,#0a0f0d 52%,#070b0a 100%);
        color:var(--ac-text);
    }
    .main .block-container {max-width:1540px;padding-top:2.2rem;padding-bottom:4rem;}
    section[data-testid="stSidebar"] {
        background:radial-gradient(circle at 30% 12%,rgba(73,242,166,.08),transparent 26%),
        linear-gradient(180deg,#0c1210 0%,#080d0b 100%);
        border-right:1px solid rgba(73,242,166,.10);
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius:11px;padding:7px 9px;transition:all .18s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background:rgba(73,242,166,.08);box-shadow:0 0 18px rgba(73,242,166,.07);
    }
    .ac-header {display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin-bottom:22px;}
    .ac-eyebrow {color:var(--ac-green);font-size:.72rem;font-weight:800;letter-spacing:.18em;text-transform:uppercase;margin-bottom:7px;}
    .main-title {font-size:clamp(2.1rem,4vw,3.15rem);line-height:1.05;font-weight:850;letter-spacing:-.045em;margin-bottom:5px;text-shadow:0 0 34px rgba(73,242,166,.10);}
    .subtitle {font-size:.98rem;color:var(--ac-muted);margin-bottom:24px;}
    .ac-system-pill {white-space:nowrap;padding:10px 15px;border:1px solid rgba(73,242,166,.22);border-radius:999px;background:rgba(73,242,166,.055);color:#a9ffd2;font-size:.78rem;font-weight:750;box-shadow:0 0 24px rgba(73,242,166,.09);}
    .ac-dot {display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--ac-green);box-shadow:0 0 7px var(--ac-green),0 0 17px rgba(73,242,166,.75);animation:ac-pulse 1.8s infinite;}
    @keyframes ac-pulse {0%,100%{transform:scale(1);opacity:.72}50%{transform:scale(1.28);opacity:1}}
    [data-testid="stMetric"] {
        background:linear-gradient(145deg,rgba(18,29,25,.92),rgba(9,15,13,.92));
        border:1px solid var(--ac-border);border-radius:16px;padding:15px 17px;min-height:112px;
        transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {transform:translateY(-3px);border-color:rgba(73,242,166,.30);box-shadow:0 0 28px rgba(73,242,166,.10);}
    [data-testid="stMetricLabel"] {color:#879a92!important;font-size:.74rem!important;font-weight:700!important;text-transform:uppercase;letter-spacing:.08em;}
    [data-testid="stMetricValue"] {color:#f4fff9!important;font-size:1.85rem!important;font-weight:820!important;}
    .stButton > button {
        border-radius:12px!important;border:1px solid rgba(73,242,166,.28)!important;
        background:linear-gradient(135deg,#123d2d,#0d261d)!important;color:#eafff4!important;
        font-weight:800!important;min-height:44px;box-shadow:0 0 18px rgba(73,242,166,.07);
        transition:all .18s ease!important;
    }
    .stButton > button:hover {transform:translateY(-2px);border-color:rgba(73,242,166,.58)!important;box-shadow:0 0 28px rgba(73,242,166,.17)!important;}
    div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,textarea {
        background:rgba(19,25,23,.90)!important;border-color:rgba(73,242,166,.12)!important;border-radius:11px!important;
    }
    div[data-baseweb="select"] > div:focus-within,div[data-baseweb="input"] > div:focus-within,textarea:focus {
        border-color:rgba(73,242,166,.42)!important;box-shadow:0 0 18px rgba(73,242,166,.08)!important;
    }
    div[data-testid="stAlert"] {border-radius:13px;border:1px solid rgba(73,242,166,.12);box-shadow:0 0 22px rgba(73,242,166,.04);}
    [data-testid="stDataFrame"] {border:1px solid rgba(73,242,166,.10);border-radius:14px;overflow:hidden;box-shadow:0 12px 35px rgba(0,0,0,.16);}
    [data-testid="stFileUploaderDropzone"] {border:1px dashed rgba(73,242,166,.24);border-radius:15px;background:rgba(73,242,166,.025);}
    hr {border-color:rgba(255,255,255,.07)!important;}
    .ac-card {
        background:linear-gradient(145deg,rgba(18,30,25,.86),rgba(8,14,12,.92));
        border:1px solid rgba(73,242,166,.12);border-radius:18px;padding:20px;
        box-shadow:0 15px 45px rgba(0,0,0,.18);transition:all .18s ease;
    }
    .ac-card:hover {border-color:rgba(73,242,166,.22);box-shadow:0 0 30px rgba(73,242,166,.07),0 15px 45px rgba(0,0,0,.18);}
    .ac-card-title {font-size:.76rem;color:#8ea39a;font-weight:800;letter-spacing:.10em;text-transform:uppercase;margin-bottom:8px;}
    .ac-card-note {color:#7f9189;font-size:.78rem;margin-top:6px;}
    .ac-ai-status {display:flex;align-items:center;justify-content:space-between;padding:13px 15px;margin:7px 0;border-radius:13px;background:rgba(73,242,166,.045);border:1px solid rgba(73,242,166,.10);}
    .ac-ai-name {font-weight:750;color:#dcebe5;}
    .ac-ready {color:#7cffba;font-size:.72rem;font-weight:800;letter-spacing:.08em;}
    .ac-section {font-size:1.12rem;font-weight:820;letter-spacing:-.02em;margin:12px 0 13px;}
    .ac-chat {background:linear-gradient(145deg,rgba(17,31,25,.92),rgba(9,17,14,.92));border:1px solid rgba(73,242,166,.15);border-radius:20px;padding:22px;box-shadow:0 0 35px rgba(73,242,166,.05);margin-bottom:16px;}
    .ac-footer {margin-top:35px;padding-top:15px;border-top:1px solid rgba(255,255,255,.06);color:#62736c;font-size:.72rem;text-align:center;letter-spacing:.04em;}
    </style>
    ''',
    unsafe_allow_html=True
)

# ============================================================
# Load Data
# ============================================================

@st.cache_data
def load_field_data():

    return pd.read_csv(
        FIELD_DATA
    )


@st.cache_data
def load_anomaly_data():

    if os.path.exists(ANOMALY_DATA):
        return pd.read_csv(
            ANOMALY_DATA
        )

    return None


@st.cache_data
def load_irrigation_data():

    if os.path.exists(IRRIGATION_DATA):
        return pd.read_csv(
            IRRIGATION_DATA
        )

    return None


@st.cache_data
def load_shap_data():

    if os.path.exists(SHAP_DATA):
        return pd.read_csv(
            SHAP_DATA
        )

    return None


df = load_field_data()
anomaly_df = load_anomaly_data()
irrigation_df = load_irrigation_data()
shap_df = load_shap_data()


# ============================================================
# Backend Health
# ============================================================

def backend_health():

    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=3
        )

        if response.status_code == 200:
            return True, response.json()

    except Exception:
        pass

    return False, None


backend_online, health_data = backend_health()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.markdown(
    "# 🚁 AeroCrop AI"
)

st.sidebar.caption(
    "Autonomous Precision Agriculture Platform"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Executive Overview",
        "🗺️ Field Intelligence",
        "🚁 Drone Imagery",
        "🧠 AI Predictions",
        "🔍 Explainable AI",
        "👤 Intervention Approval",
        "📋 Audit Log",
        "💬 Farmer AI Assistant"
    ]
)

st.sidebar.divider()

if backend_online:

    st.sidebar.success(
        "● AI Backend Online"
    )

else:

    st.sidebar.error(
        "● AI Backend Offline"
    )

st.sidebar.caption(
    "AeroCrop Drone AI v1.0"
)



# ============================================================
# Premium UI Helpers
# ============================================================

def is_positive_flag(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) == 1.0
    return str(value).strip().lower() in {
        "1", "true", "yes", "y", "pest", "pest_zone",
        "required", "high"
    }


def page_header(title, subtitle, eyebrow="AEROCROP AI • MISSION CONTROL"):
    if backend_online:
        status = (
            '<span class="ac-system-pill"><span class="ac-dot"></span>'
            'SYSTEM OPERATIONAL</span>'
        )
    else:
        status = (
            '<span class="ac-system-pill" style="color:#ff9aa7;'
            'border-color:rgba(255,92,112,.25);background:rgba(255,92,112,.05)">'
            '<span class="ac-dot" style="background:#ff5c70;'
            'box-shadow:0 0 7px #ff5c70,0 0 17px rgba(255,92,112,.55)"></span>'
            'BACKEND OFFLINE</span>'
        )
    st.markdown(
        f'''
        <div class="ac-header">
            <div>
                <div class="ac-eyebrow">{eyebrow}</div>
                <div class="main-title">{title}</div>
                <div class="subtitle">{subtitle}</div>
            </div>
            {status}
        </div>
        ''',
        unsafe_allow_html=True
    )


def ai_status_card(name, purpose):
    st.markdown(
        f'''
        <div class="ac-ai-status">
            <div>
                <div class="ac-ai-name">{name}</div>
                <div style="color:#6f8179;font-size:.70rem;margin-top:2px">{purpose}</div>
            </div>
            <div class="ac-ready"><span class="ac-dot"></span>READY</div>
        </div>
        ''',
        unsafe_allow_html=True
    )

# ============================================================
# PAGE 1 - EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    page_header(
        "🚁 AeroCrop Drone AI",
        "Autonomous Drone-Based Precision Pesticide & Irrigation System",
        "AEROCROP AI • AUTONOMOUS FIELD OPERATIONS"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_zones = len(df)

    healthy_zones = int(
        (df["stress_category"] == "Healthy").sum()
    )

    stress_zones = int(
        (df["crop_stress"] >= 0.50).sum()
    )

    pest_zones = int(
        df["pest_zone"].apply(is_positive_flag).sum()
    )

    irrigation_zones = int(
        (df["irrigation_required"] == 1).sum()
    )

    high_priority = int(
        (df["priority_category"] == "High").sum()
    )

    cols = st.columns(6)

    metrics = [
        ("Field Zones", total_zones),
        ("Healthy", healthy_zones),
        ("Crop Stress", stress_zones),
        ("Pest Zones", pest_zones),
        ("Irrigation Required", irrigation_zones),
        ("High Spray Priority", high_priority)
    ]

    for col, (label, value) in zip(
        cols,
        metrics
    ):

        with col:

            st.metric(
                label,
                value
            )

    st.divider()

    # --------------------------------------------------------
    # Charts
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "### 🌱 Crop Health Distribution"
        )

        health_counts = df[
            "stress_category"
        ].value_counts().reset_index()

        health_counts.columns = [
            "Category",
            "Count"
        ]

        fig = px.pie(
            health_counts,
            names="Category",
            values="Count",
            hole=0.45
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    with col2:

        st.markdown(
            "### 🧪 Spray Priority Distribution"
        )

        priority_counts = df[
            "priority_category"
        ].value_counts().reset_index()

        priority_counts.columns = [
            "Priority",
            "Count"
        ]

        fig = px.bar(
            priority_counts,
            x="Priority",
            y="Count",
            text="Count"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    # --------------------------------------------------------
    # AI Status
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 AI Engine Status"
    )

    status_cols = st.columns(5)

    engines = [
        "XGBoost\nSpray Priority",
        "Autoencoder\nAnomaly",
        "LSTM\nIrrigation",
        "CNN\nDrone Vision",
        "SHAP\nExplainability"
    ]

    engine_details = [
        ("XGBoost", "Spray priority scoring"),
        ("Autoencoder", "Canopy anomaly detection"),
        ("LSTM", "Irrigation forecasting"),
        ("CNN", "Drone image intelligence"),
        ("SHAP", "Explainable decisions")
    ]

    for col, (name, purpose) in zip(status_cols, engine_details):
        with col:
            ai_status_card(name, purpose)


# ============================================================
# PAGE 2 - FIELD INTELLIGENCE
# ============================================================

elif page == "🗺️ Field Intelligence":

    page_header(
        "🗺️ Field Intelligence",
        "Live field-zone intelligence and crop-risk visualization",
        "AEROCROP AI • FIELD OPERATIONS"
    )

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        crop_options = [
            "All"
        ] + sorted(
            df["crop_type"].unique().tolist()
        )

        selected_crop = st.selectbox(
            "Crop Type",
            crop_options
        )

    with col2:

        priority_options = [
            "All",
            "Low",
            "Medium",
            "High"
        ]

        selected_priority = st.selectbox(
            "Spray Priority",
            priority_options
        )

    with col3:

        stress_limit = st.slider(
            "Minimum Crop Stress",
            0.0,
            1.0,
            0.0,
            0.05
        )

    filtered = df.copy()

    if selected_crop != "All":

        filtered = filtered[
            filtered["crop_type"]
            == selected_crop
        ]

    if selected_priority != "All":

        filtered = filtered[
            filtered["priority_category"]
            == selected_priority
        ]

    filtered = filtered[
        filtered["crop_stress"]
        >= stress_limit
    ]

    st.info(
        f"Displaying {len(filtered):,} field zones"
    )

    # --------------------------------------------------------
    # Map
    # --------------------------------------------------------

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    field_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=13
    )

    # Limit markers for performance
    map_data = filtered.head(500)

    for _, row in map_data.iterrows():

        priority = row[
            "priority_category"
        ]

        if priority == "High":
            icon_color = "red"
        elif priority == "Medium":
            icon_color = "orange"
        else:
            icon_color = "green"

        popup = f"""
        <b>Zone:</b> {row['zone_id']}<br>
        <b>Crop:</b> {row['crop_type']}<br>
        <b>NDVI:</b> {row['ndvi']:.3f}<br>
        <b>Crop Stress:</b> {row['crop_stress']:.3f}<br>
        <b>Pest Probability:</b> {row['pest_probability']:.3f}<br>
        <b>Spray Priority:</b> {priority}<br>
        <b>Soil Moisture:</b> {row['soil_moisture']:.2f}
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"]
            ],
            radius=6,
            color=icon_color,
            fill=True,
            fill_opacity=0.7,
            popup=popup
        ).add_to(field_map)

    st_folium(
        field_map,
        width=None,
        height=600
    )

    st.dataframe(
        filtered[
            [
                "zone_id",
                "farm_id",
                "crop_type",
                "ndvi",
                "crop_stress",
                "pest_probability",
                "soil_moisture",
                "priority_category"
            ]
        ].head(100),
        width="stretch"
    )


# ============================================================
# PAGE 3 - DRONE IMAGERY
# ============================================================

elif page == "🚁 Drone Imagery":

    page_header(
        "🚁 Drone Image Intelligence",
        "CNN-based crop stress and pest-zone analysis",
        "AEROCROP AI • VISION INTELLIGENCE"
    )

    uploaded_file = st.file_uploader(
        "Upload a drone image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Drone Image",
                width="stretch"
            )

        with col2:

            st.markdown(
                "### 🧠 CNN Analysis"
            )

            resized = image.resize(
                (128, 128)
            )

            image_array = np.array(
                resized
            )

            flattened = image_array.flatten().tolist()

            try:

                response = requests.post(
                    f"{API_URL}/analyze/drone-image",
                    json={
                        "image_features": flattened
                    },
                    timeout=30
                )

                if response.status_code == 200:

                    result = response.json()

                    prediction = result[
                        "prediction"
                    ]

                    confidence = result[
                        "confidence"
                    ]

                    st.metric(
                        "AI Classification",
                        prediction.replace(
                            "_",
                            " "
                        ).title()
                    )

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    probabilities = result[
                        "probabilities"
                    ]

                    prob_df = pd.DataFrame({
                        "Class": list(
                            probabilities.keys()
                        ),
                        "Probability": list(
                            probabilities.values()
                        )
                    })

                    fig = px.bar(
                        prob_df,
                        x="Class",
                        y="Probability",
                        text_auto=".2f"
                    )

                    st.plotly_chart(
                        fig,
                        width="stretch"
                    )

                else:

                    st.error(
                        response.text
                    )

            except Exception as e:

                st.error(
                    f"Backend connection error: {e}"
                )

    else:

        st.info(
            "Upload a drone image to start CNN analysis."
        )


# ============================================================
# PAGE 4 - AI PREDICTIONS
# ============================================================

elif page == "🧠 AI Predictions":

    page_header(
        "🧠 AI Predictions",
        "Multi-model decision intelligence for precision intervention",
        "AEROCROP AI • DECISION INTELLIGENCE"
    )

    zone_id = st.selectbox(
        "Select Field Zone",
        df["zone_id"].astype(str).tolist()
    )

    zone = df[
        df["zone_id"].astype(str)
        == str(zone_id)
    ].iloc[0]

    # --------------------------------------------------------
    # Zone information
    # --------------------------------------------------------

    st.markdown(
        "### 📍 Selected Zone"
    )

    info_cols = st.columns(5)

    zone_metrics = [
        ("Crop", zone["crop_type"]),
        ("NDVI", f"{zone['ndvi']:.3f}"),
        ("Crop Stress", f"{zone['crop_stress']:.3f}"),
        ("Pest Probability", f"{zone['pest_probability']:.3f}"),
        ("Soil Moisture", f"{zone['soil_moisture']:.2f}")
    ]

    for col, (label, value) in zip(
        info_cols,
        zone_metrics
    ):

        with col:

            st.metric(
                label,
                value
            )

    st.divider()

    # --------------------------------------------------------
    # Spray Priority
    # --------------------------------------------------------

    st.markdown(
        "### 🧪 XGBoost Spray Priority"
    )

    spray_payload = {
        feature: float(
            zone[feature]
        )
        for feature in [
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
    }

    try:

        response = requests.post(
            f"{API_URL}/predict/spray-priority",
            json=spray_payload,
            timeout=20
        )

        if response.status_code == 200:

            spray_result = response.json()

            priority = spray_result[
                "spraying_priority"
            ]

            if priority == "High":

                st.error(
                    f"🚨 HIGH SPRAY PRIORITY — Zone {zone_id}"
                )

            elif priority == "Medium":

                st.warning(
                    f"⚠️ MEDIUM SPRAY PRIORITY — Zone {zone_id}"
                )

            else:

                st.success(
                    f"✓ LOW SPRAY PRIORITY — Zone {zone_id}"
                )

        else:

            st.error(
                response.text
            )

    except Exception as e:

        st.error(
            f"Backend error: {e}"
        )

    # --------------------------------------------------------
    # Anomaly Detection
    # --------------------------------------------------------

    st.markdown(
        "### 🔍 Canopy Anomaly Detection"
    )

    anomaly_payload = {
        "blue_band": float(zone["blue_band"]),
        "green_band": float(zone["green_band"]),
        "red_band": float(zone["red_band"]),
        "red_edge_band": float(zone["red_edge_band"]),
        "nir_band": float(zone["nir_band"]),
        "ndvi": float(zone["ndvi"]),
        "ndre": float(zone["ndre"]),
        "thermal_temperature": float(
            zone["thermal_temperature"]
        )
    }

    try:

        response = requests.post(
            f"{API_URL}/detect/anomaly",
            json=anomaly_payload,
            timeout=20
        )

        if response.status_code == 200:

            anomaly_result = response.json()

            anomaly_score = anomaly_result[
                "anomaly_score"
            ]

            anomaly_status = anomaly_result[
                "status"
            ]

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Anomaly Score",
                    f"{anomaly_score:.4f}"
                )

            with c2:

                if anomaly_status == "Anomalous":

                    st.error(
                        "⚠️ Anomalous Canopy Pattern"
                    )

                else:

                    st.success(
                        "✓ Normal Canopy Pattern"
                    )

    except Exception as e:

        st.error(
            f"Backend error: {e}"
        )

    # --------------------------------------------------------
    # Irrigation Forecast
    # --------------------------------------------------------

    st.markdown(
        "### 💧 LSTM Irrigation Forecast"
    )

    base_values = np.array([
        zone["soil_moisture"],
        zone["temperature"],
        zone["humidity"],
        zone["rainfall"]
    ])

    sequence = []

    for day in range(7):

        variation = np.array([
            np.sin(day / 2) * 1.5,
            np.cos(day / 3) * 1.0,
            np.sin(day / 3) * 2.0,
            max(0, np.cos(day / 2)) * 0.5
        ])

        values = (
            base_values
            + variation
        )

        sequence.append(
            values.tolist()
        )

    try:

        response = requests.post(
            f"{API_URL}/predict/irrigation",
            json={
                "sequence": sequence
            },
            timeout=20
        )

        if response.status_code == 200:

            irrigation_result = response.json()

            demand = irrigation_result[
                "predicted_irrigation_demand"
            ]

            recommendation = irrigation_result[
                "recommendation"
            ]

            c1, c2 = st.columns(2)

            with c1:

                st.metric(
                    "Predicted Demand",
                    f"{demand:.2f}"
                )

            with c2:

                if recommendation == "HIGH":

                    st.error(
                        "💧 HIGH Irrigation Requirement"
                    )

                elif recommendation == "MEDIUM":

                    st.warning(
                        "💧 MEDIUM Irrigation Requirement"
                    )

                else:

                    st.success(
                        "✓ LOW Irrigation Requirement"
                    )

    except Exception as e:

        st.error(
            f"Backend error: {e}"
        )


# ============================================================
# PAGE 5 - EXPLAINABLE AI
# ============================================================

elif page == "🔍 Explainable AI":

    page_header(
        "🔍 Explainable AI",
        "Understanding why AeroCrop AI makes intervention recommendations",
        "AEROCROP AI • MODEL TRANSPARENCY"
    )

    if shap_df is not None:

        shap_display = shap_df.copy()

        shap_display = shap_display.sort_values(
            "mean_abs_shap",
            ascending=True
        )

        fig = px.bar(
            shap_display,
            x="mean_abs_shap",
            y="feature",
            orientation="h",
            title="Global SHAP Feature Importance"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.markdown(
            "### 🧠 Top AI Decision Drivers"
        )

        top5 = shap_df.sort_values(
            "mean_abs_shap",
            ascending=False
        ).head(5)

        for index, row in top5.iterrows():

            st.write(
                f"**{index + 1}. "
                f"{row['feature']}** — "
                f"SHAP: "
                f"{row['mean_abs_shap']:.4f}"
            )

        st.info(
            "SHAP values represent feature contribution "
            "to the model's decision. Higher mean absolute "
            "SHAP indicates greater global influence."
        )

    else:

        st.warning(
            "SHAP results file not found."
        )


# ============================================================
# PAGE 6 - INTERVENTION APPROVAL
# ============================================================

elif page == "👤 Intervention Approval":

    page_header(
        "👤 Human Approval Center",
        "Human-in-the-loop safety gate before simulated intervention",
        "AEROCROP AI • CONTROL GATE"
    )

    st.warning(
        "⚠️ Physical pesticide or irrigation actuators are "
        "NOT connected. All intervention execution is simulation-only."
    )

    zone_id = st.selectbox(
        "Field Zone",
        df["zone_id"].astype(str).tolist(),
        key="approval_zone"
    )

    intervention = st.selectbox(
        "Intervention Type",
        [
            "spraying",
            "irrigation"
        ]
    )

    approved_by = st.text_input(
        "Approved By",
        placeholder="Farm Manager / Supervisor"
    )

    reason = st.text_area(
        "Approval Reason",
        placeholder="Enter reason for approving intervention..."
    )

    if st.button(
        "✅ Approve Simulated Intervention",
        type="primary"
    ):

        if not approved_by:

            st.error(
                "Please enter approver name."
            )

        else:

            payload = {
                "zone_id": str(zone_id),
                "intervention_type": intervention,
                "approved_by": approved_by,
                "reason": reason
            }

            try:

                response = requests.post(
                    f"{API_URL}/intervention/approve",
                    json=payload,
                    timeout=20
                )

                if response.status_code == 200:

                    result = response.json()

                    st.toast("Intervention approved • Audit trail updated", icon="✅")
                    st.markdown(
                        f'<div class="ac-card" style="border-color:rgba(73,242,166,.28);box-shadow:0 0 34px rgba(73,242,166,.10)">'
                        f'<div class="ac-card-title">ACTION APPROVED</div>'
                        f'<div style="font-size:1.35rem;font-weight:850;color:#7cffba">✓ Simulated intervention authorized</div>'
                        f'<div class="ac-card-note">Zone {zone_id} • {intervention.title()} • Human approval recorded</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    with st.expander("View API audit response"):
                        st.json(result)

                else:

                    st.error(
                        response.text
                    )

            except Exception as e:

                st.error(
                    f"Backend error: {e}"
                )


# ============================================================
# PAGE 7 - AUDIT LOG
# ============================================================

elif page == "📋 Audit Log":

    page_header(
        "📋 Intervention Audit Log",
        "Traceability of AI recommendations and approved interventions",
        "AEROCROP AI • GOVERNANCE & TRACEABILITY"
    )

    try:

        response = requests.get(
            f"{API_URL}/audit-log",
            timeout=10
        )

        if response.status_code == 200:

            result = response.json()

            records = result.get(
                "records",
                []
            )

            if records:

                audit_df = pd.DataFrame(records)
                total_events = len(audit_df)
                approved_events = int((audit_df["approval_status"].astype(str).str.upper() == "APPROVED").sum()) if "approval_status" in audit_df else 0
                simulated_events = int((audit_df["execution_mode"].astype(str).str.upper() == "SIMULATED").sum()) if "execution_mode" in audit_df else 0

                stat_cols = st.columns(3)
                stat_cols[0].metric("Total Events", total_events)
                stat_cols[1].metric("Approved", approved_events)
                stat_cols[2].metric("Simulated", simulated_events)

                st.markdown('<div class="ac-section">Recent intervention activity</div>', unsafe_allow_html=True)
                st.dataframe(
                    audit_df,
                    width="stretch",
                    hide_index=True
                )

            else:

                st.info(
                    "No intervention records available yet."
                )

    except Exception as e:

        st.error(
            f"Backend error: {e}"
        )


# ============================================================
# PAGE 8 - FARMER AI ASSISTANT
# ============================================================

elif page == "💬 Farmer AI Assistant":

    page_header(
        "💬 AeroCrop Copilot",
        "Ask questions about field conditions, AI predictions and interventions",
        "AEROCROP AI • FARM OPERATIONS ASSISTANT"
    )
    st.markdown(
        '<div class="ac-chat"><div class="ac-card-title">AI FIELD COPILOT</div>'
        '<div style="font-size:1.18rem;font-weight:800;color:#f2fff8">Ask AeroCrop about your fields.</div>'
        '<div style="color:#81928b;font-size:.82rem;margin-top:5px">Try spray priority, irrigation, pest risk or crop health.</div></div>',
        unsafe_allow_html=True
    )

    question = st.text_input(
        "Ask AeroCrop AI",
        placeholder=(
            "Example: Which zones have high spray priority?"
        )
    )

    if question:

        q = question.lower()

        if (
            "spray" in q
            or "pesticide" in q
        ):

            high = df[
                df["priority_category"]
                == "High"
            ]

            st.success(
                f"There are **{len(high):,}** "
                "zones currently classified as High spray priority."
            )

            st.dataframe(
                high[
                    [
                        "zone_id",
                        "crop_type",
                        "pest_probability",
                        "crop_stress",
                        "priority_category"
                    ]
                ].head(20),
                width="stretch"
            )

        elif (
            "irrigation" in q
            or "water" in q
        ):

            irrigation = df[
                df["irrigation_required"]
                == 1
            ]

            st.info(
                f"**{len(irrigation):,}** zones "
                "are marked as requiring irrigation."
            )

        elif (
            "pest" in q
            or "infestation" in q
        ):

            pest = df[
                df["pest_zone"].apply(is_positive_flag)
            ]

            st.warning(
                f"**{len(pest):,}** zones "
                "are classified as pest zones."
            )

        elif (
            "stress" in q
            or "health" in q
        ):

            st.info(
                f"Average crop stress across "
                f"the dataset is "
                f"**{df['crop_stress'].mean():.3f}**."
            )

        else:

            st.info(
                "I can help with spray priority, "
                "irrigation, pest zones and crop stress."
            )
st.markdown(
    '<div class="ac-footer">AeroCrop AI • Autonomous Precision Agriculture • Human-approved simulated interventions only</div>',
    unsafe_allow_html=True
)
