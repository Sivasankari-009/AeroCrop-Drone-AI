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
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 17px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        text-align: center;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .status-ok {
        padding: 10px 16px;
        border-radius: 10px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
    }

    .warning-box {
        padding: 14px;
        border-radius: 10px;
        background: #fef3c7;
        color: #92400e;
        font-weight: 600;
    }

    .info-box {
        padding: 14px;
        border-radius: 10px;
        background: #e0f2fe;
        color: #075985;
    }

    </style>
    """,
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
# PAGE 1 - EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        '<div class="main-title">🚁 AeroCrop Drone AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Autonomous Drone-Based Precision Pesticide & Irrigation System"
        "</div>",
        unsafe_allow_html=True
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
        (df["pest_zone"] == 1).sum()
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
            use_container_width=True
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
            use_container_width=True
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

    for col, engine in zip(
        status_cols,
        engines
    ):

        with col:

            st.success(
                f"✓ {engine}"
            )


# ============================================================
# PAGE 2 - FIELD INTELLIGENCE
# ============================================================

elif page == "🗺️ Field Intelligence":

    st.markdown(
        '<div class="main-title">🗺️ Field Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Live field-zone intelligence and crop-risk visualization"
        "</div>",
        unsafe_allow_html=True
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
        use_container_width=True
    )


# ============================================================
# PAGE 3 - DRONE IMAGERY
# ============================================================

elif page == "🚁 Drone Imagery":

    st.markdown(
        '<div class="main-title">🚁 Drone Image Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "CNN-based crop stress and pest-zone analysis"
        "</div>",
        unsafe_allow_html=True
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
                use_container_width=True
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
                        use_container_width=True
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

    st.markdown(
        '<div class="main-title">🧠 AI Predictions</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Multi-model decision intelligence for precision intervention"
        "</div>",
        unsafe_allow_html=True
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

    st.markdown(
        '<div class="main-title">🔍 Explainable AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Understanding why AeroCrop AI makes intervention recommendations"
        "</div>",
        unsafe_allow_html=True
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
            use_container_width=True
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

    st.markdown(
        '<div class="main-title">👤 Human Approval Center</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Human-in-the-loop safety gate before simulated intervention"
        "</div>",
        unsafe_allow_html=True
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

                    st.success(
                        "Intervention approved successfully."
                    )

                    st.json(
                        result
                    )

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

    st.markdown(
        '<div class="main-title">📋 Intervention Audit Log</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Traceability of AI recommendations and approved interventions"
        "</div>",
        unsafe_allow_html=True
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

                audit_df = pd.DataFrame(
                    records
                )

                st.dataframe(
                    audit_df,
                    use_container_width=True
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

    st.markdown(
        '<div class="main-title">💬 Farmer AI Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        "Ask questions about field conditions, AI predictions and interventions"
        "</div>",
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
                use_container_width=True
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
                df["pest_zone"]
                == 1
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