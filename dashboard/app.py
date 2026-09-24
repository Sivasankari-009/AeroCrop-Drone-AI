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
from datetime import datetime


# ============================================================
# AeroCrop Drone AI - Main Dashboard
# ============================================================

st.set_page_config(
    page_title="AeroCrop Drone AI",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# Configuration
# ============================================================

API_URL = "https://aerocrop-drone-ai.onrender.com"

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
        --ac-bg:#f5faf7; --ac-panel:#ffffff; --ac-border:#dfece5;
        --ac-green:#159b63; --ac-green-dark:#087a4a; --ac-cyan:#168bb5;
        --ac-amber:#d98b08; --ac-red:#d94a5f; --ac-text:#183128; --ac-muted:#6f8179;
    }
    .stApp {
        background:
          radial-gradient(circle at 85% 5%, rgba(21,155,99,.09), transparent 24%),
          radial-gradient(circle at 15% 12%, rgba(22,139,181,.06), transparent 20%),
          linear-gradient(180deg,#f8fcfa 0%,#f3f8f5 100%);
        color:var(--ac-text);
    }
    .main .block-container {max-width:1540px;padding-top:1.0rem;padding-bottom:3rem;}
    header[data-testid="stHeader"] {background:transparent;}
    section[data-testid="stSidebar"] {display:none!important;}
    div[data-testid="stSidebarNav"] {display:none!important;}

    .ac-topbar {
        display:flex;align-items:center;justify-content:space-between;gap:18px;
        padding:12px 18px;margin-bottom:14px;background:rgba(255,255,255,.88);
        border:1px solid var(--ac-border);border-radius:18px;
        box-shadow:0 8px 28px rgba(21,80,52,.07);backdrop-filter:blur(12px);
    }
    .ac-brand {display:flex;align-items:center;gap:11px;min-width:max-content;}
    .ac-brand-icon {
        width:42px;height:42px;border-radius:13px;display:flex;align-items:center;justify-content:center;
        background:linear-gradient(135deg,#dff8eb,#e7f7ff);border:1px solid #cdebdc;font-size:1.35rem;
    }
    .ac-brand-name {font-size:1.08rem;font-weight:900;color:#123b2b;letter-spacing:-.02em;}
    .ac-brand-sub {font-size:.67rem;color:#789087;margin-top:1px;}
    .ac-nav-note {color:#789087;font-size:.72rem;font-weight:700;text-align:right;}
    .ac-header {display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:10px 0 22px;}
    .ac-eyebrow {color:var(--ac-green-dark);font-size:.72rem;font-weight:850;letter-spacing:.16em;text-transform:uppercase;margin-bottom:7px;}
    .main-title {font-size:clamp(2rem,4vw,3.05rem);line-height:1.05;font-weight:900;letter-spacing:-.045em;margin-bottom:6px;color:#14372a;}
    .subtitle {font-size:.98rem;color:var(--ac-muted);margin-bottom:18px;}
    .ac-system-pill {white-space:nowrap;padding:10px 15px;border:1px solid #bfe5d1;border-radius:999px;background:#effaf4;color:#16794f;font-size:.78rem;font-weight:800;box-shadow:0 5px 18px rgba(21,155,99,.08);}
    .ac-dot {display:inline-block;width:7px;height:7px;margin-right:7px;border-radius:50%;background:var(--ac-green);box-shadow:0 0 7px rgba(21,155,99,.35);animation:ac-pulse 1.8s infinite;}
    @keyframes ac-pulse {0%,100%{transform:scale(1);opacity:.72}50%{transform:scale(1.28);opacity:1}}

    [data-testid="stMetric"] {
        background:rgba(255,255,255,.94);border:1px solid var(--ac-border);border-radius:17px;
        padding:15px 17px;min-height:112px;box-shadow:0 8px 24px rgba(21,80,52,.055);
        transition:transform .2s ease,border-color .2s ease,box-shadow .2s ease;
    }
    [data-testid="stMetric"]:hover {transform:translateY(-3px);border-color:#b9deca;box-shadow:0 14px 30px rgba(21,80,52,.10);}
    [data-testid="stMetricLabel"] {color:#72857c!important;font-size:.72rem!important;font-weight:800!important;text-transform:uppercase;letter-spacing:.08em;}
    [data-testid="stMetricValue"] {color:#17392c!important;font-size:1.85rem!important;font-weight:900!important;}

    .stButton > button {
        border-radius:12px!important;border:1px solid #b7dfca!important;
        background:linear-gradient(135deg,#eaf9f1,#ffffff)!important;color:#126c46!important;
        font-weight:850!important;min-height:44px;box-shadow:0 5px 15px rgba(21,155,99,.07);
        transition:all .18s ease!important;
    }
    .stButton > button:hover {transform:translateY(-2px);border-color:#72c99d!important;box-shadow:0 10px 22px rgba(21,155,99,.13)!important;}

    div[data-baseweb="select"] > div,div[data-baseweb="input"] > div,textarea {
        background:#ffffff!important;border-color:#d9e9e1!important;border-radius:11px!important;color:#183128!important;
    }
    div[data-baseweb="select"] > div:focus-within,div[data-baseweb="input"] > div:focus-within,textarea:focus {
        border-color:#78c8a0!important;box-shadow:0 0 0 3px rgba(21,155,99,.08)!important;
    }
    div[data-testid="stAlert"] {border-radius:13px;border:1px solid #d5e8df;box-shadow:0 8px 20px rgba(21,80,52,.045);}
    [data-testid="stDataFrame"] {border:1px solid var(--ac-border);border-radius:14px;overflow:hidden;box-shadow:0 10px 28px rgba(21,80,52,.07);}
    [data-testid="stFileUploaderDropzone"] {border:1px dashed #b8dcca;border-radius:15px;background:#fbfefc;}
    hr {border-color:#e2eee8!important;}

    .ac-card {
        background:rgba(255,255,255,.94);border:1px solid var(--ac-border);border-radius:18px;padding:20px;
        box-shadow:0 12px 32px rgba(21,80,52,.065);transition:all .18s ease;
    }
    .ac-card:hover {border-color:#c0e1cf;box-shadow:0 16px 38px rgba(21,80,52,.09);}
    .ac-card-title {font-size:.76rem;color:#6e8278;font-weight:850;letter-spacing:.10em;text-transform:uppercase;margin-bottom:8px;}
    .ac-card-note {color:#7f9189;font-size:.78rem;margin-top:6px;}
    .ac-ai-status {display:flex;align-items:center;justify-content:space-between;padding:13px 15px;margin:7px 0;border-radius:13px;background:#f3fbf6;border:1px solid #dcefe4;}
    .ac-ai-name {font-weight:800;color:#244438;}
    .ac-ready {color:#168354;font-size:.72rem;font-weight:850;letter-spacing:.08em;}
    .ac-section {font-size:1.12rem;font-weight:850;letter-spacing:-.02em;margin:12px 0 13px;color:#193c2e;}
    .ac-chat {background:linear-gradient(145deg,#ffffff,#f3fbf7);border:1px solid #dceee5;border-radius:20px;padding:22px;box-shadow:0 10px 30px rgba(21,80,52,.06);margin-bottom:16px;}
    .ac-footer {margin-top:35px;padding-top:15px;border-top:1px solid #e1ece6;color:#71847a;font-size:.72rem;text-align:center;letter-spacing:.04em;}

    div[data-testid="stRadio"] > div {gap:7px!important;justify-content:center!important;flex-wrap:wrap;}
    div[data-testid="stRadio"] > div > label {
        background:#ffffff!important;border:1px solid #deebe4!important;border-radius:999px!important;
        padding:7px 13px!important;box-shadow:0 3px 10px rgba(21,80,52,.04);transition:all .16s ease!important;
    }
    div[data-testid="stRadio"] > div > label:hover {border-color:#9ed4b8!important;background:#f3fbf6!important;transform:translateY(-1px);}
    div[data-testid="stRadio"] label p {font-size:.72rem!important;font-weight:800!important;color:#36584a!important;}
    .ac-hero{position:relative;overflow:hidden;border-radius:0 0 22px 22px;min-height:178px;margin:0 -2rem 0 -2rem;padding:34px 38px 26px;background-image:linear-gradient(90deg,rgba(245,252,248,.98) 0%,rgba(245,252,248,.88) 35%,rgba(245,252,248,.22) 72%,rgba(245,252,248,.08) 100%),url("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1800&q=85");background-size:cover;background-position:center 58%;box-shadow:0 12px 28px rgba(20,80,55,.10);}
    .ac-hero-inner{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;gap:20px}.ac-hero-left{display:flex;align-items:center;gap:17px}.ac-map-icon{width:54px;height:54px;border-radius:14px;background:linear-gradient(135deg,#d9f4e5,#dcefff);display:flex;align-items:center;justify-content:center;font-size:29px;border:1px solid #c4e7d5;box-shadow:0 8px 20px rgba(21,155,99,.12)}.ac-hero-kicker{font-size:.73rem;font-weight:900;letter-spacing:.19em;color:#087a4a;margin-bottom:7px}.ac-hero-title{font-size:clamp(2.35rem,4vw,3.55rem);font-weight:950;letter-spacing:-.055em;line-height:.95;color:#10273b}.ac-hero-title span{color:#159b63}.ac-hero-sub{font-size:1rem;color:#355064;margin-top:9px;font-weight:650}.ac-hero-stats{display:flex;align-items:center;gap:0;background:rgba(255,255,255,.92);border:1px solid #d9e9e2;border-radius:17px;padding:12px 10px;box-shadow:0 10px 28px rgba(30,80,70,.12);min-width:520px}.ac-hstat{padding:0 18px;min-width:150px}.ac-hstat+.ac-hstat{border-left:1px solid #e0eae6}.ac-hstat-top{display:flex;align-items:center;gap:9px;font-size:.9rem;font-weight:900;color:#203b4b}.ac-hstat-icon{width:30px;height:30px;border-radius:9px;background:#e9f8ef;display:flex;align-items:center;justify-content:center;font-size:17px}.ac-hstat-value{font-size:1.08rem;font-weight:950;color:#162d3e;margin-top:2px}.ac-hstat-note{font-size:.65rem;color:#71838d;margin-left:39px;margin-top:-2px}.ac-filterbar{background:rgba(255,255,255,.95);border:1px solid #dfece5;border-radius:18px;padding:13px 14px 8px;margin:0 0 12px;box-shadow:0 10px 28px rgba(25,80,60,.07)}.ac-map-shell{background:#fff;border:1px solid #dcebe3;border-radius:18px;padding:5px;box-shadow:0 14px 34px rgba(25,80,60,.08);overflow:hidden}.ac-side-card{background:rgba(255,255,255,.97);border:1px solid #dfece5;border-radius:18px;padding:16px;box-shadow:0 12px 30px rgba(25,80,60,.07)}.ac-side-title{font-size:1rem;font-weight:950;color:#172d3d;display:flex;align-items:center;gap:10px}.ac-side-sub{font-size:.72rem;color:#7b8b91;margin:3px 0 13px 34px}.ac-kpi-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:13px}.ac-kpi{border-radius:13px;padding:12px;border:1px solid #e0ece7;min-height:75px}.ac-kpi-blue{background:#f0f9ff;border-color:#d9eefb}.ac-kpi-red{background:#fff3f4;border-color:#f8dddd}.ac-kpi-amber{background:#fffaf0;border-color:#f5e6bd}.ac-kpi-green{background:#effaf4;border-color:#d6eddf}.ac-kpi-label{font-size:.67rem;font-weight:850}.ac-kpi-value{font-size:1.23rem;font-weight:950;margin-top:2px}.ac-kpi-blue .ac-kpi-label,.ac-kpi-blue .ac-kpi-value{color:#1683b5}.ac-kpi-red .ac-kpi-label,.ac-kpi-red .ac-kpi-value{color:#d6454d}.ac-kpi-amber .ac-kpi-label,.ac-kpi-amber .ac-kpi-value{color:#c78209}.ac-kpi-green .ac-kpi-label,.ac-kpi-green .ac-kpi-value{color:#168354}.ac-district-row{display:grid;grid-template-columns:110px 1fr 58px;gap:8px;align-items:center;margin:7px 0;font-size:.67rem;color:#42545e;font-weight:750}.ac-bar{height:7px;border-radius:99px;background:#edf3f0;overflow:hidden}.ac-bar>span{display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2ebd87,#19a96f)}.ac-map-legend{background:rgba(9,29,43,.92);color:#fff;border-radius:12px;padding:11px 13px;font-size:.68rem;box-shadow:0 8px 20px rgba(0,0,0,.18)}.ac-legend-title{font-weight:900;font-size:.72rem;margin-bottom:7px}.ac-legend-item{display:flex;align-items:center;gap:7px;margin:5px 0}.ac-dot-risk{width:11px;height:11px;border-radius:50%;display:inline-block}.ac-dot-high{background:#ef3f4d}.ac-dot-med{background:#f6a20a}.ac-dot-low{background:#42c83b}
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
# Top Navigation
# ============================================================

st.markdown(
    """
    <div class="ac-topbar">
        <div class="ac-brand">
            <div class="ac-brand-icon" style="background:linear-gradient(135deg,#35c987,#1eaa73);color:#fff;border:none;box-shadow:0 8px 20px rgba(21,155,99,.20)">🌿</div>
            <div>
                <div class="ac-brand-name" style="font-size:1.22rem">AeroCrop <span style="color:#159b63">AI</span></div>
                <div class="ac-brand-sub">Autonomous Precision Agriculture Platform</div>
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:10px;flex:1;justify-content:flex-end;">
            <div style="width:42px;height:42px;border:1px solid #dfe9e5;border-radius:13px;background:#fff;display:flex;align-items:center;justify-content:center;font-size:20px">☼</div>
            <div class="ac-system-pill"><span class="ac-dot"></span>SYSTEM OPERATIONAL</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

page = st.radio(
    "Navigation",
    [
        "🏠 Executive Overview",
        "🗺️ Field Intelligence",
        "🚁 Drone Imagery",
        "🛰️ Multispectral Drone Intelligence",
        "🧠 AI Predictions",
        "🔍 Explainable AI",
        "👤 Intervention Approval",
        "📋 Audit Log",
        "💬 Farmer AI Assistant"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

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

    total_zones = len(df)
    high_count = int((df["priority_category"] == "High").sum())
    med_count = int((df["priority_category"] == "Medium").sum())
    low_count = int((df["priority_category"] == "Low").sum())

    st.markdown(
        f"""
        <div class="ac-hero">
          <div class="ac-hero-inner">
            <div class="ac-hero-left">
              <div class="ac-map-icon">🗺️</div>
              <div>
                <div class="ac-hero-kicker">AEROCROP AI • FIELD OPERATIONS</div>
                <div class="ac-hero-title">Field <span>Intelligence</span></div>
                <div class="ac-hero-sub">Live field-zone intelligence and crop-risk visualization across Tamil Nadu</div>
              </div>
            </div>
            <div class="ac-hero-stats">
              <div class="ac-hstat"><div class="ac-hstat-top"><span class="ac-hstat-icon">📍</span>Tamil Nadu</div><div class="ac-hstat-note">All 38 Districts</div></div>
              <div class="ac-hstat"><div class="ac-hstat-top"><span class="ac-hstat-icon">🗄️</span>{total_zones:,}</div><div class="ac-hstat-note">Field Zones</div></div>
              <div class="ac-hstat"><div class="ac-hstat-top"><span class="ac-hstat-icon">🗓️</span>{datetime.now().strftime('%b %d, %Y')}</div><div class="ac-hstat-note">Last Analysis</div></div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div style="margin-top:-4px;margin-bottom:10px;padding:7px 12px;border-radius:10px;background:#fff7df;border:1px solid #f0d89b;color:#735f26;font-size:.72rem;font-weight:800">⚠️ DEMO MODE • 38-district Tamil Nadu coverage uses simulated field coordinates for presentation. AI values are based on the project dataset.</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    districts = [
        "All Tamil Nadu","Ariyalur","Chengalpattu","Chennai","Coimbatore","Cuddalore","Dharmapuri",
        "Dindigul","Erode","Kallakurichi","Kanchipuram","Kanyakumari","Karur","Krishnagiri","Madurai",
        "Mayiladuthurai","Nagapattinam","Namakkal","Nilgiris","Perambalur","Pudukkottai","Ramanathapuram",
        "Ranipet","Salem","Sivaganga","Tenkasi","Thanjavur","Theni","Thoothukudi","Tiruchirappalli",
        "Tirunelveli","Tiruppur","Tiruvallur","Tiruvannamalai","Tiruvarur","Vellore","Viluppuram","Virudhunagar"
    ]
    crop_options = ["All"] + sorted(df["crop_type"].dropna().unique().tolist())
    priority_options = ["All","Low","Medium","High"]
    risk_options = ["All","High Risk","Medium Risk","Low Risk"]

    st.markdown('<div class="ac-filterbar">', unsafe_allow_html=True)
    f1, f2, f3, f4, f5, f6 = st.columns([1.1,1.05,1.05,1.05,1.45,.72])
    with f1:
        selected_district = st.selectbox("📍 District", districts, key="fi_district")
    with f2:
        selected_crop = st.selectbox("🌿 Crop Type", crop_options, key="fi_crop")
    with f3:
        selected_priority = st.selectbox("🧪 Spray Priority", priority_options, key="fi_priority")
    with f4:
        selected_risk = st.selectbox("🛡️ Risk Level", risk_options, key="fi_risk")
    with f5:
        search_text = st.text_input("⌕ Search Location", placeholder="Enter district, city or village...", key="fi_search")
    with f6:
        st.markdown('<div style="height:27px"></div>', unsafe_allow_html=True)
        if st.button("↻ Reset Filters", key="fi_reset", width="stretch"):
            for k in ["fi_district","fi_crop","fi_priority","fi_risk","fi_search"]:
                st.session_state.pop(k, None)
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = df.copy()
    if selected_crop != "All":
        filtered = filtered[filtered["crop_type"] == selected_crop]
    if selected_priority != "All":
        filtered = filtered[filtered["priority_category"] == selected_priority]
    if selected_risk != "All":
        wanted = selected_risk.replace(" Risk", "")
        filtered = filtered[filtered["priority_category"] == wanted]
    if search_text.strip():
        q = search_text.strip().lower()
        mask = (
            filtered["zone_id"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["farm_id"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["crop_type"].astype(str).str.lower().str.contains(q, na=False)
            | filtered.get("district", pd.Series(index=filtered.index, dtype=str)).astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]
    if selected_district != "All Tamil Nadu" and "district" in filtered.columns:
        filtered = filtered[filtered["district"] == selected_district]

    TN_GEOJSON_URL = (
        "https://cdn.jsdelivr.net/gh/udit-001/india-maps-data@2884453/"
        "geojson/states/tamil-nadu.geojson"
    )

    @st.cache_data(ttl=86400)
    def load_tamil_nadu_boundary():
        try:
            response = requests.get(TN_GEOJSON_URL, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    field_map = folium.Map(
        location=[11.1271,78.6569], zoom_start=7, min_zoom=5, max_zoom=14,
        control_scale=True, tiles=None, zoom_control=True, prefer_canvas=True,
    )
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics",
        name="Satellite", overlay=False, control=True,
    ).add_to(field_map)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        attr="© OpenStreetMap contributors", name="Street Map", overlay=False, control=True,
    ).add_to(field_map)
    field_map.fit_bounds([[8.00,76.20],[13.60,80.40]], padding=(8,8))

    tn_boundary = load_tamil_nadu_boundary()
    if tn_boundary:
        risk_colors = {"High": "#ef5350", "Medium": "#f6b73c", "Low": "#4fc45b"}
        district_risk = {}
        if "district" in df.columns:
            for d, g in df.groupby("district"):
                district_risk[str(d).lower()] = str(g["priority_category"].value_counts().idxmax()) if len(g) else "Low"
        for feature in tn_boundary.get("features", []):
            props = feature.setdefault("properties", {})
            name = props.get("district") or props.get("DISTRICT") or props.get("district_name") or props.get("NAME_2") or props.get("name") or "Tamil Nadu"
            props["_aero_name"] = str(name)
            risk = district_risk.get(str(name).lower(), "Low")
            props["_aero_color"] = risk_colors.get(risk, "#4fc45b")
            props["_aero_risk"] = risk

        def district_style(feature):
            c = feature.get("properties", {}).get("_aero_color", "#58c95b")
            return {"color":"#ffffff","weight":1.7,"fillColor":c,"fillOpacity":0.46}
        def district_highlight(feature):
            return {"color":"#ffffff","weight":3,"fillColor":"#35b878","fillOpacity":0.58}

        folium.GeoJson(
            tn_boundary, name="Tamil Nadu Districts", style_function=district_style,
            highlight_function=district_highlight,
            tooltip=folium.GeoJsonTooltip(
                fields=["_aero_name", "_aero_risk"], aliases=["District", "Dominant Risk"], sticky=False, labels=True,
                style="background-color:white;color:#20333d;font-family:Arial;font-size:12px;padding:6px;",
            ),
        ).add_to(field_map)

        def coords_flat(coords):
            if not isinstance(coords, list): return []
            if coords and isinstance(coords[0], (int,float)): return [coords]
            out=[]
            for c in coords: out.extend(coords_flat(c))
            return out
        for feature in tn_boundary.get("features", []):
            pts = coords_flat((feature.get("geometry") or {}).get("coordinates", []))
            if not pts: continue
            lon = sum(p[0] for p in pts)/len(pts)
            lat = sum(p[1] for p in pts)/len(pts)
            name = feature.get("properties", {}).get("_aero_name", "")
            if not name: continue
            folium.Marker(
                [lat,lon],
                icon=folium.DivIcon(html=f'''<div style="font-family:Arial;font-size:10px;font-weight:800;color:#fff;text-shadow:0 1px 3px #123,0 -1px 3px #123;white-space:nowrap;transform:translate(-50%,-50%);">{name}</div>''')
            ).add_to(field_map)

    # ------------------------------------------------------------
    # District coverage markers
    # ------------------------------------------------------------
    # The dataset contains 38 district-labelled demo records.  At the
    # statewide zoom level, individual 4px points can disappear against
    # the colored district polygons.  Add one prominent summary marker
    # per district so every district remains visibly represented.
    district_summary = filtered.groupby("district", dropna=False).agg(
        zones=("zone_id", "count"),
        avg_ndvi=("ndvi", "mean"),
        avg_stress=("crop_stress", "mean"),
    ).reset_index()

    risk_rank = {"Low": 0, "Medium": 1, "High": 2}
    district_summary["dominant_risk"] = (
        filtered.groupby("district")["priority_category"]
        .agg(lambda s: max(s.value_counts().index, key=lambda x: risk_rank.get(str(x), 0)))
        .reindex(district_summary["district"])
        .fillna("Low")
        .values
    )

    for _, drow in district_summary.iterrows():
        district_name = str(drow["district"])
        group = filtered[filtered["district"] == district_name]
        if group.empty:
            continue
        center_lat = float(group["latitude"].mean())
        center_lon = float(group["longitude"].mean())
        risk = str(drow["dominant_risk"])
        marker_color = {"High":"#ef3f4d", "Medium":"#f6a20a", "Low":"#42c83b"}.get(risk, "#42c83b")
        popup_html = (
            f"<div style='font-family:Arial;min-width:235px'>"
            f"<div style='font-size:15px;font-weight:800'>{district_name}</div>"
            f"<div style='margin-top:5px'>Field zones: <b>{int(drow['zones']):,}</b></div>"
            f"<div>Dominant risk: <b style='color:{marker_color}'>{risk}</b></div>"
            f"<div>Avg NDVI: <b>{float(drow['avg_ndvi']):.3f}</b></div>"
            f"<div>Avg crop stress: <b>{float(drow['avg_stress']):.3f}</b></div>"
            f"<small>Demo district coverage • presentation data</small></div>"
        )
        folium.CircleMarker(
            location=[center_lat, center_lon],
            radius=9,
            color="#ffffff",
            weight=2.2,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.96,
            tooltip=f"{district_name} • {int(drow['zones']):,} zones • {risk} risk",
            popup=folium.Popup(popup_html, max_width=285),
        ).add_to(field_map)

    # ------------------------------------------------------------
    # Field-level points
    # ------------------------------------------------------------
    # Keep a representative set of actual project/demo coordinates.
    # Sampling by district guarantees that all selected districts are
    # represented instead of letting the first rows dominate the map.
    max_points = 900
    if len(filtered) > max_points:
        sample_parts = []
        per_district = max(4, max_points // max(1, filtered["district"].nunique()))
        for _, group in filtered.groupby("district", sort=True):
            n = min(len(group), per_district)
            sample_parts.append(group.sample(n=n, random_state=42))
        point_df = pd.concat(sample_parts, ignore_index=False)
        if len(point_df) > max_points:
            point_df = point_df.sample(n=max_points, random_state=42)
    else:
        point_df = filtered

    for _, row in point_df.iterrows():
        priority = row["priority_category"]
        icon_color = {"High":"#ef3f4d","Medium":"#f6a20a","Low":"#42c83b"}.get(priority,"#42c83b")
        folium.CircleMarker(
            location=[float(row["latitude"]), float(row["longitude"])],
            radius=5.2,
            color="#ffffff",
            weight=1.4,
            fill=True,
            fill_color=icon_color,
            fill_opacity=.94,
            popup=folium.Popup(
                f"<div style='font-family:Arial;min-width:210px'><b>{row['zone_id']}</b><br>District: {row.get('district','—')}<br>Crop: {row['crop_type']}<br>NDVI: {row['ndvi']:.3f}<br>Stress: {row['crop_stress']:.3f}<br>Priority: {priority}<br><small>Demo field coordinate</small></div>",
                max_width=250
            )
        ).add_to(field_map)

    folium.LayerControl(collapsed=True).add_to(field_map)
    legend = '<div class="ac-map-legend"><div class="ac-legend-title">Crop Risk Level</div><div class="ac-legend-item"><span class="ac-dot-risk ac-dot-high"></span>High Risk</div><div class="ac-legend-item"><span class="ac-dot-risk ac-dot-med"></span>Medium Risk</div><div class="ac-legend-item"><span class="ac-dot-risk ac-dot-low"></span>Low Risk</div><div style="margin-top:8px;border-top:1px solid rgba(255,255,255,.22);padding-top:7px">▰ Tamil Nadu Boundary</div><div>┄ District Boundary</div></div>'
    field_map.get_root().html.add_child(folium.Element(f'<div style="position:fixed;left:28px;bottom:24px;z-index:9999">{legend}</div>'))

    left, right = st.columns([3.35,1.15], gap="small")
    with left:
        st.markdown(f'<div style="font-size:.75rem;font-weight:800;color:#687d86;margin:0 0 7px 5px">Displaying {len(filtered):,} field zones</div>', unsafe_allow_html=True)
        st.markdown('<div class="ac-map-shell">', unsafe_allow_html=True)
        st_folium(field_map, width=None, height=575, returned_objects=[])
        st.markdown('</div>', unsafe_allow_html=True)

    district_counts = df.groupby("district").size().sort_values(ascending=False).to_dict() if "district" in df.columns else {}
    district_display = list(district_counts.items())
    with right:
        html = f'''<div class="ac-side-card">
          <div class="ac-side-title">📊 Tamil Nadu Overview</div>
          <div class="ac-side-sub">All districts field-zone distribution</div>
          <div class="ac-kpi-grid">
            <div class="ac-kpi ac-kpi-blue"><div class="ac-kpi-label">🗄️ Total Zones</div><div class="ac-kpi-value">{total_zones:,}</div></div>
            <div class="ac-kpi ac-kpi-red"><div class="ac-kpi-label">⚠️ High Risk</div><div class="ac-kpi-value">{high_count:,}</div></div>
            <div class="ac-kpi ac-kpi-amber"><div class="ac-kpi-label">❕ Medium Risk</div><div class="ac-kpi-value">{med_count:,}</div></div>
            <div class="ac-kpi ac-kpi-green"><div class="ac-kpi-label">🌿 Low Risk</div><div class="ac-kpi-value">{low_count:,}</div></div>
          </div>
          <div style="border-top:1px solid #e5eee9;padding-top:12px;font-weight:900;font-size:.84rem;color:#263d49">🗺️ District-wise Zone Count</div>
          <div style="font-size:.68rem;color:#7a8b92;margin:3px 0 9px">Presentation view — 38-district demo coverage with project AI values</div>'''
        max_v = max(district_counts.values()) if district_counts else 1
        for n,v in district_display:
            html += f'<div class="ac-district-row"><span>{n}</span><div class="ac-bar"><span style="width:{max(8,int(v/max_v*100))}%"></span></div><span style="text-align:right">{v:,}</span></div>'
        html += '''<div style="margin-top:12px;padding:9px 10px;border-radius:11px;background:#fff8e8;border:1px solid #f2dfb0;color:#6e5b2e;font-size:.66rem;line-height:1.4">⚠️ DEMO / SIMULATED DATA — district field points are generated for presentation coverage across all 38 Tamil Nadu districts. They are not live telemetry.</div>
        </div>'''
        st.markdown(html, unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.dataframe(filtered[["zone_id","farm_id","district","crop_type","ndvi","crop_stress","pest_probability","soil_moisture","priority_category"]].head(100), width="stretch")


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
# PAGE 4 - MULTISPECTRAL DRONE INTELLIGENCE
# ============================================================

elif page == "🛰️ Multispectral Drone Intelligence":

    page_header(
        "🛰️ Multispectral Drone Intelligence",
        "Multispectral + thermal field-zone intelligence",
        "AEROCROP AI • DRONE SENSOR FUSION"
    )

    # --------------------------------------------------------
    # File paths
    # --------------------------------------------------------

    BASE_DIR = os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )

    multispectral_dir = os.path.join(
        BASE_DIR,
        "data",
        "multispectral"
    )

    segmentation_dir = os.path.join(
        BASE_DIR,
        "data",
        "segmentation"
    )

    rgb_path = os.path.join(
        multispectral_dir,
        "drone_rgb_composite.png"
    )

    ndvi_path = os.path.join(
        multispectral_dir,
        "ndvi_map.png"
    )

    ndre_path = os.path.join(
        multispectral_dir,
        "ndre_map.png"
    )

    thermal_path = os.path.join(
        multispectral_dir,
        "thermal_map.png"
    )

    segmentation_path = os.path.join(
        segmentation_dir,
        "field_zone_segmentation.png"
    )

    drone_analysis_path = os.path.join(
        segmentation_dir,
        "drone_ai_zone_analysis.csv"
    )

    spray_analysis_path = os.path.join(
        segmentation_dir,
        "drone_spray_priority_analysis.csv"
    )

    # --------------------------------------------------------
    # Load drone analysis data
    # --------------------------------------------------------

    if os.path.exists(drone_analysis_path):

        drone_analysis_df = pd.read_csv(
            drone_analysis_path
        )

    else:

        drone_analysis_df = pd.DataFrame()

    if os.path.exists(spray_analysis_path):

        spray_analysis_df = pd.read_csv(
            spray_analysis_path
        )

    else:

        spray_analysis_df = pd.DataFrame()

    # --------------------------------------------------------
    # Sensor status
    # --------------------------------------------------------

    st.markdown(
        "### 🛰️ Sensor Pipeline Status"
    )

    status_cols = st.columns(5)

    with status_cols[0]:

        st.success(
            "✓ Multispectral"
        )

    with status_cols[1]:

        st.success(
            "✓ Thermal"
        )

    with status_cols[2]:

        st.success(
            "✓ Segmentation"
        )

    with status_cols[3]:

        st.success(
            "✓ CNN Vision"
        )

    with status_cols[4]:

        st.success(
            "✓ XGBoost"
        )

    # --------------------------------------------------------
    # KPI cards
    # --------------------------------------------------------

    st.markdown(
        "### 📊 Drone Intelligence Summary"
    )

    k1, k2, k3, k4 = st.columns(4)

    if not drone_analysis_df.empty:

        zone_count = len(
            drone_analysis_df
        )

        avg_ndvi = drone_analysis_df[
            "mean_ndvi"
        ].mean()

        avg_ndre = drone_analysis_df[
            "mean_ndre"
        ].mean()

        avg_thermal = drone_analysis_df[
            "mean_thermal_temperature"
        ].mean()

        with k1:

            st.metric(
                "Segmented Zones",
                zone_count
            )

        with k2:

            st.metric(
                "Average NDVI",
                f"{avg_ndvi:.3f}"
            )

        with k3:

            st.metric(
                "Average NDRE",
                f"{avg_ndre:.3f}"
            )

        with k4:

            st.metric(
                "Average Thermal",
                f"{avg_thermal:.2f} °C"
            )

    else:

        with k1:
            st.metric(
                "Segmented Zones",
                "—"
            )

        with k2:
            st.metric(
                "Average NDVI",
                "—"
            )

        with k3:
            st.metric(
                "Average NDRE",
                "—"
            )

        with k4:
            st.metric(
                "Average Thermal",
                "—"
            )

    # --------------------------------------------------------
    # Multispectral products
    # --------------------------------------------------------

    st.markdown(
        "### 🌿 Multispectral & Thermal Products"
    )

    image_cols = st.columns(2)

    with image_cols[0]:

        if os.path.exists(rgb_path):

            st.image(
                rgb_path,
                caption="Drone RGB Composite",
                width="stretch"
            )

        else:

            st.warning(
                "RGB composite not found."
            )

    with image_cols[1]:

        if os.path.exists(ndvi_path):

            st.image(
                ndvi_path,
                caption="NDVI Vegetation Index",
                width="stretch"
            )

        else:

            st.warning(
                "NDVI map not found."
            )

    image_cols_2 = st.columns(2)

    with image_cols_2[0]:

        if os.path.exists(ndre_path):

            st.image(
                ndre_path,
                caption="NDRE Vegetation Stress Index",
                width="stretch"
            )

        else:

            st.warning(
                "NDRE map not found."
            )

    with image_cols_2[1]:

        if os.path.exists(thermal_path):

            st.image(
                thermal_path,
                caption="Thermal Temperature Map",
                width="stretch"
            )

        else:

            st.warning(
                "Thermal map not found."
            )

    # --------------------------------------------------------
    # Field-zone segmentation
    # --------------------------------------------------------

    st.markdown(
        "### 🗺️ Field-Zone Segmentation"
    )

    if os.path.exists(segmentation_path):

        st.image(
            segmentation_path,
            caption=(
                "Image-Based Field-Zone "
                "Segmentation"
            ),
            width="stretch"
        )

    else:

        st.warning(
            "Segmentation output not found."
        )

    # --------------------------------------------------------
    # Zone AI analysis
    # --------------------------------------------------------

    st.markdown(
        "### 🤖 Zone-Level AI Analysis"
    )

    if not drone_analysis_df.empty:

        display_columns = [
            "zone_id",
            "pixel_count",
            "mean_ndvi",
            "mean_ndre",
            "mean_thermal_temperature",
            "vegetation_status",
            "cnn_prediction",
            "cnn_confidence",
            "pest_zone_probability"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in drone_analysis_df.columns
        ]

        display_df = drone_analysis_df[
            available_columns
        ].copy()

        if "cnn_confidence" in display_df.columns:

            display_df[
                "cnn_confidence"
            ] = (
                display_df[
                    "cnn_confidence"
                ] * 100
            ).round(2)

        if "pest_zone_probability" in display_df.columns:

            display_df[
                "pest_zone_probability"
            ] = (
                display_df[
                    "pest_zone_probability"
                ] * 100
            ).round(2)

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "Drone zone analysis data is unavailable."
        )

    # --------------------------------------------------------
    # Spray priority integration
    # --------------------------------------------------------

    st.markdown(
        "### 🎯 AI Spray Priority"
    )

    if not spray_analysis_df.empty:

        priority_counts = (
            spray_analysis_df[
                "spray_priority"
            ]
            .value_counts()
            .reset_index()
        )

        priority_counts.columns = [
            "Priority",
            "Zones"
        ]

        chart = px.bar(
            priority_counts,
            x="Priority",
            y="Zones",
            text_auto=True,
            title="Drone-Zone Spray Priority"
        )

        st.plotly_chart(
            chart,
            width="stretch"
        )

        priority_columns = [
            "drone_zone_id",
            "source_zone",
            "ndvi",
            "ndre",
            "thermal_temperature",
            "cnn_prediction",
            "cnn_confidence",
            "spray_priority"
        ]

        available_priority_columns = [
            column
            for column in priority_columns
            if column in spray_analysis_df.columns
        ]

        st.dataframe(
            spray_analysis_df[
                available_priority_columns
            ],
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "Drone spray-priority integration data "
            "is unavailable."
        )

    # --------------------------------------------------------
    # Pipeline explanation
    # --------------------------------------------------------

    st.markdown(
        "### 🔗 Autonomous AI Pipeline"
    )

    st.info(
        "Drone Imagery → Multispectral + Thermal "
        "Processing → Field-Zone Segmentation → "
        "CNN Crop Analysis → Feature Fusion → "
        "XGBoost Spray Priority → Human Approval → "
        "Simulated Intervention → Audit Log"
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

                approved_events = (
                    int(
                        (
                            audit_df[
                                "approval_status"
                            ]
                            .astype(str)
                            .str.upper()
                            == "APPROVED"
                        ).sum()
                    )
                    if "approval_status" in audit_df
                    else 0
                )

                simulated_events = (
                    int(
                        (
                            audit_df[
                                "execution_mode"
                            ]
                            .astype(str)
                            .str.upper()
                            == "SIMULATED"
                        ).sum()
                    )
                    if "execution_mode" in audit_df
                    else 0
                )

                # ------------------------------------------------
                # Cost and resource tracking
                # ------------------------------------------------

                if "estimated_cost" in audit_df:

                    cost_series = pd.to_numeric(
                        audit_df[
                            "estimated_cost"
                        ],
                        errors="coerce"
                    ).fillna(0)

                    total_cost = float(
                        cost_series.sum()
                    )

                else:

                    total_cost = 0.0

                if "resource_volume" in audit_df:

                    resource_series = pd.to_numeric(
                        audit_df[
                            "resource_volume"
                        ],
                        errors="coerce"
                    ).fillna(0)

                    total_resource = float(
                        resource_series.sum()
                    )

                else:

                    total_resource = 0.0

                # ------------------------------------------------
                # Compliance tracking
                # ------------------------------------------------

                if "compliance_status" in audit_df:

                    compliant_events = int(
                        (
                            audit_df[
                                "compliance_status"
                            ]
                            .astype(str)
                            .str.upper()
                            == "COMPLIANT"
                        ).sum()
                    )

                else:

                    compliant_events = 0

                compliance_rate = (
                    (
                        compliant_events
                        / total_events
                    ) * 100
                    if total_events > 0
                    else 0
                )

                # ------------------------------------------------
                # Governance KPI cards
                # ------------------------------------------------

                stat_cols_1 = st.columns(4)

                stat_cols_1[0].metric(
                    "Total Events",
                    total_events
                )

                stat_cols_1[1].metric(
                    "Approved",
                    approved_events
                )

                stat_cols_1[2].metric(
                    "Simulated",
                    simulated_events
                )

                stat_cols_1[3].metric(
                    "Compliant",
                    f"{compliance_rate:.1f}%"
                )

                stat_cols_2 = st.columns(3)

                stat_cols_2[0].metric(
                    "Total Resource",
                    f"{total_resource:.2f}"
                )

                stat_cols_2[1].metric(
                    "Total Cost",
                    f"₹{total_cost:,.2f}"
                )

                stat_cols_2[2].metric(
                    "Approved Interventions",
                    approved_events
                )
                # ------------------------------------------------
                # Detailed audit table
                # ------------------------------------------------

                st.markdown(
                    '<div class="ac-section">Recent intervention activity</div>',
                    unsafe_allow_html=True
                )

                display_columns = [
                    "timestamp",
                    "zone_id",
                    "intervention_type",
                    "approved_by",
                    "approval_status",
                    "execution_mode",
                    "resource_volume",
                    "estimated_cost",
                    "currency",
                    "compliance_status",
                    "human_approval_required",
                    "reason"
                ]

                available_columns = [
                    column
                    for column in display_columns
                    if column in audit_df.columns
                ]

                audit_display = audit_df[
                    available_columns
                ].copy()

                if "estimated_cost" in audit_display.columns:
                    audit_display["estimated_cost"] = pd.to_numeric(
                        audit_display["estimated_cost"],
                        errors="coerce"
                    ).fillna(0).round(2)

                if "resource_volume" in audit_display.columns:
                    audit_display["resource_volume"] = pd.to_numeric(
                        audit_display["resource_volume"],
                        errors="coerce"
                    ).fillna(0).round(2)

                st.dataframe(
                    audit_display,
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
