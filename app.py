import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure repo root is on sys.path
sys.path.insert(0, os.path.abspath("."))

from src.predict import load_artifacts, predict_patient_risk, DISCLAIMER_TEXT
from src.train_final_model import train_and_save_final_models


# Set Streamlit Page Config
st.set_page_config(
    page_title="CardioPulse AI — Ensemble Risk Intelligence",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom CSS for Health-Tech SaaS Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Background & Padding */
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

/* Header / Hero Styling */
.hero-container {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
    padding: 2.5rem 2rem;
    border-radius: 16px;
    color: #FFFFFF;
    margin-bottom: 2rem;
    box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
}

.hero-title {
    font-size: 2.25rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.hero-tagline {
    font-size: 1.05rem;
    color: #94A3B8;
    margin-top: 0.5rem;
    font-weight: 400;
}

.badge-cohort {
    display: inline-block;
    background-color: rgba(13, 148, 136, 0.25);
    color: #2DD4BF;
    border: 1px solid rgba(45, 212, 191, 0.3);
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-top: 1rem;
}

/* Card Styling */
.card-box {
    background-color: #FFFFFF;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #E2E8F0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    margin-bottom: 1.5rem;
}

.card-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1E293B;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    border-bottom: 1px solid #F1F5F9;
    padding-bottom: 0.5rem;
}

/* Custom Risk Badges */
.risk-pill {
    display: inline-flex;
    align-items: center;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}

.risk-low {
    background-color: #DCFCE7;
    color: #166534;
    border: 1px solid #BBF7D0;
}

.risk-moderate {
    background-color: #FEF3C7;
    color: #92400E;
    border: 1px solid #FDE68A;
}

.risk-high {
    background-color: #FEE2E2;
    color: #991B1B;
    border: 1px solid #FECACA;
}

/* Agreement Flags */
.flag-banner {
    padding: 0.85rem 1.2rem;
    border-radius: 8px;
    font-size: 0.92rem;
    font-weight: 500;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.flag-high {
    background-color: #F0FDF4;
    color: #166534;
    border-left: 4px solid #16A34A;
}

.flag-moderate {
    background-color: #FFFBEB;
    color: #B45309;
    border-left: 4px solid #F59E0B;
}

.flag-low {
    background-color: #FEF2F2;
    color: #B91C1C;
    border-left: 4px solid #EF4444;
}

/* Disclaimer Banner */
.disclaimer-banner {
    background-color: #F1F5F9;
    border: 1px solid #CBD5E1;
    color: #475569;
    padding: 0.9rem 1.25rem;
    border-radius: 8px;
    font-size: 0.875rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-top: 1.5rem;
}

/* Button Customization */
div.stButton > button {
    background: linear-gradient(135deg, #0D9488 0%, #0F766E 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 4px 6px -1px rgba(13, 148, 136, 0.25) !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #0F766E 0%, #115E59 100%) !important;
    box-shadow: 0 6px 12px -2px rgba(13, 148, 136, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Custom Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 2px solid #E2E8F0;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.95rem;
    color: #64748B;
    padding: 0.75rem 1.25rem;
    border-radius: 8px 8px 0 0;
}

.stTabs [aria-selected="true"] {
    color: #0D9488 !important;
    background-color: #FFFFFF !important;
    border-bottom: 2px solid #0D9488 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Auto-Load or Auto-Train Model
@st.cache_resource
def get_or_train_models():
    models_dir = "models"
    required_files = [
        "preprocessor.joblib", "random_forest.joblib",
        "xgboost.joblib", "adaboost.joblib", "ensemble.joblib"
    ]
    all_exist = all(os.path.exists(os.path.join(models_dir, f)) for f in required_files)

    if not all_exist:
        with st.spinner("Setting up the model for first use... (training on full 920-patient cohort)"):
            train_and_save_final_models(models_dir=models_dir)

    return load_artifacts(models_dir=models_dir)


# Load artifacts
try:
    artifacts = get_or_train_models()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# Hero Banner Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🫀 CardioPulse AI</span>
    </div>
    <div class="hero-tagline">
        Reliable, Explainable Ensemble Intelligence for Clinical Heart Disease Risk Assessment
    </div>
    <div class="badge-cohort">
        Combined 4-Site Cohort • N = 920 Patients • Soft-Voting Ensemble (RF + XGBoost + AdaBoost)
    </div>
</div>
""", unsafe_allow_html=True)


# Main Tabs
tab1, tab2 = st.tabs(["🩺 Risk Assessment", "ℹ️ About This Project"])


# Categorical Mapping Handlers
SEX_MAP = {"Female": 0.0, "Male": 1.0}
CP_MAP = {
    "Typical Angina (1)": 1.0,
    "Atypical Angina (2)": 2.0,
    "Non-Anginal Pain (3)": 3.0,
    "Asymptomatic (4)": 4.0
}
FBS_MAP = {"False (≤ 120 mg/dL)": 0.0, "True (> 120 mg/dL)": 1.0}
RESTECG_MAP = {
    "Normal (0)": 0.0,
    "ST-T Wave Abnormality (1)": 1.0,
    "Left Ventricular Hypertrophy (2)": 2.0
}
EXANG_MAP = {"No": 0.0, "Yes": 1.0}
SLOPE_MAP = {
    "Upsloping (1)": 1.0,
    "Flat (2)": 2.0,
    "Downsloping (3)": 3.0
}
CA_MAP = {"0 Vessels": 0.0, "1 Vessel": 1.0, "2 Vessels": 2.0, "3 Vessels": 3.0}
THAL_MAP = {
    "Normal (3)": 3.0,
    "Fixed Defect (6)": 6.0,
    "Reversable Defect (7)": 7.0
}


# TAB 1: RISK ASSESSMENT
with tab1:
    st.markdown("### Enter Patient Physiological Profile")
    st.caption("Provide clinical findings across demographic, vitals, and diagnostic cardiac test parameters.")

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)

        # Card 1: Demographics & History
        with col1:
            st.markdown("""
            <div class="card-box">
                <div class="card-title">👤 Demographics & History</div>
            """, unsafe_allow_html=True)

            age = st.slider("Age (Years)", min_value=20, max_value=90, value=58, step=1)
            sex_label = st.radio("Sex", list(SEX_MAP.keys()), index=1, horizontal=True)
            fbs_label = st.selectbox("Fasting Blood Sugar > 120 mg/dL", list(FBS_MAP.keys()), index=0)

            st.markdown("</div>", unsafe_allow_html=True)

        # Card 2: Vitals & Lab Measurements
        with col2:
            st.markdown("""
            <div class="card-box">
                <div class="card-title">🩺 Vitals & Laboratory</div>
            """, unsafe_allow_html=True)

            trestbps = st.slider("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=135, step=1)
            chol = st.slider("Serum Cholesterol (mg/dL)", min_value=100, max_value=600, value=240, step=1)
            thalach = st.slider("Max Heart Rate Achieved (bpm)", min_value=60, max_value=220, value=145, step=1)

            st.markdown("</div>", unsafe_allow_html=True)

        # Card 3: Cardiac Diagnostic Tests
        with col3:
            st.markdown("""
            <div class="card-box">
                <div class="card-title">⚡ Cardiac Test Results</div>
            """, unsafe_allow_html=True)

            cp_label = st.selectbox("Chest Pain Type", list(CP_MAP.keys()), index=3)
            exang_label = st.radio("Exercise-Induced Angina", list(EXANG_MAP.keys()), index=1, horizontal=True)
            oldpeak = st.slider("ST Depression (oldpeak)", min_value=0.0, max_value=7.0, value=1.5, step=0.1)

            st.markdown("</div>", unsafe_allow_html=True)

        # Additional Diagnostic Parameters Expandable Grid
        with st.expander("🔬 Additional Diagnostic Parameters (ECG, Vessel & Thalassemia Scans)", expanded=True):
            ecg_col1, ecg_col2, ecg_col3 = st.columns(3)
            with ecg_col1:
                restecg_label = st.selectbox("Resting ECG Results", list(RESTECG_MAP.keys()), index=0)
            with ecg_col2:
                slope_label = st.selectbox("Peak Exercise ST Slope", list(SLOPE_MAP.keys()), index=1)
            with ecg_col3:
                ca_label = st.selectbox("Major Vessels Colored (ca)", list(CA_MAP.keys()), index=0)
                thal_label = st.selectbox("Thalassemia (thal)", list(THAL_MAP.keys()), index=2)

        submit_btn = st.form_submit_button("🔍 Analyze Heart Disease Risk", use_container_width=True)

    # Process Form Submission
    if submit_btn or "has_run" in st.session_state:
        st.session_state["has_run"] = True

        patient_input = {
            "age": float(age),
            "sex": SEX_MAP[sex_label],
            "cp": CP_MAP[cp_label],
            "trestbps": float(trestbps),
            "chol": float(chol),
            "fbs": FBS_MAP[fbs_label],
            "restecg": RESTECG_MAP[restecg_label],
            "thalach": float(thalach),
            "exang": EXANG_MAP[exang_label],
            "oldpeak": float(oldpeak),
            "slope": SLOPE_MAP[slope_label],
            "ca": CA_MAP[ca_label],
            "thal": THAL_MAP[thal_label]
        }

        # Run Prediction using src/predict.py logic
        res = predict_patient_risk(patient_input, models_dir="models")

        prob = res["predicted_probability"]
        prob_pct = res["predicted_probability_percent"]
        risk_label = res["risk_label"]
        agreement_flag = res["agreement_flag"]
        base_probs = res["base_model_probabilities"]
        top_features = res["top_contributing_features"]

        st.markdown("---")
        st.markdown("### 📊 Assessment Results")

        res_col1, res_col2 = st.columns([1, 1.2])

        # Column 1: Probability Gauge & Risk Badge
        with res_col1:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🎯 Predicted Risk Level</div>', unsafe_allow_html=True)

            # Circular Gauge Plotly
            gauge_color = "#16A34A" if prob < 0.35 else ("#D97706" if prob < 0.65 else "#DC2626")

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={'suffix': "%", 'font': {'size': 44, 'color': '#0F172A', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#CBD5E1"},
                    'bar': {'color': gauge_color, 'thickness': 0.3},
                    'bgcolor': "#F1F5F9",
                    'bordercolor': "#E2E8F0",
                    'steps': [
                        {'range': [0, 35], 'color': '#DCFCE7'},
                        {'range': [35, 65], 'color': '#FEF3C7'},
                        {'range': [65, 100], 'color': '#FEE2E2'}
                    ]
                }
            ))

            fig_gauge.update_layout(
                height=240,
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'family': "Inter"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

            # Risk Label Badge Pill
            pill_class = "risk-low" if prob < 0.35 else ("risk-moderate" if prob < 0.65 else "risk-high")
            st.markdown(f"""
            <div style="text-align: center; margin-top: -10px; margin-bottom: 10px;">
                <span class="risk-pill {pill_class}">Risk Classification: {risk_label.upper()}</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # Column 2: Model Consensus & Agreement Breakdown
        with res_col2:
            st.markdown('<div class="card-box">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">🤝 Model Consensus & Uncertainty</div>', unsafe_allow_html=True)

            # Agreement Banner
            if "high agreement" in agreement_flag:
                flag_class = "flag-high"
            elif "moderate agreement" in agreement_flag:
                flag_class = "flag-moderate"
            else:
                flag_class = "flag-low"

            st.markdown(f"""
            <div class="flag-banner {flag_class}">
                <span>ℹ️</span> {agreement_flag}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Base Classifiers Probability Comparison:**")

            # Base models comparison chart
            df_base = pd.DataFrame([
                {"Model": m, "Probability": p * 100} for m, p in base_probs.items()
            ])

            fig_base = px.bar(
                df_base,
                x="Probability",
                y="Model",
                orientation="h",
                text=df_base["Probability"].apply(lambda val: f"{val:.1f}%"),
                color="Model",
                color_discrete_map={
                    "Random Forest": "#0D9488",
                    "XGBoost": "#0284C7",
                    "AdaBoost": "#6366F1"
                }
            )

            fig_base.update_layout(
                height=180,
                xaxis=dict(range=[0, 100], title="Probability (%)", gridcolor="#F1F5F9"),
                yaxis=dict(title=""),
                showlegend=False,
                margin=dict(l=0, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'family': "Inter"}
            )
            fig_base.update_traces(textposition="outside")

            st.plotly_chart(fig_base, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

        # SHAP Feature Explanations Card
        st.markdown('<div class="card-box">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔍 SHAP Feature Impact Breakdown</div>', unsafe_allow_html=True)

        if top_features:
            df_shap = pd.DataFrame(top_features)
            df_shap["color"] = df_shap["shap_value"].apply(lambda v: "#DC2626" if v > 0 else "#16A34A")

            fig_shap = px.bar(
                df_shap,
                x="shap_value",
                y="feature",
                orientation="h",
                text=df_shap["summary"],
                color="color",
                color_discrete_map="identity"
            )

            fig_shap.update_layout(
                height=220,
                xaxis=dict(title="SHAP Impact Value (Direction & Magnitude)", gridcolor="#F1F5F9"),
                yaxis=dict(title="", autorange="reversed"),
                showlegend=False,
                margin=dict(l=0, r=20, t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={'family': "Inter"}
            )
            fig_shap.update_traces(textposition="outside")

            st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("SHAP feature explanations unavailable for this input.")

        st.markdown("</div>", unsafe_allow_html=True)

        # Disclaimer Info Banner
        st.markdown(f"""
        <div class="disclaimer-banner">
            <span style="font-size: 1.25rem;">⚠️</span>
            <span><strong>Clinical Disclaimer:</strong> {DISCLAIMER_TEXT}</span>
        </div>
        """, unsafe_allow_html=True)


# TAB 2: ABOUT THIS PROJECT
with tab2:
    st.markdown("""
    <div class="card-box">
        <div class="card-title">📌 Project Overview</div>
        <p style="color: #475569; font-size: 0.98rem; line-height: 1.6;">
            <strong>CardioPulse AI</strong> expands upon the foundational research by <em>Mienye & Jere (2024)</em>.
            While the original study evaluated ensemble classifiers on single-site datasets, this framework introduces
            a reliable, multi-site architecture trained on the <strong>combined 920-patient UCI cohort</strong>
            (Cleveland, Hungarian, Switzerland, VA Long Beach).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🚀 Key Architectural Additions Beyond Reference Paper")

    additions = [
        ("🔒", "Nested Cross-Validation", "Leak-free 5-fold outer / 3-fold inner cross-validation for unbiased evaluation."),
        ("🎯", "Out-of-Sample Calibration", "Platt Scaling & Isotonic Regression fit on out-of-sample predictions."),
        ("⚖️", "Subgroup Fairness", "Evaluated across sex, age bands, and multi-site source cohorts."),
        ("🤝", "Per-Patient Model Agreement", "Uncertainty flagging via standard deviation of base probabilities."),
        ("🌐", "Cross-Dataset Validation", "External generalizability evaluation on Framingham Heart Study (N=4,240)."),
        ("🔬", "Explanation Depth", "LIME vs SHAP comparison, counterfactual scenarios, and ranking stability.")
    ]

    add_col1, add_col2 = st.columns(2)
    for idx, (icon, title, desc) in enumerate(additions):
        col_target = add_col1 if idx % 2 == 0 else add_col2
        with col_target:
            st.markdown(f"""
            <div class="card-box" style="margin-bottom: 1rem;">
                <div style="font-weight: 600; font-size: 1rem; color: #0F172A; display: flex; align-items: center; gap: 0.5rem;">
                    <span>{icon}</span> {title}
                </div>
                <div style="color: #64748B; font-size: 0.88rem; margin-top: 0.35rem;">
                    {desc}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Dynamic Model Metadata Section
    metadata_path = "models/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta = json.load(f)

        st.markdown("### 📊 Model Training Metadata & Optuna Validation ROC-AUC")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Training Cohort Size", f"{meta.get('dataset_size', 920)} Patients")
        with m_col2:
            st.metric("Feature Schema", f"{meta.get('processed_feature_count', 13)} Attributes")
        with m_col3:
            st.metric("Random Seed", meta.get("random_state", 42))

        scores = meta.get("optuna_validation_roc_auc_scores", {})
        if scores:
            df_scores = pd.DataFrame([
                {"Model": m.replace("_", " ").title(), "Optuna Inner-CV ROC-AUC": f"{score:.4f}"}
                for m, score in scores.items()
            ])
            st.table(df_scores)
