import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import io
import shap
import seaborn as sns
import json

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

st.set_page_config(page_title="Evoastra Telecom Churn Detector", layout="wide")

# ------------------------
# CONFIG
# ------------------------
MODELS_DIR = "models"
ARPU_DEFAULT = 200
RETENTION_COST = 50
TOP_PERCENT = 0.10

# ------------------------
# UTIL: load components cached
# ------------------------
@st.cache_resource
def load_deployment_components(models_dir=MODELS_DIR):
    model_path = os.path.join(models_dir, "churn_prediction_model.pkl")
    imputer_path = os.path.join(models_dir, "imputer.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    feature_names_path = os.path.join(models_dir, "feature_names.pkl")
    model, imputer, scaler, feature_names = None, None, None, None
    errors = []
    try:
        model = joblib.load(model_path)
    except Exception as e:
        errors.append(f"Model load error: {e}")
    try:
        imputer = joblib.load(imputer_path)
    except Exception as e:
        errors.append(f"Imputer load error: {e}")
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        errors.append(f"Scaler load error: {e}")
    try:
        feature_names = joblib.load(feature_names_path)
    except Exception as e:
        errors.append(
            f"Feature names load warning: {e} (feature alignment will use uploaded columns)"
        )
    return model, imputer, scaler, feature_names, errors


MODEL, IMPUTER, SCALER, FEATURE_NAMES, LOAD_ERRORS = load_deployment_components()

# ------------------------
# UI: header & instructions
# ------------------------
st.title("📞 Evoastra Telecom Churn Detector — Production Preview")
st.markdown(
    """
This app predicts churn risk for telecom operator-month rows and highlights **top strengths**:
- Top predictive features  
- Business impact (protected revenue, ROI) when targeting top risky subscribers  
- Provider revenue ranks and loss estimates  

Upload a CSV with the same columns used at training (`service_provider`, `circle`, `value`, etc.)
"""
)

for e in LOAD_ERRORS:
    st.sidebar.warning(e)
if MODEL is None:
    st.sidebar.error("Model not loaded. Place 'churn_prediction_model.pkl' in the models directory.")
else:
    st.sidebar.success("Model loaded ✓")

ARPU = st.sidebar.number_input("Monthly ARPU (₹)", value=ARPU_DEFAULT, step=10)
RET_COST = st.sidebar.number_input("Retention cost per customer (₹)", value=RETENTION_COST, step=5)
TOP_PCT = st.sidebar.slider("Top % at-risk targeted", 1, 50, int(TOP_PERCENT * 100)) / 100.0

# ------------------------
# File upload
# ------------------------
uploaded_file = st.file_uploader("📂 Upload Telecom Data CSV", type=["csv"])

def safe_reindex(df, feature_names):
    if feature_names is None:
        return df.copy()
    return df.reindex(columns=feature_names, fill_value=0)

def calculate_business_impact(y_true, y_pred_proba, arpu_monthly, retention_cost, top_pct):
    n = len(y_pred_proba)
    top_k = max(1, int(np.ceil(n * top_pct)))
    top_idx = np.argsort(y_pred_proba)[-top_k:]
    y_true_indexed = y_true.reset_index(drop=True)
    if len(y_true_indexed) < top_k:
        true_positives_in_target = int(y_true_indexed.sum())
    else:
        true_positives_in_target = int(y_true_indexed.iloc[top_idx].sum())

    protected_revenue = true_positives_in_target * arpu_monthly * 12
    campaign_cost = top_k * retention_cost
    net_benefit = protected_revenue - campaign_cost
    roi = (net_benefit / campaign_cost) * 100 if campaign_cost > 0 else 0

    return {
        "protected_revenue": protected_revenue,
        "campaign_cost": campaign_cost,
        "net_benefit": net_benefit,
        "roi_percentage": roi,
        "saved_customers": true_positives_in_target,
        "total_targeted": top_k,
    }

def filter_strength_columns(df):
    meaningful_cols = []
    for col in df.columns:
        if df[col].nunique() > 1:
            meaningful_cols.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]) and df[col].abs().sum() != 0:
            meaningful_cols.append(col)
    return df[meaningful_cols]

if uploaded_file is None:
    st.info("Upload a CSV to run predictions.")
    st.stop()

try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

df_raw = df_raw.fillna(0)
st.write("### 🔎 Uploaded data preview")
st.dataframe(filter_strength_columns(df_raw).head())

X_aligned = safe_reindex(df_raw, FEATURE_NAMES)

try:
    if IMPUTER and SCALER:
        X_imputed = IMPUTER.transform(X_aligned)
        X_scaled = SCALER.transform(X_imputed)
        X_processed = X_scaled
    else:
        X_processed = X_aligned.select_dtypes(include=np.number).values
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

try:
    predictions = MODEL.predict(X_processed)
    probabilities = MODEL.predict_proba(X_processed)[:, 1]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

df_out = df_raw.copy().reset_index(drop=True)
df_out["predicted_churn"] = predictions.astype(int)
df_out["churn_probability"] = probabilities

# ------------------------
# Key Outputs
# ------------------------
st.write("### ✅ Key Outputs")
col1, col2, col3 = st.columns(3)
if "churn_binary" in df_out.columns:
    y_true = df_out["churn_binary"].astype(int)
    y_pred = df_out["predicted_churn"].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    col1.metric("Accuracy", f"{accuracy:.2%}")
    col2.metric("F1-Score", f"{f1:.2%}")
    col3.metric("High-risk slice", f"{int(TOP_PCT * 100)}%")
else:
    col1.metric("Predictions", f"{len(df_out):,}")
    col2.metric("Avg churn risk", f"{df_out['churn_probability'].mean():.2%}")
    col3.metric("High-risk slice", f"{int(TOP_PCT * 100)}%")

# ------------------------
# Business Impact
# ------------------------
st.markdown("---")
st.write(f"### 💰 Business Impact — Targeting Top {int(TOP_PCT * 100)}%")

if "churn_binary" in df_out.columns:
    business_impact = calculate_business_impact(
        df_out["churn_binary"], df_out["churn_probability"].values, ARPU, RET_COST, TOP_PCT
    )
else:
    probs = df_out["churn_probability"].values
    n = len(probs)
    top_k = max(1, int(np.ceil(n * TOP_PCT)))
    top_indices = np.argsort(probs)[-top_k:]
    expected_saved_customers = df_out.loc[top_indices, "churn_probability"].sum()
    protected_revenue = expected_saved_customers * ARPU * 12
    campaign_cost = top_k * RET_COST
    net_benefit = protected_revenue - campaign_cost
    roi = (net_benefit / campaign_cost) * 100 if campaign_cost > 0 else 0
    business_impact = {
        "protected_revenue": protected_revenue,
        "campaign_cost": campaign_cost,
        "net_benefit": net_benefit,
        "roi_percentage": roi,
    }

colA, colB, colC = st.columns(3)
colA.metric("Protected Revenue", f"₹ {business_impact['protected_revenue']:,.0f}")
colB.metric("Net Benefit", f"₹ {business_impact['net_benefit']:,.0f}")
colC.metric("ROI", f"{business_impact['roi_percentage']:.2f}%")

st.markdown("---")

# ------------------------
# Model Comparison (Safe Load)
# ------------------------
comparison_df, summary = None, None
try:
    comparison_df = pd.read_csv("models/model_comparison.csv", index_col=0)
except FileNotFoundError:
    st.warning("⚠️ model_comparison.csv not found. Upload to see model performance comparison.")
except Exception as e:
    st.error(f"Error loading comparison CSV: {e}")

try:
    with open("models/best_model_summary.json", "r") as f:
        summary = json.load(f)
except FileNotFoundError:
    st.warning("⚠️ best_model_summary.json not found. Upload to see model summary.")
except Exception as e:
    st.error(f"Error loading best model summary: {e}")

if summary is not None:
    st.subheader("🏆 Best Model Summary")
    st.write(summary)

if comparison_df is not None:
    st.subheader("📈 Model Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

# ------------------------
# Predictions Download
# ------------------------
st.markdown("---")
csv_buffer = io.StringIO()
df_out.to_csv(csv_buffer, index=False)
st.download_button("📥 Download Predictions", csv_buffer.getvalue(), "predictions_with_churn.csv", "text/csv")

st.write("### 🔎 Preview")
st.dataframe(df_out.head())

st.info(
    "✅ Notes:\n- Handles missing files safely\n- Upload model comparison files for extended dashboard\n- Designed for Streamlit Cloud deployment"
)

