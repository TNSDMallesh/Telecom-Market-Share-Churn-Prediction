# Streamlit dashboard for Telecom Churn Prediction Deployment
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

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(page_title="Evoastra Telecom Churn Detector", layout="wide")

# ------------------------
# CONFIG
# ------------------------
MODELS_DIR = "models"  # adjust if different
ARPU_DEFAULT = 200  # default monthly ARPU for revenue calculations (can be adjusted in UI)
RETENTION_COST = 50  # cost per targeted customer for retention campaign (monthly)
TOP_PERCENT = 0.10  # top 10% targeted by risk

# ------------------------
# UTIL: load components cached
# ------------------------
@st.cache_resource
def load_deployment_components(models_dir=MODELS_DIR):
    """Load model, imputer, scaler, feature_names from disk (joblib)."""
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
        errors.append(f"Feature names load warning: {e} (feature alignment will try to use uploaded columns)")
    return model, imputer, scaler, feature_names, errors

MODEL, IMPUTER, SCALER, FEATURE_NAMES, LOAD_ERRORS = load_deployment_components()

# ------------------------
# UI: header & instructions
# ------------------------
st.title("📞 Evoastra Telecom Churn Detector — Production Preview")
st.markdown(
    """
This app predicts churn risk for telecom operator-month rows, highlights the *top strengths*:
- Top 20 predictive features
- Business impact (protected revenue, ROI) when targeting top 10% most at-risk subscribers
- Service provider revenue ranks and which providers are losing revenue
Upload a CSV with the same raw/engineered columns used at training (or include primary identifiers like service_provider, circle, value).
"""
)

# show any load errors
if any(LOAD_ERRORS):
    for e in LOAD_ERRORS:
        st.sidebar.warning(e)
if MODEL is None:
    st.sidebar.error("Model not loaded. Place 'churn_prediction_model.pkl' in the models directory.")
else:
    st.sidebar.success("Model loaded ✓")

# ARPU controls
st.sidebar.header("Business params (optional)")
ARPU = st.sidebar.number_input("Estimated monthly ARPU (₹)", value=ARPU_DEFAULT, step=10)
RET_COST = st.sidebar.number_input("Retention cost per targeted customer (₹)", value=RETENTION_COST, step=5)
TOP_PCT = st.sidebar.slider("Target top percentage of risk", min_value=1, max_value=50, value=int(TOP_PERCENT*100)) / 100.0

# ------------------------
# File upload
# ------------------------
uploaded_file = st.file_uploader("📂 Upload Telecom Data CSV (rows = provider x month)", type=["csv"])

def safe_reindex(df, feature_names):
    """Reindex df to feature_names. If feature_names is None, just return df."""
    if feature_names is None:
        return df.copy()
    # ensure any missing columns added with 0
    out = df.reindex(columns=feature_names, fill_value=0)
    return out

def calculate_business_impact(y_true, y_pred_proba, arpu_monthly=ARPU, retention_cost=RET_COST, top_pct=TOP_PCT):
    """Return dict of business impact metrics for top pct targeted customers, based on ground truth."""
    n = len(y_pred_proba)
    top_k = max(1, int(np.ceil(n * top_pct)))
    top_idx = np.argsort(y_pred_proba)[-top_k:]
    y_true_indexed = y_true.reset_index(drop=True) if isinstance(y_true, pd.Series) else pd.Series(y_true).reset_index(drop=True)
    
    # Ensure y_true_indexed has enough elements before trying to index
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
    """
    Keep only columns that contain meaningful variation (not constant or all zeros)
    This meets the requirement: "show only columns which have data; don't show the weakness".
    """
    meaningful_cols = []
    for col in df.columns:
        if df[col].nunique() > 1:
            meaningful_cols.append(col)
        else:
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].abs().sum() != 0:
                meaningful_cols.append(col)
    return df[meaningful_cols]

if uploaded_file is None:
    st.info("Upload a CSV to run predictions. Expected columns include engineered features used at training (or at least 'service_provider','circle','value').")
    st.stop()

# read csv
try:
    df_raw = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Failed to read uploaded file: {e}")
    st.stop()

# drop 'notes' if present
if 'notes' in df_raw.columns:
    df_raw = df_raw.drop(columns=['notes'])

# Replace NaNs with 0
df_raw = df_raw.fillna(0)

st.write("### 🔎 Uploaded data preview (showing only meaningful columns)")
df_preview = filter_strength_columns(df_raw)
st.dataframe(df_preview.head(5))

# Align features for model input
X_aligned = safe_reindex(df_raw, FEATURE_NAMES)

# Preprocess
X_processed = None
preproc_error = None
try:
    if IMPUTER is not None and SCALER is not None:
        X_imputed = IMPUTER.transform(X_aligned)
        X_scaled = SCALER.transform(X_imputed)
        X_processed = X_scaled
    else: # Fallback if preprocessors are missing
        X_processed = X_aligned.select_dtypes(include=np.number).values
except Exception as e:
    preproc_error = e

if X_processed is None:
    st.error(f"Could not preprocess the uploaded file for prediction. Error: {preproc_error}")
    st.stop()

# model predict
try:
    predictions = MODEL.predict(X_processed)
    probabilities = MODEL.predict_proba(X_processed)[:, 1]
except Exception as e:
    st.error(f"Model prediction failed: {e}")
    st.stop()

# attach predictions
df_out = df_raw.copy().reset_index(drop=True)
df_out['predicted_churn'] = predictions.astype(int)
df_out['churn_probability'] = probabilities

# Show top-level metrics
st.write("### ✅ Key Outputs — Strengths of the Project")

col1, col2, col3 = st.columns(3)
if 'churn_binary' in df_out.columns:
    y_true = df_out['churn_binary'].astype(int)
    y_pred = df_out['predicted_churn'].astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
    with col2:
        st.metric("F1-Score", f"{f1:.2%}")
    with col3:
        st.metric("High-risk slice (top %)", f"{int(TOP_PCT*100)}%")
    st.write("#### Classification Report:")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    st.dataframe(df_report.style.format("{:.2f}"))
else:
    with col1:
        st.metric("Predictions", f"{len(df_out):,} rows")
    with col2:
        st.metric("Average predicted churn risk", f"{df_out['churn_probability'].mean():.2%}")
    with col3:
        st.metric("High-risk slice (top %)", f"{int(TOP_PCT*100)}%")

# --------------------------------------------------------------------------
# FIXED: Business Impact Section
# --------------------------------------------------------------------------
st.markdown("---")
st.write(f"### 💰 Business Impact — Targeting Top {int(TOP_PCT*100)}% Risk")

# Calculate business impact based on whether ground truth is available
if 'churn_binary' in df_out.columns:
    # Scenario 1: Ground truth exists. Calculate impact using actual churn data.
    business_impact = calculate_business_impact(
        df_out['churn_binary'], 
        df_out['churn_probability'].values, 
        arpu_monthly=ARPU, 
        retention_cost=RET_COST, 
        top_pct=TOP_PCT
    )
    impact_subtitle = f"Based on *actual* churners identified in the targeted group ({business_impact['saved_customers']} customers)."

else:
    pass
    # Scenario 2: No ground truth. Estimate impact using churn probabilities.
    # This simulates a real-world scenario where we act on predictions.
    probs = df_out['churn_probability'].values
    n = len(probs)
    top_k = max(1, int(np.ceil(n * TOP_PCT)))
    top_indices = np.argsort(probs)[-top_k:]
    
    # *FIX*: Instead of using binary predictions, sum probabilities in the top slice
    # to get the expected number of churners we can save. This is more robust.
    expected_saved_customers = df_out.loc[top_indices, 'churn_probability'].sum()
    
    protected_revenue = expected_saved_customers * ARPU * 12
    campaign_cost = top_k * RET_COST
    net_benefit = protected_revenue - campaign_cost
    roi = (net_benefit / campaign_cost) * 100 if campaign_cost > 0 else 0
    
    business_impact = {
        "protected_revenue": protected_revenue,
        "campaign_cost": campaign_cost,
        "net_benefit": net_benefit,
        "roi_percentage": roi
    }
    impact_subtitle = f"Based on an *estimated* {expected_saved_customers:.1f} churners in the targeted group."

# Display the calculated business impact metrics
colA, colB, colC = st.columns(3)
colA.metric("Protected Annual Revenue", f"₹ {business_impact['protected_revenue']:,.0f}", delta=f"Cost: ₹ {business_impact['campaign_cost']:,.0f}", 
            help="Estimated annual revenue retained from customers who were predicted to churn and were targeted.")
colB.metric("Net Benefit (Annual)", f"₹ {business_impact['net_benefit']:,.0f}", 
            help="Protected Revenue minus" \
            "" \
            " the cost of the retention campaign.")
colC.metric("ROI", f"{business_impact['roi_percentage']:.2f}%", 
            help="Return on Investment from the retention campaign. (Net Benefit / Campaign Cost).")

st.caption(f"📈 Campaign Cost: *₹{business_impact['campaign_cost']:,.0f}*. {impact_subtitle}")

st.markdown("---")
# ------------------------
# Model Comparison (Safe Load)
# ------------------------
comparison_df, summary = None, None
try:
    comparison_df = pd.read_csv("models/model_comparison.csv", index_col=0)
except FileNotFoundError:
    st.warning("⚠ model_comparison.csv not found. Upload to see model performance comparison.")
except Exception as e:
    st.error(f"Error loading comparison CSV: {e}")

try:
    with open("models/best_model_summary.json", "r") as f:
        summary = json.load(f)
except FileNotFoundError:
    st.warning("⚠ best_model_summary.json not found. Upload to see model summary.")
except Exception as e:
    st.error(f"Error loading best model summary: {e}")

if summary is not None:
    st.subheader("🏆 Best Model Summary")
    st.write(summary)

if comparison_df is not None:
    st.subheader("📈 Model Comparison")
    st.dataframe(comparison_df.style.highlight_max(axis=0, color="lightgreen"))

st.markdown("---")
# ------------------------
# Predictions Download
# ------------------------
csv_buffer = io.StringIO()
df_out.to_csv(csv_buffer, index=False)
st.download_button("📥 Download Predictions", csv_buffer.getvalue(), "predictions_with_churn.csv", "text/csv")

st.write("### 🔎 Preview")
st.dataframe(df_out.head())

st.info(
    "✅ Notes:\n- Handles missing files safely\n- Upload model comparison files for extended dashboard\n- Designed for Streamlit Cloud deployment"
)

# Top 20 features (from model or SHAP)
st.write("### 📈 Top 20 Model Features (what drives churn predictions)")
top20_df = None
try:
    if hasattr(MODEL, "feature_importances_") and FEATURE_NAMES is not None:
        fi = MODEL.feature_importances_
        feat = FEATURE_NAMES
        if len(fi) == len(feat):
            top_idx = np.argsort(fi)[-20:][::-1]
            top20_df = pd.DataFrame({
                "feature": [feat[i] for i in top_idx],
                "importance": fi[top_idx]
            })
    # fallback to SHAP if available and top20_df is None
    if top20_df is None:
        try:
            expl = shap.Explainer(MODEL, X_aligned.sample(n=min(500, len(X_aligned)), random_state=42))
            sv = expl(X_aligned.sample(n=min(500, len(X_aligned)), random_state=42))
            mean_imp = np.abs(sv.values).mean(axis=0)
            # need names
            names = X_aligned.columns.tolist()
            idx = np.argsort(mean_imp)[-20:][::-1]
            top20_df = pd.DataFrame({"feature": [names[i] for i in idx], "importance": mean_imp[idx]})
        except Exception:
            # last fallback: use df_out columns variance
            cols = X_aligned.columns
            imp = np.var(X_aligned, axis=0)
            idx = np.argsort(imp)[-20:][::-1]
            top20_df = pd.DataFrame({"feature": [cols[i] for i in idx], "importance": imp[idx]})
except Exception as e:
    st.warning(f"Could not compute top features exactly: {e}")
    top20_df = None

if top20_df is not None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x="importance", y="feature", data=top20_df, ax=ax)
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Top 20 Features (model importance)")
    st.pyplot(fig)
    st.dataframe(top20_df.reset_index(drop=True))
else:
    st.info("Top features not available for this model.")

# Provider-level revenue & churn loss analysis
st.markdown("---")
st.write("### 🏦 Service Provider Revenue & Risk Analysis (Top to Bottom)")

# ensure 'value' column exists - using 'value' as subscriber count as in main code
if 'value' not in df_out.columns:
    st.warning("No 'value' column found in uploaded file; revenue calculations require subscriber counts in 'value'. Showing provider-level predicted churn counts instead.")
    # compute predicted churners counts only
    provider_group = df_out.groupby('service_provider').agg(
        total_rows=('predicted_churn','size'),
        predicted_churners=('predicted_churn','sum'),
        avg_churn_risk=('churn_probability','mean')
    ).reset_index().sort_values('predicted_churners', ascending=False)
    st.dataframe(provider_group)
else:
    # compute provider revenue (monthly) as sum(value) * ARPU scaling? 
    # We will interpret 'value' as subscriber count per row
    provider_group = df_out.groupby('service_provider').agg(
        total_subscribers=('value','sum'),
        predicted_churners=('predicted_churn','sum'),
        avg_churn_risk=('churn_probability','mean'),
        rows=('predicted_churn','size')
    ).reset_index()
    provider_group['estimated_annual_revenue'] = provider_group['total_subscribers'] * ARPU * 12
    provider_group['estimated_annual_loss'] = provider_group['predicted_churners'] * ARPU * 12
    provider_group = provider_group.sort_values('estimated_annual_revenue', ascending=False)
    st.write("#### Providers sorted by Estimated Annual Revenue (Top → Bottom)")
    st.dataframe(provider_group[['service_provider','total_subscribers','estimated_annual_revenue','predicted_churners','estimated_annual_loss','avg_churn_risk']].head(50).style.format({
        'total_subscribers':'{:,}',
        'estimated_annual_revenue':'₹ {:,.0f}',
        'estimated_annual_loss':'₹ {:,.0f}',
        'avg_churn_risk':'{:.2%}'
    }))

    st.write("#### Providers losing revenue (sorted by estimated annual loss)")
    loss_sorted = provider_group.sort_values('estimated_annual_loss', ascending=False)
    st.dataframe(loss_sorted[['service_provider','predicted_churners','estimated_annual_loss','avg_churn_risk']].head(50).style.format({
        'estimated_annual_loss':'₹ {:,.0f}',
        'avg_churn_risk':'{:.2%}'
    }))

# Provide downloadable predictions CSV
st.markdown("---")
st.write("### ✅ Download Predictions")
csv_buffer = io.StringIO()
df_out.to_csv(csv_buffer, index=False)
st.download_button("📥 Download predictions as CSV", data=csv_buffer.getvalue(), file_name="predictions_with_churn.csv", mime="text/csv")

st.write("### 🔎 Quick Preview of predictions")
st.dataframe(df_out.head(10))

# Load saved data
comparison_df = pd.read_csv("saved_models_and_results/model_comparison.csv", index_col=0)

with open("saved_models_and_results/best_model_summary.json", "r") as f:
    summary = json.load(f)

st.title("📊 Model Performance Dashboard")

st.subheader("🏆 Best Model Summary")
st.write(summary)

st.subheader("📈 Detailed Comparison of All Models")
st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'))

st.title("📊 Telecom Subscriber Churn Analysis Dashboard")

st.markdown("""
Welcome to the Telecom Dashboard!  
Upload your dataset and explore which operator loses more subscribers, view model results, 
and discover top influencing features.
""")

# --- Your Note Section ---
st.markdown("### Note:")
st.write("This dashboard analyzes telecom churn data...")

# Churn distribution
try:
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    if 'churn_binary' in df_out.columns:
        counts = df_out['churn_binary'].value_counts().sort_index()
        labels = [f"Not churn ({counts.index[0]})", f"Churn ({counts.index[1]})"] if len(counts)>1 else [f"{counts.index[0]}"]
        ax[0].pie(counts, labels=labels, autopct='%1.1f%%', startangle=140)
        ax[0].set_title("Churn share")
    else:
        counts = df_out['predicted_churn'].value_counts().sort_index()
        ax[0].pie(counts, labels=[f"Not churn","Churn"][:len(counts)], autopct='%1.1f%%', startangle=140)
        ax[0].set_title("Predicted churn share")

    sns.countplot(x='predicted_churn', data=df_out, ax=ax[1])
    ax[1].set_title("Count by predicted churn")
    ax[1].set_xlabel("Predicted churn (0=no, 1=yes)")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Churn distribution plot failed: {e}")

# Probability histogram
try:
    fig, ax = plt.subplots(figsize=(8,4))
    sns.histplot(df_out['churn_probability'], bins=30, kde=True, ax=ax)
    ax.set_title("Churn probability distribution")
    ax.set_xlabel("Predicted churn probability")
    st.pyplot(fig)
except Exception as e:
    st.warning(f"Prob histogram failed: {e}")

# ROC curve
from sklearn.metrics import roc_curve, auc
try:
    if 'churn_binary' in df_out.columns:
        fpr, tpr, _ = roc_curve(df_out['churn_binary'], df_out['churn_probability'])
        roc_auc = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0,1],[0,1],'--', linewidth=0.7)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc='lower right')
        st.pyplot(fig)
except Exception as e:
    st.warning(f"ROC curve failed: {e}")

# Precision-Recall
from sklearn.metrics import precision_recall_curve, average_precision_score
try:
    if 'churn_binary' in df_out.columns:
        precision, recall, _ = precision_recall_curve(df_out['churn_binary'], df_out['churn_probability'])
        ap = average_precision_score(df_out['churn_binary'], df_out['churn_probability'])
        fig, ax = plt.subplots(figsize=(6,5))
        ax.plot(recall, precision, label=f'AP = {ap:.3f}')
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve")
        ax.legend()
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Precision-Recall plot failed: {e}")

# Confusion matrix
try:
    if 'churn_binary' in df_out.columns:
        cm = confusion_matrix(df_out['churn_binary'], df_out['predicted_churn'])
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Confusion matrix failed: {e}")


# Lift / cumulative gains (simple)
try:
    if 'churn_binary' in df_out.columns:
        df_lift = df_out[['churn_probability','churn_binary']].sort_values('churn_probability', ascending=False).reset_index(drop=True)
        df_lift['cum_churn'] = df_lift['churn_binary'].cumsum()
        df_lift['pct_customers'] = (df_lift.index + 1) / len(df_lift)
        df_lift['cum_churn_rate'] = df_lift['cum_churn'] / df_lift['churn_binary'].sum()
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(df_lift['pct_customers'], df_lift['cum_churn_rate'], label='Model (cumulative)')
        ax.plot([0,1],[0,1], '--', label='Random')
        ax.set_xlabel("Fraction of customers targeted")
        ax.set_ylabel("Fraction of total churn captured")
        ax.set_title("Cumulative gains / lift")
        ax.legend()
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Lift chart failed: {e}")

# Correlation heatmap (show only numeric meaningful columns)
try:
    num = df_out.select_dtypes(include=[np.number]).drop(columns=['predicted_churn','churn_probability'], errors='ignore')
    num = num.loc[:, num.apply(pd.Series.nunique) > 1]  # meaningful cols
    if num.shape[1] > 1:
        corr = num.corr()
        fig, ax = plt.subplots(figsize=(10,8))
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False, ax=ax)
        ax.set_title("Feature correlation (numeric cols)")
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Correlation heatmap failed: {e}")

# SHAP summary (requires shap and sample size guard)
try:
    if 'churn_probability' in df_out.columns and 'service_provider' in df_out.columns:
        X_for_shap = X_aligned if 'X_aligned' in globals() else df_out.select_dtypes(include=[np.number]).drop(columns=['predicted_churn','churn_probability'], errors='ignore')
    else:
        X_for_shap = None

    if X_for_shap is not None and 'MODEL' in globals():
        sample = X_for_shap.sample(n=min(500, len(X_for_shap)), random_state=42)
        expl = shap.Explainer(MODEL, sample)  # may raise for some models
        shap_values = expl(sample)
        st.write("#### SHAP summary (sample)")
        plt.figure(figsize=(8,5))
        shap.summary_plot(shap_values, sample, show=False)
        st.pyplot(plt.gcf())
        plt.clf()
except Exception as e:
    st.warning(f"SHAP summary not available: {e}")

# Provider by churners & risk
try:
    if 'service_provider' in df_out.columns:
        prov = df_out.groupby('service_provider').agg(
            predicted_churners=('predicted_churn','sum'),
            avg_risk=('churn_probability','mean'),
            total_rows=('predicted_churn','size')
        ).reset_index().sort_values('predicted_churners', ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='predicted_churners', y='service_provider', data=prov, ax=ax)
        ax.set_title("Top 20 providers by predicted churners")
        st.pyplot(fig)
        st.dataframe(prov)
except Exception as e:
    st.warning(f"Provider churn plot failed: {e}")

# Value vs risk scatter
try:
    if 'value' in df_out.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.scatterplot(x='value', y='churn_probability', data=df_out, alpha=0.6)
        ax.set_xlabel("Subscriber count / value")
        ax.set_ylabel("Churn probability")
        ax.set_title("Subscriber value vs predicted risk")
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Value vs risk plot failed: {e}")

# Time series churn trend
try:
    if 'month' in df_out.columns or 'date' in df_out.columns:
        date_col = 'month' if 'month' in df_out.columns else 'date'
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors='coerce')
        ts = df_out.dropna(subset=[date_col]).set_index(date_col).resample('M').agg({
            'predicted_churn':'sum',
            'churn_probability':'mean'
        })
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(ts.index, ts['predicted_churn'], marker='o', label='Predicted churners (count)')
        ax.set_ylabel("Count")
        ax2 = ax.twinx()
        ax2.plot(ts.index, ts['churn_probability'], color='orange', linestyle='--', marker='x', label='Average risk')
        ax2.set_ylabel("Average predicted risk")
        ax.set_title("Monthly churners & average risk")
        st.pyplot(fig)
except Exception as e:
    st.warning(f"Time series plot failed: {e}")

# Top N risky customers for action
try:
    top_n = 200
    if 'customer_id' in df_out.columns or 'subscriber_id' in df_out.columns:
        id_col = 'customer_id' if 'customer_id' in df_out.columns else 'subscriber_id'
    else:
        id_col = df_out.columns[0]  # best-effort id
    top_risky = df_out.sort_values('churn_probability', ascending=False).head(top_n)
    st.write(f"### Top {top_n} high-risk rows (for retention action)")
    st.dataframe(top_risky[[id_col, 'service_provider','circle','value','churn_probability','predicted_churn']].reset_index(drop=True))
    csv_buf = io.StringIO()
    top_risky.to_csv(csv_buf, index=False)
    st.download_button("Download top risky as CSV", csv_buf.getvalue(), file_name="top_risky_customers.csv")
except Exception as e:
    st.warning(f"Top risky customers table failed: {e}")

# ---------------------- NEW SECTION: SUBSCRIBER LOSS PREDICTION (2025-2026) ----------------------

st.markdown("---")
st.title("📉 Telecom Subscriber Loss & Revenue Risk Prediction (2025–2026)")

st.write("""
This section analyzes which telecom service providers might *lose subscribers* in upcoming years,
based on the existing trend data.  
More subscribers = more revenue.  
A decline indicates potential *revenue loss* or *market weakness* in that circle and month.
""")

# Ensure data is available
if 'df_out' in locals() or 'df_out' in globals():
    df_loss = df_out.copy()

    # Check required columns
    required_cols = ['service_provider', 'circle', 'month', 'year', 'value']
    missing_cols = [col for col in required_cols if col not in df_loss.columns]

    if missing_cols:
        st.warning(f"⚠ Missing columns: {', '.join(missing_cols)}. Please ensure these exist in your dataset.")
    else:
        # Convert month and year if needed
        df_loss['month'] = df_loss['month'].astype(str)
        df_loss['year'] = df_loss['year'].astype(int)

        # Group by and calculate monthly change
        df_loss = df_loss.sort_values(by=['service_provider', 'circle', 'year', 'month'])
        df_loss['subscriber_change'] = df_loss.groupby(['service_provider', 'circle'])['value'].diff()

        # Predict for 2025–2026 using linear projection (simple trend)
        future_years = [2025, 2026]
        predicted_rows = []
        for (sp, circle), group in df_loss.groupby(['service_provider', 'circle']):
            if group['year'].nunique() > 1:
                # Fit a basic linear model for trend projection
                X = group['year'].values.reshape(-1, 1)
                y = group['value'].values
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)
                for year in future_years:
                    pred = model.predict([[year]])[0]
                    predicted_rows.append({'service_provider': sp, 'circle': circle, 'year': year, 'predicted_subscribers': pred})
        df_future = pd.DataFrame(predicted_rows)

        # Calculate change compared to last known year
        latest_year = df_loss['year'].max()
        df_latest = df_loss[df_loss['year'] == latest_year].groupby(['service_provider', 'circle'])['value'].mean().reset_index()
        df_future = df_future.merge(df_latest, on=['service_provider', 'circle'], how='left', suffixes=('_future', '_latest'))
        df_future['change_in_subscribers'] = df_future['predicted_subscribers'] - df_future['value']
        df_future['loss_flag'] = df_future['change_in_subscribers'] < 0

        st.subheader("🔮 Predicted Loss Summary (2025–2026)")
        st.dataframe(df_future[['service_provider', 'circle', 'year', 'predicted_subscribers', 'change_in_subscribers', 'loss_flag']])

        # Highlight companies expected to lose subscribers
        losing_companies = df_future[df_future['loss_flag'] == True]
        if not losing_companies.empty:
            st.markdown("### 🚨 Companies predicted to face subscriber decline")
            st.dataframe(losing_companies[['service_provider', 'circle', 'year', 'predicted_subscribers', 'change_in_subscribers']])
        else:
            st.success("✅ No companies are projected to lose subscribers based on current data trends.")

        # Visualization: Subscriber Prediction Trend
        import plotly.express as px
        st.subheader("📊 Predicted Subscriber Trends by Service Provider (2025–2026)")
        fig = px.bar(
            df_future,
            x='circle',
            y='predicted_subscribers',
            color='service_provider',
            barmode='group',
            facet_col='year',
            title="Predicted Subscriber Count by Circle and Provider (2025–2026)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Optional: Circle-wise Loss Visualization
        st.subheader("🌍 Circles with Highest Expected Loss")

        # Ensure we have valid numeric data
        losing_companies = losing_companies.dropna(subset=['change_in_subscribers'])
        losing_companies['change_in_subscribers'] = losing_companies['change_in_subscribers'].astype(float)

        # Only include losses (negative changes)
        loss_summary = (
            losing_companies[losing_companies['change_in_subscribers'] < 0]
            .groupby('circle')['change_in_subscribers']
            .sum()
            .sort_values()
            .reset_index()
        )

        if not loss_summary.empty:
            import numpy as np
            # Convert to absolute values for better pie display (positive slice sizes)
            loss_summary['abs_loss'] = np.abs(loss_summary['change_in_subscribers'])
            fig2 = px.pie(
                loss_summary,
                names='circle',
                values='abs_loss',
                title="Proportion of Expected Subscriber Loss by Circle (2025–2026)",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("📡 No significant subscriber losses detected per circle — all providers show stable or growing trends.")

else:
    st.warning("⚠ No data available yet. Please upload or generate predictions above to continue this analysis.")

st.info("Notes: \n- This app highlights strengths (top features, revenue saved, ROI, providers by revenue). It does not surface every low-level weakness. \n- Ensure your uploaded CSV contains the engineered features used by training (or include the original raw columns and the same preprocessing pipeline files).")
