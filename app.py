"""
Streamlit app – Credit risk classification (UCI German Credit) with 5 ML models.

Features:
  a. Upload test data (CSV)
  b. Model selection dropdown
  c. Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
  d. Confusion matrix + classification report
The app only LOADS the saved models from /model – it never retrains.
"""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TARGET = "credit_risk"
CLASS_NAMES = ["Good credit (0)", "Bad credit (1)"]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl",
}

st.set_page_config(page_title="Credit Risk Classification", layout="wide")


@st.cache_resource
def load_model(file_name: str):
    return joblib.load(MODEL_DIR / file_name)


@st.cache_resource
def load_feature_names():
    return joblib.load(MODEL_DIR / "feature_names.pkl")


def get_scores(model, X):
    """Positive-class scores for AUC: probabilities if available, else SVM decision values."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def compute_metrics(model, X, y) -> dict:
    y_pred = model.predict(X)
    metrics = {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, get_scores(model, X)) if y.nunique() == 2 else float("nan"),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }
    return metrics, y_pred


# ---------------- Sidebar ----------------
st.sidebar.header("1. Upload test data")
uploaded = st.sidebar.file_uploader("Upload CSV (20 features + 'credit_risk' column)", type=["csv"])

sample_path = ROOT / "test_data.csv"
if sample_path.exists():
    st.sidebar.download_button("Download sample test_data.csv",
                               sample_path.read_bytes(), "test_data.csv", "text/csv")

st.sidebar.header("2. Select model")
model_name = st.sidebar.selectbox("Model", list(MODEL_FILES.keys()))

# ---------------- Main ----------------
st.title("Credit Risk Classification – 5 ML Models")
st.caption("Dataset: Statlog (German Credit Data), UCI · 1000 applicants · 20 features · "
           "Target: 0 = Good credit risk, 1 = Bad credit risk")

if uploaded is None:
    st.info("Upload a test CSV from the sidebar to evaluate the models. "
            "You can download the sample test_data.csv from the sidebar.")
    st.stop()

df = pd.read_csv(uploaded)
features = load_feature_names()

missing = [c for c in features + [TARGET] if c not in df.columns]
if missing:
    st.error(f"The uploaded file is missing these columns: {missing}")
    st.stop()

X, y = df[features], df[TARGET].astype(int)

with st.expander(f"Preview uploaded data ({df.shape[0]} rows, {df.shape[1]} columns)"):
    st.dataframe(df.head(20), width="stretch")

model = load_model(MODEL_FILES[model_name])
metrics, y_pred = compute_metrics(model, X, y)

# c. Metrics
st.subheader(f"Evaluation metrics – {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

# d. Confusion matrix + classification report
left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, labels=[0, 1], target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

# Comparison of all models on the uploaded test data
st.subheader("Comparison of all 5 models on the uploaded test data")
rows = []
for name, fname in MODEL_FILES.items():
    m, _ = compute_metrics(load_model(fname), X, y)
    rows.append({"ML Model Name": name, **m})
comparison = pd.DataFrame(rows).set_index("ML Model Name").round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#cde2fb"), width="stretch")
