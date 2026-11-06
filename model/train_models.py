"""
ML Assignment 2 – Credit risk classification on the UCI Statlog (German Credit Data) dataset.

Trains 5 classifiers, evaluates them with 6 metrics on a held-out test set and saves:
  - model/*.pkl          trained pipelines (pre-processing + model)
  - model/metrics.csv    comparison table
  - test_data.csv        test data (features + target) for the Streamlit app
  - data/german_credit.csv  decoded full dataset

Run from the project root:
    python model/train_models.py
"""
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
RANDOM_STATE = 42
TARGET = "credit_risk"  # 1 = bad credit risk, 0 = good credit risk

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "own_telephone", "foreign_worker", "class",
]

# Code book from the UCI documentation (german.doc)
CODES = {
    "checking_status": {"A11": "< 0 DM", "A12": "0-200 DM", "A13": ">= 200 DM", "A14": "no checking account"},
    "credit_history": {"A30": "no credits / all paid", "A31": "all paid at this bank",
                       "A32": "existing paid duly", "A33": "delay in past",
                       "A34": "critical account"},
    "purpose": {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment",
                "A43": "radio/television", "A44": "domestic appliances", "A45": "repairs",
                "A46": "education", "A47": "vacation", "A48": "retraining",
                "A49": "business", "A410": "others"},
    "savings": {"A61": "< 100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
                "A64": ">= 1000 DM", "A65": "unknown / none"},
    "employment_since": {"A71": "unemployed", "A72": "< 1 year", "A73": "1-4 years",
                         "A74": "4-7 years", "A75": ">= 7 years"},
    "personal_status_sex": {"A91": "male: divorced/separated", "A92": "female: div/sep/married",
                            "A93": "male: single", "A94": "male: married/widowed",
                            "A95": "female: single"},
    "other_debtors": {"A101": "none", "A102": "co-applicant", "A103": "guarantor"},
    "property": {"A121": "real estate", "A122": "savings agreement / life insurance",
                 "A123": "car or other", "A124": "unknown / none"},
    "other_installment_plans": {"A141": "bank", "A142": "stores", "A143": "none"},
    "housing": {"A151": "rent", "A152": "own", "A153": "for free"},
    "job": {"A171": "unemployed / unskilled non-resident", "A172": "unskilled resident",
            "A173": "skilled employee", "A174": "management / highly qualified"},
    "own_telephone": {"A191": "none", "A192": "yes"},
    "foreign_worker": {"A201": "yes", "A202": "no"},
}
CATEGORICAL = list(CODES.keys())
NUMERIC = ["duration_months", "credit_amount", "installment_rate", "residence_since",
           "age", "existing_credits", "num_dependents"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "german_raw.csv", header=None, names=COLUMNS)
    for col, mapping in CODES.items():
        df[col] = df[col].map(mapping)
    df[TARGET] = (df.pop("class") == 2).astype(int)  # 1 = bad, 0 = good
    return df


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


def build_models() -> dict:
    """Each model is a Pipeline, so the pre-processing is saved together with the model."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced",
                                                  random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10,
                                                class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", class_weight="balanced",
                                      random_state=RANDOM_STATE),
    }


def get_scores(model, X):
    """Positive-class scores for AUC: probabilities if available, else SVM decision values."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, get_scores(model, X)),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1": f1_score(y, y_pred),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = load_data()
    df.to_csv(DATA_DIR / "german_credit.csv", index=False)
    print(f"Dataset: {df.shape[0]} rows, {df.shape[1] - 1} features "
          f"({len(NUMERIC)} numeric, {len(CATEGORICAL)} categorical)")
    print("Class balance:", df[TARGET].value_counts().to_dict())

    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    rows = []
    for name, clf in build_models().items():
        model = Pipeline([("prep", preprocessor()), ("clf", clf)])
        model.fit(X_train, y_train)
        fname = name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, MODEL_DIR / fname)
        rows.append({"ML Model Name": name, **evaluate(model, X_test, y_test)})
        print(f"Saved model/{fname}")

    results = pd.DataFrame(rows).round(4)
    results.to_csv(MODEL_DIR / "metrics.csv", index=False)
    joblib.dump(list(X.columns), MODEL_DIR / "feature_names.pkl")
    (MODEL_DIR / "sklearn_version.txt").write_text(sklearn.__version__)
    print(f"\nscikit-learn version used: {sklearn.__version__}\n")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
