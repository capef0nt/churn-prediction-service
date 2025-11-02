# pipeline/train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
from pathlib import Path

DATA_PATH = Path("data/ibm_telco_clean.csv")
MODEL_PATH = Path("artifacts/model.joblib")

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def prepare_features(df: pd.DataFrame):
    # --- Separate features and target ---
    X = df.drop(columns=["churn_flag", "customerid"])
    y = df["churn_flag"]

    # --- Identify column types ---
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    print(f"Numeric features: {list(num_features)}")
    print(f"Categorical features: {list(cat_features)}")

    # --- Preprocess ---
    numeric_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor):
    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Define model ---
    model = LogisticRegression(max_iter=200, solver="liblinear")

    # --- Combine preprocessing + model into a pipeline ---
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # --- Train ---
    clf.fit(X_train, y_train)

    # --- Predict on test set ---
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # --- Evaluate ---
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.3f}")
    print(f"ROC-AUC: {auc:.3f}")

    return clf

def save_model(clf, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    print(f"Model saved to {path.resolve()}")

if __name__ == "__main__":
    df = load_data()
    X, y, preprocessor = prepare_features(df)
    clf = train_and_evaluate(X, y, preprocessor)
    save_model(clf)
