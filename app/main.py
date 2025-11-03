# app/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# 
# --- load the trained pipeline (preprocessor + model) ---
# 
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
model = joblib.load(MODEL_PATH)

# 
# --- define the expected input schema ---
# 
class Customer(BaseModel):
    gender: str
    seniorcitizen: int
    partner: int
    dependents: int
    tenure: int
    tenure_years: float
    phoneservice: int
    multiplelines: str
    internetservice: str
    onlinebackup: int
    onlinesecurity: int
    deviceprotection: int
    techsupport: int
    streamingtv: int
    streamingmovies: int
    contract: str
    paperlessbilling: int
    paymentmethod: str
    monthlycharges: float
    totalcharges: float

# 
# --- init fastapi app ---
# 
app = FastAPI(title="Churn Prediction API")

# 
# --- health endpoint ---
# 
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model)}

# 
# --- predict endpoint ---
# 
@app.post("/predict")
def predict(customer: Customer):
    """
    Accept a customer JSON, return churn probability
    """
    # convert input into dataframe (1-row)
    X = pd.DataFrame([customer.model_dump()])

    # model handles preprocessing internally
    proba = model.predict_proba(X)[0, 1]
    label = int(proba >= 0.5)
    return {"churn_probability": round(float(proba), 3), "churn_flag": label}
