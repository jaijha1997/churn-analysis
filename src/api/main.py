"""
FastAPI service for real-time churn prediction and retention recommendations.

Endpoints:
  POST /predict          — predict churn probability for one customer
  POST /predict/batch    — batch prediction (up to 100 customers)
  GET  /customer/{id}/explain — SHAP explanation for a customer
  GET  /health           — health check

TODO:
- Add auth middleware (API key or JWT)
- Add request logging + Prometheus metrics
- Wire up /customer/{id}/explain endpoint (currently stub)
- Add rate limiting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path

from config import FEATURE_COLS, CHURN_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction API",
    description="AI-powered customer churn prediction with SHAP explainability and LLM retention recommendations",
    version="0.1.0",
)

# Lazy-loaded model artifacts
_model = None
_feature_engineer = None


def get_model():
    global _model, _feature_engineer
    if _model is None:
        model_path = Path("outputs/ensemble_model.pkl")
        fe_path = Path("outputs/feature_engineer.pkl")
        if not model_path.exists():
            raise HTTPException(status_code=503, detail="Model not trained yet. Run scripts/train.py first.")
        _model = joblib.load(model_path)
        _feature_engineer = joblib.load(fe_path)
    return _model, _feature_engineer


class CustomerFeatures(BaseModel):
    customer_id: str
    tenure_months: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1, le=10)
    support_tickets_90d: int = Field(0, ge=0)
    login_frequency_30d: int = Field(0, ge=0)
    last_login_days_ago: int = Field(0, ge=0)
    avg_session_duration_min: float = Field(0.0, ge=0)
    payment_failures_6m: int = Field(0, ge=0)
    contract_type_encoded: int = Field(..., ge=0, le=2)
    internet_service_encoded: int = Field(..., ge=0, le=2)
    has_tech_support: int = Field(0, ge=0, le=1)
    has_online_backup: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_predicted: bool
    risk_tier: str


class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures] = Field(..., max_length=100)


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict_single(customer: CustomerFeatures):
    model, fe = get_model()
    df = pd.DataFrame([customer.model_dump()])
    X = fe.transform(df, FEATURE_COLS)
    proba = model.predict_proba(X)[0, 1]
    predicted = bool(proba >= CHURN_THRESHOLD)
    risk_tier = _get_risk_tier(proba)
    return PredictionResponse(
        customer_id=customer.customer_id,
        churn_probability=round(float(proba), 4),
        churn_predicted=predicted,
        risk_tier=risk_tier,
    )


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    model, fe = get_model()
    records = [c.model_dump() for c in request.customers]
    df = pd.DataFrame(records)
    X = fe.transform(df, FEATURE_COLS)
    probas = model.predict_proba(X)[:, 1]
    results = []
    for i, customer in enumerate(request.customers):
        proba = float(probas[i])
        results.append({
            "customer_id": customer.customer_id,
            "churn_probability": round(proba, 4),
            "churn_predicted": proba >= CHURN_THRESHOLD,
            "risk_tier": _get_risk_tier(proba),
        })
    return {"predictions": results, "count": len(results)}


@app.get("/customer/{customer_id}/explain")
def explain_customer(customer_id: str):
    # TODO: implement — need to store SHAP values per customer after batch prediction
    # For now return a stub
    raise HTTPException(status_code=501, detail="SHAP explanation endpoint not yet implemented")


def _get_risk_tier(proba: float) -> str:
    if proba >= 0.7:
        return "critical"
    elif proba >= 0.5:
        return "high"
    elif proba >= 0.35:
        return "moderate"
    return "low"
