# Customer Churn Analysis & Prediction

> **WIP** — core pipeline is functional, some features still being wired up (see TODOs)

AI-powered churn prediction system using behavioral and transactional data. End-to-end pipeline: feature engineering → XGBoost + LightGBM ensemble → SHAP explainability → LLM-generated retention recommendations per at-risk customer segment.

## Stack

`Python` · `XGBoost` · `LightGBM` · `SHAP` · `OpenAI API` · `FastAPI` · `Pandas` · `scikit-learn`

## Architecture

```
Raw Customer Data
       ↓
Feature Engineering (ChurnFeatureEngineer)
  - Behavioral features: engagement score, login frequency, session duration
  - Derived: charges/product, support ticket rate, revenue trend
       ↓
Ensemble Model (XGBoost 50% + LightGBM 50%)
  - Outputs churn probability per customer
       ↓
SHAP Explainer
  - Per-customer top risk factors
  - Global feature importance
       ↓
LLM Retention Advisor (GPT-4o-mini)
  - Personalized retention recommendations
  - Customer segment classification
  - Urgency scoring
       ↓
FastAPI Service (real-time + batch prediction)
```

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Train on synthetic data
python scripts/train.py

# Score customers + SHAP explanations
python scripts/predict.py

# With LLM recommendations (requires OPENAI_API_KEY in .env)
python scripts/predict.py --llm --top-k 50

# Start API server
uvicorn src.api.main:app --reload
```

## API

```
POST /predict              — single customer churn prediction
POST /predict/batch        — batch (up to 100 customers)
GET  /customer/{id}/explain — SHAP explanation (not yet implemented)
GET  /health
```

## Config

Copy `.env.example` to `.env` and set:
```
OPENAI_API_KEY=sk-...
```

## Results (on synthetic data)

| Metric  | Score  |
|---------|--------|
| AUC-ROC | ~0.87  |
| PR-AUC  | ~0.72  |

> Numbers on real data will vary. Threshold tuned to 0.45 to catch more at-risk customers at cost of some precision.

## TODOs

- [ ] Wire up `/customer/{id}/explain` endpoint
- [ ] Add async batch LLM processing
- [ ] Add auth middleware (API key)
- [ ] Optimize ensemble weights via cross-validation
- [ ] Add RFM features + rolling time-series aggregations
- [ ] Connect to actual data warehouse (Snowflake/BigQuery)
- [ ] Prometheus metrics + dashboard
- [ ] Human eval set for LLM recommendation quality
