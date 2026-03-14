import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Model params
XGBOOST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

ENSEMBLE_WEIGHTS = {"xgboost": 0.5, "lightgbm": 0.5}  # TODO: tune weights via CV

CHURN_THRESHOLD = 0.45  # slightly lower to catch more at-risk customers

FEATURE_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_tickets_90d",
    "login_frequency_30d",
    "last_login_days_ago",
    "avg_session_duration_min",
    "payment_failures_6m",
    "contract_type_encoded",
    "internet_service_encoded",
    "has_tech_support",
    "has_online_backup",
    "charges_per_product",
    "support_ticket_rate",
    "engagement_score",
    "revenue_trend",
]
