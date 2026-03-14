"""
Feature engineering pipeline.
Transforms raw customer data into model-ready features.

TODO:
- Add rolling window aggregations once we have time-series data
- Normalize engagement_score across segments
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging

logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features to raw customer dataframe."""
        df = df.copy()

        # Charges per product — proxy for value perception
        df["charges_per_product"] = df["monthly_charges"] / df["num_products"].clip(1)

        # Support burden rate
        df["support_ticket_rate"] = df["support_tickets_90d"] / df["tenure_months"].clip(1)

        # Engagement score (0–1 range) — higher is better
        max_login = df["login_frequency_30d"].max() or 1
        max_session = df["avg_session_duration_min"].max() or 1
        df["engagement_score"] = (
            0.5 * (df["login_frequency_30d"] / max_login)
            + 0.3 * (df["avg_session_duration_min"] / max_session)
            + 0.2 * (1 - df["last_login_days_ago"] / 90).clip(0, 1)
        )

        # Revenue trend proxy — are they a high-value customer relative to their tenure?
        df["revenue_trend"] = df["total_charges"] / (df["monthly_charges"] * df["tenure_months"]).clip(0.01)
        df["revenue_trend"] = df["revenue_trend"].clip(0, 2)

        # RFM-style recency score — penalizes customers who haven't logged in recently
        df["recency_score"] = (1 - df["last_login_days_ago"] / 90).clip(0, 1)

        logger.info(f"Built features for {len(df)} customers")
        return df

    def fit_transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        df = self.build_features(df)
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        self.fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame, feature_cols: list) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Call fit_transform first")
        df = self.build_features(df)
        X = df[feature_cols].values
        return self.scaler.transform(X)

    def save(self, path: str):
        joblib.dump(self, path)
        logger.info(f"Saved feature engineer to {path}")

    @classmethod
    def load(cls, path: str) -> "ChurnFeatureEngineer":
        return joblib.load(path)
