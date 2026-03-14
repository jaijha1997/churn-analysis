"""
SHAP-based explainability for churn predictions.
Generates per-customer and segment-level explanations.

TODO:
- Add waterfall plots for individual customer explanations
- Export SHAP summary plots to outputs/ for the dashboard
- Cluster customers by SHAP profile for segment-level insights
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

FEATURE_DISPLAY_NAMES = {
    "tenure_months": "Customer Tenure",
    "monthly_charges": "Monthly Charges",
    "total_charges": "Total Revenue",
    "num_products": "# Products",
    "support_tickets_90d": "Support Tickets (90d)",
    "login_frequency_30d": "Login Frequency (30d)",
    "last_login_days_ago": "Days Since Last Login",
    "avg_session_duration_min": "Avg Session Duration",
    "payment_failures_6m": "Payment Failures (6m)",
    "contract_type_encoded": "Contract Type",
    "internet_service_encoded": "Internet Service",
    "has_tech_support": "Has Tech Support",
    "has_online_backup": "Has Online Backup",
    "charges_per_product": "Charges per Product",
    "support_ticket_rate": "Support Ticket Rate",
    "engagement_score": "Engagement Score",
    "revenue_trend": "Revenue Trend",
}


class ChurnExplainer:
    def __init__(self, model, feature_names: List[str]):
        self.feature_names = feature_names
        self.display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names]
        # Use TreeExplainer for XGBoost/LightGBM — fast and exact
        self.xgb_explainer = shap.TreeExplainer(model.xgb_model)
        self.lgb_explainer = shap.TreeExplainer(model.lgb_model)
        self._shap_values = None

    def compute_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Compute ensemble SHAP values (avg of XGB + LGB)."""
        logger.info("Computing SHAP values...")
        xgb_shap = self.xgb_explainer.shap_values(X)
        lgb_shap = self.lgb_explainer.shap_values(X)

        # Handle LightGBM returning list for binary classification
        if isinstance(lgb_shap, list):
            lgb_shap = lgb_shap[1]
        if isinstance(xgb_shap, list):
            xgb_shap = xgb_shap[1]

        self._shap_values = (xgb_shap + lgb_shap) / 2
        logger.info(f"SHAP values computed for {len(X)} customers")
        return self._shap_values

    def get_top_risk_factors(self, customer_idx: int, top_k: int = 5) -> List[Dict]:
        """Get top churn drivers for a single customer."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values first")

        shap_vals = self._shap_values[customer_idx]
        top_indices = np.argsort(np.abs(shap_vals))[::-1][:top_k]

        return [
            {
                "feature": self.display_names[i],
                "shap_value": float(shap_vals[i]),
                "direction": "increases churn risk" if shap_vals[i] > 0 else "decreases churn risk",
            }
            for i in top_indices
        ]

    def plot_summary(self, X: np.ndarray, save_path: Optional[str] = None):
        if self._shap_values is None:
            self.compute_shap_values(X)
        shap.summary_plot(
            self._shap_values,
            X,
            feature_names=self.display_names,
            show=save_path is None,
        )
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info(f"Saved SHAP summary plot to {save_path}")

    def get_global_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature."""
        if self._shap_values is None:
            raise RuntimeError("Call compute_shap_values first")
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.display_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )
