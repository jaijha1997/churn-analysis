"""
XGBoost + LightGBM ensemble for churn prediction.
Uses weighted average of predicted probabilities.

TODO: experiment with stacking (meta-learner) instead of simple weighted avg
"""

import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ChurnEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        xgb_params: Optional[Dict] = None,
        lgb_params: Optional[Dict] = None,
        weights: Optional[Dict] = None,
        threshold: float = 0.45,
    ):
        self.xgb_params = xgb_params or {}
        self.lgb_params = lgb_params or {}
        self.weights = weights or {"xgboost": 0.5, "lightgbm": 0.5}
        self.threshold = threshold
        self.xgb_model = None
        self.lgb_model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ChurnEnsemble":
        logger.info("Training XGBoost...")
        self.xgb_model = xgb.XGBClassifier(**self.xgb_params)
        self.xgb_model.fit(X, y)

        logger.info("Training LightGBM...")
        self.lgb_model = lgb.LGBMClassifier(**self.lgb_params)
        self.lgb_model.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        xgb_proba = self.xgb_model.predict_proba(X)
        lgb_proba = self.lgb_model.predict_proba(X)

        w_xgb = self.weights["xgboost"]
        w_lgb = self.weights["lightgbm"]
        ensemble_proba = w_xgb * xgb_proba + w_lgb * lgb_proba
        return ensemble_proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

    def cross_val_optimize_weights(self, X: np.ndarray, y: np.ndarray, cv: int = 5):
        """Grid search over ensemble weights using CV AUC. Not yet wired into trainer."""
        # TODO: implement this — for now just use 50/50
        raise NotImplementedError("Weight optimization not yet implemented")

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances from both models."""
        xgb_imp = self.xgb_model.feature_importances_
        lgb_imp = self.lgb_model.feature_importances_
        return (xgb_imp + lgb_imp) / 2
