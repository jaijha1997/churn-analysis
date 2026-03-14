"""
Training pipeline — handles train/test split, eval metrics, and model persistence.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
)
import joblib
import logging
from pathlib import Path
from typing import Dict, Tuple

from src.models.ensemble import ChurnEnsemble
from src.features.feature_engineering import ChurnFeatureEngineer
from config import XGBOOST_PARAMS, LIGHTGBM_PARAMS, ENSEMBLE_WEIGHTS, CHURN_THRESHOLD, FEATURE_COLS

logger = logging.getLogger(__name__)


class ChurnTrainer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.feature_engineer = ChurnFeatureEngineer()
        self.model = ChurnEnsemble(
            xgb_params=XGBOOST_PARAMS,
            lgb_params=LIGHTGBM_PARAMS,
            weights=ENSEMBLE_WEIGHTS,
            threshold=CHURN_THRESHOLD,
        )

    def train(self, df: pd.DataFrame) -> Dict:
        logger.info(f"Training on {len(df)} customers...")

        X = self.feature_engineer.fit_transform(df, FEATURE_COLS)
        y = df["churned"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        self.model.fit(X_train, y_train)

        metrics = self._evaluate(X_test, y_test)
        logger.info(f"Test AUC: {metrics['auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
        print(classification_report(y_test, self.model.predict(X_test), target_names=["Retained", "Churned"]))

        self._save()
        return metrics

    def _evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        y_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        return {
            "auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        }

    def _save(self):
        joblib.dump(self.model, self.output_dir / "ensemble_model.pkl")
        self.feature_engineer.save(str(self.output_dir / "feature_engineer.pkl"))
        logger.info(f"Saved model artifacts to {self.output_dir}/")

    def load(self):
        self.model = joblib.load(self.output_dir / "ensemble_model.pkl")
        self.feature_engineer = ChurnFeatureEngineer.load(str(self.output_dir / "feature_engineer.pkl"))
        return self
