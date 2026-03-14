"""
Batch prediction script — scores all customers and generates LLM retention recommendations
for the top at-risk segment.

Usage:
    python scripts/predict.py
    python scripts/predict.py --data data/customers.csv --top-k 50 --llm
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from data.generate_data import generate_customer_data
from src.models.trainer import ChurnTrainer
from src.explainability.shap_analysis import ChurnExplainer
from src.llm.retention_advisor import RetentionAdvisor
from config import FEATURE_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Batch churn prediction + retention recommendations")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--top-k", type=int, default=20, help="Number of top at-risk customers to generate LLM recs for")
    parser.add_argument("--llm", action="store_true", help="Enable LLM retention recommendations (requires OPENAI_API_KEY)")
    parser.add_argument("--output", type=str, default="outputs")
    args = parser.parse_args()

    # Load or generate data
    if args.data:
        df = pd.read_csv(args.data)
    else:
        logger.info("No data provided — generating synthetic customers")
        df = generate_customer_data(1000)

    # Load trained model
    trainer = ChurnTrainer(output_dir=args.output)
    try:
        trainer.load()
    except FileNotFoundError:
        logger.error("No trained model found. Run scripts/train.py first.")
        sys.exit(1)

    # Score all customers
    X = trainer.feature_engineer.transform(df, FEATURE_COLS)
    probas = trainer.model.predict_proba(X)[:, 1]
    df["churn_probability"] = probas
    df["churn_predicted"] = (probas >= 0.45).astype(int)

    high_risk = df[df["churn_probability"] >= 0.5].sort_values("churn_probability", ascending=False)
    logger.info(f"High-risk customers: {len(high_risk)} / {len(df)} ({len(high_risk)/len(df):.1%})")

    # SHAP explanations for top-k
    top_k_df = high_risk.head(args.top_k)
    X_top = trainer.feature_engineer.transform(top_k_df, FEATURE_COLS)
    explainer = ChurnExplainer(trainer.model, FEATURE_COLS)
    shap_values = explainer.compute_shap_values(X_top)

    results = []
    for i, (_, row) in enumerate(top_k_df.iterrows()):
        risk_factors = explainer.get_top_risk_factors(i, top_k=5)
        results.append({
            "customer_id": row["customer_id"],
            "churn_probability": round(row["churn_probability"], 4),
            "risk_factors": risk_factors,
            "context": {
                "tenure_months": int(row["tenure_months"]),
                "monthly_charges": float(row["monthly_charges"]),
                "contract_type": ["Month-to-Month", "One Year", "Two Year"][int(row["contract_type_encoded"])],
            },
        })

    if args.llm:
        logger.info("Generating LLM retention recommendations...")
        advisor = RetentionAdvisor()
        recommendations = advisor.batch_generate(results)
        out_path = Path(args.output) / "retention_recommendations.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(recommendations, f, indent=2)
        logger.info(f"Saved recommendations to {out_path}")
    else:
        out_path = Path(args.output) / "at_risk_customers.json"
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved at-risk customer profiles to {out_path}")

    print(f"\nDone. Top at-risk customers scored: {len(results)}")


if __name__ == "__main__":
    main()
