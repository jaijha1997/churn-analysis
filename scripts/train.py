"""
Training script — run this to train the ensemble model.

Usage:
    python scripts/train.py
    python scripts/train.py --data data/customers.csv --output outputs/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from data.generate_data import generate_customer_data
from src.models.trainer import ChurnTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction ensemble")
    parser.add_argument("--data", type=str, default=None, help="Path to customer CSV (generates synthetic if not provided)")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory for model artifacts")
    parser.add_argument("--n-customers", type=int, default=5000, help="Number of synthetic customers to generate")
    args = parser.parse_args()

    if args.data:
        logger.info(f"Loading data from {args.data}")
        df = pd.read_csv(args.data)
    else:
        logger.info(f"Generating {args.n_customers} synthetic customers...")
        df = generate_customer_data(args.n_customers)

    trainer = ChurnTrainer(output_dir=args.output)
    metrics = trainer.train(df)

    print("\n=== Training Complete ===")
    print(f"  AUC-ROC:  {metrics['auc']:.4f}")
    print(f"  PR-AUC:   {metrics['pr_auc']:.4f}")
    print(f"  Artifacts: {args.output}/")


if __name__ == "__main__":
    main()
