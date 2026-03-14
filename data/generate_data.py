"""
Synthetic customer data generator for churn analysis.
In production this would pull from the data warehouse (Snowflake/BigQuery).
TODO: add connector for actual data source
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)


def generate_customer_data(n_customers: int = 5000) -> pd.DataFrame:
    """Generate synthetic telecom-style customer data."""

    tenure = np.random.exponential(scale=24, size=n_customers).clip(1, 72).astype(int)
    monthly_charges = np.random.normal(65, 25, n_customers).clip(20, 150)
    num_products = np.random.randint(1, 6, n_customers)
    contract_type = np.random.choice([0, 1, 2], n_customers, p=[0.5, 0.3, 0.2])  # month, 1yr, 2yr
    internet_service = np.random.choice([0, 1, 2], n_customers, p=[0.1, 0.45, 0.45])
    has_tech_support = np.random.randint(0, 2, n_customers)
    has_online_backup = np.random.randint(0, 2, n_customers)

    # Behavioral signals
    login_frequency = np.random.poisson(15, n_customers).clip(0, 60)
    last_login_days = np.random.exponential(scale=7, size=n_customers).clip(0, 90).astype(int)
    avg_session_duration = np.random.exponential(scale=12, size=n_customers).clip(1, 60)
    support_tickets = np.random.poisson(1.5, n_customers).clip(0, 15)
    payment_failures = np.random.poisson(0.5, n_customers).clip(0, 10)

    total_charges = monthly_charges * tenure + np.random.normal(0, 50, n_customers)
    total_charges = total_charges.clip(0)

    # Churn probability — higher for month-to-month, low tenure, high support tickets
    churn_logit = (
        -2.0
        + 0.8 * (contract_type == 0).astype(float)
        - 0.03 * tenure
        + 0.02 * support_tickets
        + 0.015 * payment_failures
        - 0.01 * login_frequency
        + 0.01 * last_login_days
        + 0.005 * monthly_charges
        - 0.2 * has_tech_support
        + np.random.normal(0, 0.3, n_customers)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churned = (np.random.random(n_customers) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:05d}" for i in range(n_customers)],
        "tenure_months": tenure,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "num_products": num_products,
        "support_tickets_90d": support_tickets,
        "login_frequency_30d": login_frequency,
        "last_login_days_ago": last_login_days,
        "avg_session_duration_min": avg_session_duration.round(1),
        "payment_failures_6m": payment_failures,
        "contract_type_encoded": contract_type,
        "internet_service_encoded": internet_service,
        "has_tech_support": has_tech_support,
        "has_online_backup": has_online_backup,
        "churned": churned,
    })

    print(f"Generated {n_customers} customers | Churn rate: {churned.mean():.1%}")
    return df


if __name__ == "__main__":
    df = generate_customer_data(5000)
    out_path = Path(__file__).parent / "customers.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
