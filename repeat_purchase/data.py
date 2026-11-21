"""Dataset generation, cleaning and feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SEED


class CustomerDataGenerator:
    """Creates a reproducible, anonymised retail-customer dataset (no real PII).

    Attribute distributions and their relationship to the target follow retail
    behaviour described in the customer-retention literature (recency, frequency,
    monetary value, loyalty tier, service quality). Realistic data-quality issues
    (missing values, duplicates, extreme basket values) are injected on purpose so
    the cleaning stage has something to fix.
    """

    def __init__(self, n: int = 5000, seed: int = SEED):
        self.n = n
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        r, n = self.rng, self.n
        df = pd.DataFrame({
            "customer_id": [f"C{100000 + i}" for i in range(n)],
            "age": r.integers(18, 66, n),
            "gender": r.choice(["Male", "Female"], n, p=[0.48, 0.52]),
            "city_tier": r.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.55, 0.30, 0.15]),
            "channel": r.choice(["In-store", "Online", "Both"], n, p=[0.45, 0.30, 0.25]),
            "membership": r.choice(["None", "Silver", "Gold"], n, p=[0.50, 0.35, 0.15]),
            "visits_per_month": np.round(r.gamma(2.0, 1.4, n), 1),
            "avg_basket_value": np.round(r.lognormal(7.0, 0.5, n), 0),
            "days_since_last_purchase": r.integers(1, 181, n),
            "discount_usage_pct": np.round(r.beta(2, 5, n) * 100, 1),
            "satisfaction_rating": r.integers(1, 6, n),
            "complaints_last_6m": r.poisson(0.4, n),
        })
        tier = df["membership"].map({"None": 0, "Silver": 0.6, "Gold": 1.2})
        both = (df["channel"] == "Both").astype(float) * 0.5
        logit = (-1.0 + 0.45 * df["visits_per_month"] - 0.022 * df["days_since_last_purchase"]
                 + 0.35 * (df["satisfaction_rating"] - 3) + tier + both
                 - 0.45 * df["complaints_last_6m"] + 0.012 * df["discount_usage_pct"]
                 + 0.15 * np.log(df["avg_basket_value"] / 1000) + r.normal(0, 0.8, n))
        df["repeat_purchase_30d"] = (r.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

        # inject realistic data-quality issues for the cleaning stage
        for col, frac in [("age", 0.03), ("avg_basket_value", 0.04), ("satisfaction_rating", 0.02)]:
            df.loc[r.choice(n, int(n * frac), replace=False), col] = np.nan
        df = pd.concat([df, df.sample(60, random_state=SEED)], ignore_index=True)  # duplicates
        df.loc[r.choice(len(df), 15, replace=False), "avg_basket_value"] = 250000  # outliers
        return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop duplicate customers and cap basket value at the 99th percentile.

    Missing values are *not* filled here: imputation happens inside the model
    pipeline so it is learned from training folds only (no data leakage).

    Note: percentile capping only removes extreme values when they make up
    less than 1 % of rows (true for the default 5,000-customer dataset).
    """
    log = {"raw_rows": len(df), "missing_before": int(df.isna().sum().sum())}
    df = df.drop_duplicates(subset="customer_id").copy()
    log["duplicates_removed"] = log["raw_rows"] - len(df)
    cap = df["avg_basket_value"].quantile(0.99)
    log["outliers_capped"] = int((df["avg_basket_value"] > cap).sum())
    df["avg_basket_value"] = df["avg_basket_value"].clip(upper=cap)
    log["clean_rows"] = len(df)
    return df, log


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add RFM-style features: recency bucket, monthly spend and engagement score."""
    df = df.copy()
    df["recency_bucket"] = pd.cut(df["days_since_last_purchase"], [0, 30, 90, 180],
                                  labels=["Recent", "Lapsing", "Dormant"]).astype(str)
    df["monthly_spend"] = df["visits_per_month"] * df["avg_basket_value"]
    df["engagement_score"] = df["visits_per_month"] / (1 + df["days_since_last_purchase"] / 30)
    return df
