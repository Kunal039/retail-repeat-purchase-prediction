"""Per-customer explanations for the (linear) final model.

For Logistic Regression the log-odds is intercept + sum(coef_i * x_i), where x_i
are the pre-processed (scaled / one-hot) features. Each term is that feature's
contribution; one-hot columns are summed back into their original feature.
"""
from __future__ import annotations

import pandas as pd

from .config import CATEGORICAL_FEATURES, INPUT_COLUMNS
from .data import engineer

LABELS = {
    "age": "Age",
    "gender": "Gender",
    "city_tier": "City tier",
    "channel": "Shopping channel",
    "membership": "Membership",
    "visits_per_month": "Visits per month",
    "avg_basket_value": "Average basket value",
    "days_since_last_purchase": "Days since last purchase",
    "discount_usage_pct": "Discount usage %",
    "satisfaction_rating": "Satisfaction rating",
    "complaints_last_6m": "Complaints (6 months)",
    "monthly_spend": "Monthly spend",
    "engagement_score": "Engagement score",
    "recency_bucket": "Recency bucket",
}


def _original_feature(transformed_name: str) -> str:
    kind, name = transformed_name.split("__", 1)
    if kind == "cat":
        for col in CATEGORICAL_FEATURES:
            if name.startswith(col + "_"):
                return col
    return name


def contributions(model, customer: pd.DataFrame) -> pd.DataFrame:
    """Return each feature's contribution to the log-odds for a single customer, largest effect first."""
    if len(customer) != 1:
        raise ValueError("contributions() expects exactly one customer")
    prep, clf = model.named_steps["prep"], model.named_steps["model"]
    if not hasattr(clf, "coef_"):
        raise TypeError("contributions() needs a linear model with coef_")
    x = prep.transform(engineer(customer[INPUT_COLUMNS]))
    x = x.toarray() if hasattr(x, "toarray") else x
    terms = pd.Series(x[0] * clf.coef_[0], index=prep.get_feature_names_out())
    by_feature = terms.groupby(_original_feature).sum()
    out = by_feature.rename("contribution").reset_index().rename(columns={"index": "feature"})
    out["label"] = out["feature"].map(LABELS).fillna(out["feature"])
    out["effect"] = out["contribution"].map(lambda v: "raises chance" if v > 0 else "lowers chance")
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
