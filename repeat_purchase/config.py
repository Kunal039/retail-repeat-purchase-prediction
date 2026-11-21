"""Shared constants: random seed, feature lists and output locations."""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FIGURES_DIR = ROOT / "reports" / "figures"
METRICS_DIR = ROOT / "reports" / "metrics"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "repeat_purchase_model.joblib"

TARGET = "repeat_purchase_30d"

NUMERIC_FEATURES = [
    "age", "visits_per_month", "avg_basket_value", "days_since_last_purchase",
    "discount_usage_pct", "satisfaction_rating", "complaints_last_6m",
    "monthly_spend", "engagement_score",
]
CATEGORICAL_FEATURES = ["gender", "city_tier", "channel", "membership", "recency_bucket"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Raw input columns a new customer record must provide (engineered features are derived).
INPUT_COLUMNS = [
    "age", "gender", "city_tier", "channel", "membership", "visits_per_month",
    "avg_basket_value", "days_since_last_purchase", "discount_usage_pct",
    "satisfaction_rating", "complaints_last_6m",
]

DECISION_THRESHOLD = 0.5
