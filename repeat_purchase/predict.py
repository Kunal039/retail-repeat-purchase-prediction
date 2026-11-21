"""Score customers with the trained model.

    python -m repeat_purchase.predict                                  # scores data/sample_customers.csv
    python -m repeat_purchase.predict --input my_customers.csv --output scored.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .config import DATA_DIR, DECISION_THRESHOLD, INPUT_COLUMNS, MODEL_PATH
from .data import engineer


def load_model(path: Path = MODEL_PATH):
    if not Path(path).exists():
        raise SystemExit(f"Model not found at {path}. Train it first: python -m repeat_purchase.train")
    return joblib.load(path)


def score(customers: pd.DataFrame, model=None, threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Return the input with repeat_probability, prediction and recommended_action columns."""
    missing = [c for c in INPUT_COLUMNS if c not in customers.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    model = model if model is not None else load_model()
    prob = model.predict_proba(engineer(customers[INPUT_COLUMNS]))[:, 1]
    out = customers.copy()
    out["repeat_probability"] = prob.round(4)
    out["will_repeat"] = (prob >= threshold).astype(int)
    out["recommended_action"] = ["Loyalty reward / upsell" if p >= threshold else "Win-back discount campaign"
                                 for p in prob]
    return out


def main():
    ap = argparse.ArgumentParser(description="Predict 30-day repeat purchase for customers in a CSV file.")
    ap.add_argument("--input", type=Path, default=DATA_DIR / "sample_customers.csv")
    ap.add_argument("--output", type=Path, help="optional CSV path for the scored results")
    ap.add_argument("--threshold", type=float, default=DECISION_THRESHOLD)
    args = ap.parse_args()

    scored = score(pd.read_csv(args.input), threshold=args.threshold)
    cols = [c for c in ("customer_id",) if c in scored.columns] + \
           ["repeat_probability", "will_repeat", "recommended_action"]
    with pd.option_context("display.width", 120):
        print(scored[cols].to_string(index=False))
    if args.output:
        scored.to_csv(args.output, index=False)
        print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
