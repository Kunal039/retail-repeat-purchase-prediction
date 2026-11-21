"""End-to-end training pipeline.

    python -m repeat_purchase.train            # full run (~30 s)
    python -m repeat_purchase.train --n 2000   # smaller dataset
"""
from __future__ import annotations

import argparse
import json
import time

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split

from . import plots
from .config import (CATEGORICAL_FEATURES, DATA_DIR, FEATURES, FIGURES_DIR, METRICS_DIR, MODEL_PATH,
                     MODELS_DIR, NUMERIC_FEATURES, SEED, TARGET)
from .data import CustomerDataGenerator, clean, engineer
from .models import LR_GRID, RF_GRID, candidate_models, make_pipeline


def scores(model, X, y) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred),
            "Recall": recall_score(y, pred), "F1-Score": f1_score(y, pred),
            "ROC-AUC": roc_auc_score(y, prob)}


def run(n: int = 5000, seed: int = SEED, make_plots: bool = True) -> dict:
    t0 = time.time()
    for d in (DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print(" RETAIL CUSTOMER REPEAT-PURCHASE PREDICTION  |  ML PIPELINE")
    print("=" * 64)
    raw = CustomerDataGenerator(n, seed).generate()
    raw.to_csv(DATA_DIR / "customers_raw.csv", index=False)
    print(f"[1] Dataset generated        : {raw.shape[0]} rows x {raw.shape[1]} cols")

    df, log = clean(raw)
    print(f"[2] Cleaning                 : {log['duplicates_removed']} duplicates removed, "
          f"{log['missing_before']} missing values, {log['outliers_capped']} outliers capped")
    df = engineer(df)
    if make_plots:
        plots.eda(df, FIGURES_DIR)
    print(f"[3] Feature engineering + EDA: {len(NUMERIC_FEATURES)} numeric, "
          f"{len(CATEGORICAL_FEATURES)} categorical features")

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    print(f"[4] Train/Test split (80/20) : {len(Xtr)} / {len(Xte)}  | positive rate {y.mean():.3f}")

    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    print("[5] Training & evaluating models ...")
    for name, model in candidate_models().items():
        pipe = make_pipeline(model)
        start = time.time()
        cv_auc = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="roc_auc").mean()
        pipe.fit(Xtr, ytr)
        rows.append({"Model": name, **scores(pipe, Xte, yte), "CV ROC-AUC": cv_auc,
                     "Time (s)": time.time() - start})
        fitted[name] = pipe
        print(f"    {name:<24} acc={rows[-1]['Accuracy']:.3f}  f1={rows[-1]['F1-Score']:.3f}  "
              f"auc={rows[-1]['ROC-AUC']:.3f}")
    results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    results.to_csv(METRICS_DIR / "model_results.csv", index=False)

    print("[6] Hyper-parameter tuning (GridSearchCV, Random Forest) ...")
    rf_grid = GridSearchCV(make_pipeline(RandomForestClassifier(random_state=seed)), RF_GRID,
                           cv=cv, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
    rf_best = rf_grid.best_estimator_
    tuned_rf = {**scores(rf_best, Xte, yte), "best_params": rf_grid.best_params_, "cv_best": rf_grid.best_score_}
    print(f"    best params {rf_grid.best_params_}")
    print(f"    tuned RF   acc={tuned_rf['Accuracy']:.3f}  f1={tuned_rf['F1-Score']:.3f}  "
          f"auc={tuned_rf['ROC-AUC']:.3f}  (cv auc={rf_grid.best_score_:.3f})")

    top = results.sort_values("CV ROC-AUC", ascending=False).iloc[0]["Model"]
    print(f"[7] Best model by 5-fold CV ROC-AUC: {top}. Tuning C + class_weight ...")
    lr_grid = GridSearchCV(make_pipeline(LogisticRegression(max_iter=2000)), LR_GRID,
                           cv=cv, scoring="f1", n_jobs=-1).fit(Xtr, ytr)
    final = lr_grid.best_estimator_
    final_scores = {"model": top, **scores(final, Xte, yte), "best_params": lr_grid.best_params_}
    print(f"    best params {lr_grid.best_params_}")
    print(f"    FINAL      acc={final_scores['Accuracy']:.3f}  prec={final_scores['Precision']:.3f}  "
          f"rec={final_scores['Recall']:.3f}  f1={final_scores['F1-Score']:.3f}  "
          f"auc={final_scores['ROC-AUC']:.3f}")

    names = [c.split("__")[1] for c in rf_best.named_steps["prep"].get_feature_names_out()]
    importance = pd.Series(rf_best.named_steps["model"].feature_importances_, index=names) \
        .sort_values(ascending=False).head(10)

    y_pred = final.predict(Xte)
    if make_plots:
        plots.model_comparison(results, FIGURES_DIR)
        plots.confusion_and_roc(final, y_pred, {"Gradient Boosting": fitted["Gradient Boosting"],
                                                "Neural Network (MLP)": fitted["Neural Network (MLP)"],
                                                "Tuned Random Forest": rf_best}, Xte, yte, FIGURES_DIR)
        plots.feature_importance(importance, FIGURES_DIR)

    joblib.dump(final, MODEL_PATH)
    report = classification_report(yte, y_pred, target_names=["No repeat", "Repeat"])
    summary = {"cleaning": log, "n_train": len(Xtr), "n_test": len(Xte), "positive_rate": float(y.mean()),
               "results": results.to_dict("records"), "tuned_rf": tuned_rf, "final": final_scores,
               "top_features": importance.round(4).to_dict(), "classification_report": report,
               "runtime_s": time.time() - t0}
    with open(METRICS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print("[8] Classification report (final model):")
    print(report)
    print(f"[9] Model saved -> {MODEL_PATH.relative_to(MODEL_PATH.parents[1])}   ({time.time() - t0:.1f}s total)")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Train the repeat-purchase prediction model.")
    ap.add_argument("--n", type=int, default=5000, help="number of customers to generate (default 5000)")
    ap.add_argument("--seed", type=int, default=SEED, help="random seed (default 42)")
    ap.add_argument("--no-plots", action="store_true", help="skip writing figures")
    args = ap.parse_args()
    run(args.n, args.seed, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
