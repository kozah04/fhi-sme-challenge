"""
predict.py

Loads trained model artifacts and test features, generates ensemble
predictions, applies optimised thresholds, and saves a submission CSV.

Run from the project root:
    python src/predict.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────

ID_COL = "ID"
TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}


# ── Threshold application ─────────────────────────────────────────────────────

def apply_thresholds(probs, thresholds):
    """
    Apply per-class thresholds to probability matrix.
    Decision order: High first, then Medium, then Low as default.
    Mirrors the logic in train.py exactly.
    """
    n = len(probs)
    preds = np.zeros(n, dtype=int)

    for i in range(n):
        if probs[i, 2] >= thresholds[2]:
            preds[i] = 2
        elif probs[i, 1] >= thresholds[1]:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds


# ── Weighted ensemble ─────────────────────────────────────────────────────────

def weighted_ensemble(probs_lgb, probs_xgb, probs_cat,
                      w_lgb=0.40, w_xgb=0.40, w_cat=0.20):
    """
    Weighted average of three model probability matrices.

    LightGBM and XGBoost performed similarly (0.807 and 0.804 OOF F1).
    CatBoost lagged at 0.765. We give LGB and XGB equal weight at 0.40
    each and CatBoost 0.20 rather than the equal 0.33 used during training.
    This should improve over the simple average.

    Weights sum to 1.0 by construction.
    """
    return w_lgb * probs_lgb + w_xgb * probs_xgb + w_cat * probs_cat


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Load artifacts saved by train.py
    print("Loading model artifacts...")
    with open(MODELS_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    models_lgb = artifacts["models_lgb"]
    models_xgb = artifacts["models_xgb"]
    models_cat = artifacts["models_cat"]
    feature_cols = artifacts["feature_cols"]
    best_thresholds = artifacts["best_thresholds"]
    oof_f1 = artifacts["oof_f1"]

    print(f"  OOF macro F1 from training: {oof_f1:.4f}")
    print(f"  Thresholds: { {k: round(v, 3) for k, v in best_thresholds.items()} }")
    print(f"  Models per type: {len(models_lgb)} folds")

    # Load test features
    print("\nLoading test features...")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    print(f"  Shape: {test.shape}")

    # Keep ID for submission, extract feature matrix
    test_ids = test[ID_COL]
    X_test = test[feature_cols]

    print(f"  Feature columns aligned: {X_test.shape[1]}")

    # Generate predictions from each fold model
    # Average across folds within each model type, then weighted ensemble
    print("\nGenerating predictions...")

    test_probs_lgb = np.zeros((len(X_test), 3))
    test_probs_xgb = np.zeros((len(X_test), 3))
    test_probs_cat = np.zeros((len(X_test), 3))

    for i, model in enumerate(models_lgb):
        test_probs_lgb += model.predict_proba(X_test)
    test_probs_lgb /= len(models_lgb)

    for i, model in enumerate(models_xgb):
        test_probs_xgb += model.predict_proba(X_test)
    test_probs_xgb /= len(models_xgb)

    for i, model in enumerate(models_cat):
        test_probs_cat += model.predict_proba(X_test)
    test_probs_cat /= len(models_cat)

    # Weighted ensemble
    test_probs_ensemble = weighted_ensemble(
        test_probs_lgb, test_probs_xgb, test_probs_cat
    )

    # Apply optimised thresholds
    test_preds = apply_thresholds(test_probs_ensemble, best_thresholds)

    # Map back to string labels
    test_labels = [TARGET_INVERSE[p] for p in test_preds]

    # Build submission dataframe
    submission = pd.DataFrame({
        ID_COL: test_ids.values,
        "Target": test_labels,
    })

    # Verify submission format matches sample
    print("\nPrediction distribution:")
    dist = submission["Target"].value_counts()
    for label in ["Low", "Medium", "High"]:
        count = dist.get(label, 0)
        pct = count / len(submission) * 100
        print(f"  {label:<8} {count:>5}  ({pct:.1f}%)")

    # Version the submission file by OOF score
    version = f"submission_oof{oof_f1:.4f}"
    output_path = SUBMISSIONS_DIR / f"{version}.csv"
    submission.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print(f"Total rows: {len(submission)}")

    # Sanity check against sample submission
    sample = pd.read_csv("data/raw/SampleSubmission.csv")
    assert len(submission) == len(sample), (
        f"Row count mismatch: got {len(submission)}, expected {len(sample)}"
    )
    assert set(submission["Target"].unique()).issubset({"Low", "Medium", "High"}), (
        "Unexpected label values in submission"
    )
    assert list(submission.columns) == ["ID", "Target"], (
        f"Column mismatch: {submission.columns.tolist()}"
    )
    print("Submission format validated against SampleSubmission.csv")


if __name__ == "__main__":
    main()