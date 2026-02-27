"""
predict.py

Loads trained model artifacts and test features, generates weighted
ensemble predictions using the same weights that were used during
threshold optimisation in train.py, and saves a versioned submission CSV.

Run from the project root:
    python src/predict.py
"""

import random
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "ID"
TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}


# ── Threshold application ─────────────────────────────────────────────────────

def apply_thresholds(probs, thresholds):
    """High checked first (rarest class priority). Mirrors train.py exactly."""
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


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading model artifacts...")
    with open(MODELS_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    models_lgb = artifacts["models_lgb"]
    models_xgb = artifacts["models_xgb"]
    feature_cols = artifacts["feature_cols"]
    best_thresholds = artifacts["best_thresholds"]
    best_w_lgb = artifacts["best_w_lgb"]
    best_w_xgb = artifacts["best_w_xgb"]
    oof_f1 = artifacts["oof_f1"]

    print(f"  OOF macro F1 from training: {oof_f1:.4f}")
    print(f"  Ensemble weights: LGB={best_w_lgb:.2f}, XGB={best_w_xgb:.2f}")
    print(f"  Thresholds: { {k: round(v, 3) for k, v in best_thresholds.items()} }")

    print("\nLoading test features...")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    print(f"  Shape: {test.shape}")

    test_ids = test[ID_COL]
    X_test = test[feature_cols]
    print(f"  Feature columns aligned: {X_test.shape[1]}")

    # Average predictions across folds within each model type
    print("\nGenerating predictions...")
    test_probs_lgb = np.zeros((len(X_test), 3))
    test_probs_xgb = np.zeros((len(X_test), 3))

    for model in models_lgb:
        test_probs_lgb += model.predict_proba(X_test)
    test_probs_lgb /= len(models_lgb)

    for model in models_xgb:
        test_probs_xgb += model.predict_proba(X_test)
    test_probs_xgb /= len(models_xgb)

    # Use the exact same weights that were used in threshold optimisation
    # This is the key consistency fix - thresholds and ensemble weights are locked together
    test_probs_ensemble = best_w_lgb * test_probs_lgb + best_w_xgb * test_probs_xgb

    # Apply optimised thresholds
    test_preds = apply_thresholds(test_probs_ensemble, best_thresholds)
    test_labels = [TARGET_INVERSE[p] for p in test_preds]

    submission = pd.DataFrame({
        ID_COL: test_ids.values,
        "Target": test_labels,
    })

    print("\nPrediction distribution:")
    dist = submission["Target"].value_counts()
    for label in ["Low", "Medium", "High"]:
        count = dist.get(label, 0)
        pct = count / len(submission) * 100
        print(f"  {label:<8} {count:>5}  ({pct:.1f}%)")

    # Version by OOF score
    output_path = SUBMISSIONS_DIR / f"submission_oof{oof_f1:.4f}.csv"
    submission.to_csv(output_path, index=False)

    # Validate format
    sample = pd.read_csv("data/raw/SampleSubmission.csv")
    assert len(submission) == len(sample), \
        f"Row count mismatch: got {len(submission)}, expected {len(sample)}"
    assert set(submission["Target"].unique()).issubset({"Low", "Medium", "High"}), \
        "Unexpected label values in submission"
    assert list(submission.columns) == ["ID", "Target"], \
        f"Column mismatch: {submission.columns.tolist()}"

    print(f"\nSaved: {output_path}")
    print(f"Total rows: {len(submission)}")
    print("Submission format validated against SampleSubmission.csv")


if __name__ == "__main__":
    main()