"""
ordinal_predict.py

Loads ordinal model artifacts and generates a submission using the
ordinal decision rule:

  if P_B >= tB  → predict High
  elif P_A >= tA → predict Medium
  else           → predict Low

Run from project root:
    python src/ordinal_predict.py
"""

import sys
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import TargetEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "ID"
TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}
COUNTRY_NAMES = {0: "Lesotho", 1: "Zim/Malawi", 3: "Eswatini"}


def ordinal_predict(p_A, p_B, tA, tB):
    n = len(p_A)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        if p_B[i] >= tB:
            preds[i] = 2
        elif p_A[i] >= tA:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def avg_proba(models, X):
    probs = np.zeros(len(X))
    for m in models:
        probs += m.predict_proba(X)[:, 1]
    return probs / len(models)


def main():
    print("Loading ordinal artifacts...")
    with open(MODELS_DIR / "ordinal_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    feature_cols = artifacts["feature_cols"]
    target_encode_cols = artifacts["target_encode_cols"]
    full_te = artifacts["full_target_encoder"]
    best_weights_A = artifacts["best_weights_A"]
    best_tA = artifacts["best_tA"]
    best_tB = artifacts["best_tB"]
    oof_f1 = artifacts["oof_f1"]
    fast_mode = artifacts.get("fast_mode", False)

    w_lgb, w_xgb, w_cat = best_weights_A

    print(f"  OOF weighted F1: {oof_f1:.4f}")
    print(f"  Weights: LGB={w_lgb:.2f} XGB={w_xgb:.2f} CAT={w_cat:.2f}")
    print(f"  tA (>=Medium): {best_tA:.3f}")
    print(f"  tB (==High):   {best_tB:.3f}")

    if fast_mode:
        print("  WARNING: artifacts were trained in fast mode — not suitable for submission")

    print("\nLoading test features...")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    print(f"  Shape: {test.shape}")

    test_ids = test[ID_COL]
    X_test = test[feature_cols]
    test_country = test["country"].values

    # Apply full-data target encoder
    X_test = full_te.transform(X_test)
    print(f"  After encoding: {X_test.shape[1]} features")

    print("\nGenerating predictions...")
    # Task A probabilities
    p_A = (
        w_lgb * avg_proba(artifacts["models_A_lgb"], X_test) +
        w_xgb * avg_proba(artifacts["models_A_xgb"], X_test) +
        w_cat * avg_proba(artifacts["models_A_cat"], X_test)
    )

    # Task B probabilities
    p_B = (
        w_lgb * avg_proba(artifacts["models_B_lgb"], X_test) +
        w_xgb * avg_proba(artifacts["models_B_xgb"], X_test) +
        w_cat * avg_proba(artifacts["models_B_cat"], X_test)
    )

    test_preds = ordinal_predict(p_A, p_B, best_tA, best_tB)
    test_labels = [TARGET_INVERSE[p] for p in test_preds]

    submission = pd.DataFrame({"ID": test_ids.values, "Target": test_labels})

    print("\nPrediction distribution:")
    for label in ["Low", "Medium", "High"]:
        count = (submission["Target"] == label).sum()
        pct = count / len(submission) * 100
        print(f"  {label:<8} {count:>5}  ({pct:.1f}%)")

    print("\nPer-country High predictions:")
    for val, name in sorted(COUNTRY_NAMES.items()):
        mask = test_country == val
        high = (test_preds[mask] == 2).sum()
        total = mask.sum()
        print(f"  {name:<15} High={high}/{total} ({high/total:.1%})")

    output_path = SUBMISSIONS_DIR / f"submission_ordinal_oof{oof_f1:.4f}.csv"
    submission.to_csv(output_path, index=False)

    sample = pd.read_csv("data/raw/SampleSubmission.csv")
    assert len(submission) == len(sample), \
        f"Row count mismatch: {len(submission)} vs {len(sample)}"
    assert set(submission["Target"].unique()).issubset({"Low", "Medium", "High"})
    assert list(submission.columns) == ["ID", "Target"]

    print(f"\nSaved: {output_path}")
    print("Submission validated.")


if __name__ == "__main__":
    main()