"""
tune_thresholds.py

Searches for per-country probability thresholds that maximise weighted F1.
Uses existing OOF predictions from artifacts.pkl - no retraining required.

The insight: a global High threshold is suboptimal because the class
distribution is radically different per country:
  - Eswatini:       11.5% High  -> lower threshold appropriate
  - Zimbabwe/Malawi: 3.1% High  -> medium threshold
  - Lesotho:         0.3% High  -> very high threshold (effectively suppress High)

Under weighted F1, the ~6 Lesotho High cases are mathematically negligible.
Getting Low/Medium right in Lesotho (1,938 cases) is worth far more.

Run from project root:
    python src/tune_thresholds.py
"""

import sys
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import TargetEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")
SUBMISSIONS_DIR = Path("submissions")
SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}

# Country encoding values from preprocess.py
COUNTRY_LESOTHO = 0
COUNTRY_ZIM_MALAWI = 1
COUNTRY_ESWATINI = 3


def apply_thresholds_global(probs, thresholds):
    """Standard global threshold application."""
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


def apply_thresholds_by_country(probs, country_arr, country_thresholds):
    """
    Apply different thresholds per country.

    country_thresholds: dict mapping country_value -> {1: t_med, 2: t_high}
    """
    n = len(probs)
    preds = np.zeros(n, dtype=int)
    for i in range(n):
        c = country_arr[i]
        t = country_thresholds.get(c, country_thresholds["default"])
        if probs[i, 2] >= t[2]:
            preds[i] = 2
        elif probs[i, 1] >= t[1]:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def search_country_thresholds(oof_probs, y_true, country_arr, n_steps=30):
    """
    Search for optimal per-country thresholds maximising weighted F1.

    Strategy:
    - Lesotho: fix High threshold very high (0.90+) since 6 High cases
      are negligible under weighted F1 and chasing them hurts Low/Medium
    - Eswatini: search freely since 11.5% High rate means real signal
    - Zimbabwe/Malawi: search with moderate range

    Returns best thresholds dict and best F1.
    """
    grid = np.linspace(0.05, 0.95, n_steps)
    high_grid = np.linspace(0.3, 0.95, n_steps)

    best_f1 = 0.0
    best_thresholds = None

    # Baseline: global thresholds
    for t2 in grid:
        for t1 in grid:
            ct = {
                COUNTRY_LESOTHO:    {1: t1, 2: 0.95},  # suppress High in Lesotho
                COUNTRY_ZIM_MALAWI: {1: t1, 2: t2},
                COUNTRY_ESWATINI:   {1: t1, 2: t2},
                "default":          {1: t1, 2: t2},
            }
            preds = apply_thresholds_by_country(oof_probs, country_arr, ct)
            score = f1_score(y_true, preds, average="weighted")
            if score > best_f1:
                best_f1 = score
                best_thresholds = {
                    "lesotho":    {1: t1, 2: 0.95},
                    "zim_malawi": {1: t1, 2: t2},
                    "eswatini":   {1: t1, 2: t2},
                }

    return best_thresholds, best_f1


def thresholds_to_country_arr_format(best_thresholds):
    """Convert named thresholds to country-value-keyed dict for prediction."""
    return {
        COUNTRY_LESOTHO:    best_thresholds["lesotho"],
        COUNTRY_ZIM_MALAWI: best_thresholds["zim_malawi"],
        COUNTRY_ESWATINI:   best_thresholds["eswatini"],
        "default":          best_thresholds["zim_malawi"],
    }


def main():
    print("Loading artifacts...")
    with open(MODELS_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    oof_probs = artifacts["oof_probs_ensemble"]
    best_w_lgb = artifacts["best_w_lgb"]
    best_w_xgb = artifacts["best_w_xgb"]
    old_thresholds = artifacts["best_thresholds"]
    old_f1 = artifacts["oof_f1"]
    feature_cols = artifacts["feature_cols"]
    full_te = artifacts.get("full_target_encoder")
    target_encode_cols = artifacts.get("target_encode_cols", [])

    print(f"  Current OOF weighted F1: {old_f1:.4f}")
    print(f"  Current thresholds: { {k: round(v, 3) for k, v in old_thresholds.items()} }")

    # Load train data for OOF evaluation
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    y = train["Target"].values
    country_arr = train["country"].values

    print(f"\n  Country distribution in train:")
    for val, name in [(0, "Lesotho"), (1, "Zim/Malawi"), (3, "Eswatini")]:
        mask = country_arr == val
        high_rate = (y[mask] == 2).mean()
        print(f"    {name:<15} n={mask.sum()}  High={high_rate:.3f}")

    # Baseline: global thresholds on current ensemble
    baseline_preds = apply_thresholds_global(oof_probs, old_thresholds)
    baseline_f1 = f1_score(y, baseline_preds, average="weighted")
    print(f"\nBaseline (global thresholds): {baseline_f1:.4f}")

    # Search country-calibrated thresholds
    print("\nSearching per-country thresholds...")
    print("  (Lesotho High threshold fixed at 0.95 - negligible under weighted F1)")
    best_thresholds, best_f1 = search_country_thresholds(
        oof_probs, y, country_arr, n_steps=30
    )

    print(f"\nBest per-country thresholds:")
    for country, t in best_thresholds.items():
        print(f"  {country:<15} Medium>={t[1]:.3f}  High>={t[2]:.3f}")
    print(f"Best OOF weighted F1: {best_f1:.4f}")
    print(f"Improvement: {best_f1 - baseline_f1:+.4f}")

    if best_f1 <= baseline_f1:
        print("\nNo improvement from country thresholds. Keeping global thresholds.")
        return

    # Generate submission with country-calibrated thresholds
    print("\nGenerating submission with country-calibrated thresholds...")

    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    test_ids = test["ID"]
    X_test = test[feature_cols]
    test_country = test["country"].values

    if full_te is not None and target_encode_cols:
        X_test = full_te.transform(X_test)

    models_lgb = artifacts["models_lgb"]
    models_xgb = artifacts["models_xgb"]
    models_cat = artifacts["models_cat"]

    test_probs_lgb = np.zeros((len(X_test), 3))
    test_probs_xgb = np.zeros((len(X_test), 3))
    test_probs_cat = np.zeros((len(X_test), 3))

    for model in models_lgb:
        test_probs_lgb += model.predict_proba(X_test)
    test_probs_lgb /= len(models_lgb)

    for model in models_xgb:
        test_probs_xgb += model.predict_proba(X_test)
    test_probs_xgb /= len(models_xgb)

    for model in models_cat:
        test_probs_cat += model.predict_proba(X_test)
    test_probs_cat /= len(models_cat)

    lgbxgb = best_w_lgb * test_probs_lgb + best_w_xgb * test_probs_xgb
    test_probs_ensemble = 0.7 * lgbxgb + 0.3 * test_probs_cat

    country_threshold_map = thresholds_to_country_arr_format(best_thresholds)
    test_preds = apply_thresholds_by_country(
        test_probs_ensemble, test_country, country_threshold_map
    )
    test_labels = [TARGET_INVERSE[p] for p in test_preds]

    submission = pd.DataFrame({"ID": test_ids.values, "Target": test_labels})

    print("\nPrediction distribution:")
    dist = submission["Target"].value_counts()
    for label in ["Low", "Medium", "High"]:
        count = dist.get(label, 0)
        pct = count / len(submission) * 100
        print(f"  {label:<8} {count:>5}  ({pct:.1f}%)")

    # Check Lesotho predictions specifically
    lesotho_mask = test_country == COUNTRY_LESOTHO
    lesotho_preds = submission["Target"][lesotho_mask].value_counts()
    print(f"\n  Lesotho predictions: {lesotho_preds.to_dict()}")

    output_path = SUBMISSIONS_DIR / f"submission_country_thresh_oof{best_f1:.4f}.csv"
    submission.to_csv(output_path, index=False)

    sample = pd.read_csv("data/raw/SampleSubmission.csv")
    assert len(submission) == len(sample)
    assert set(submission["Target"].unique()).issubset({"Low", "Medium", "High"})
    assert list(submission.columns) == ["ID", "Target"]

    print(f"\nSaved: {output_path}")
    print("Submission validated.")

    # Save updated thresholds back to artifacts
    artifacts["best_thresholds_country"] = best_thresholds
    artifacts["country_f1"] = best_f1
    with open(MODELS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print("Updated artifacts saved.")


if __name__ == "__main__":
    main()