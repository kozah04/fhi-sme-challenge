"""
diagnose.py

Analyses OOF predictions to understand exactly where macro F1 is being
lost. Run after train.py to get a breakdown by class and by country.

Run from the project root:
    python src/diagnose.py
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")


def apply_thresholds(probs, thresholds):
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


def main():
    print("Loading artifacts and train data...")
    with open(MODELS_DIR / "artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    oof_probs = artifacts["oof_probs_ensemble"]
    thresholds = artifacts["best_thresholds"]

    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    y_true = train["Target"].values
    oof_preds = apply_thresholds(oof_probs, thresholds)

    label_names = ["Low", "Medium", "High"]

    # ── Per-class F1 breakdown ────────────────────────────────────────────────
    print("\n=== Per-class F1 breakdown ===")
    print(classification_report(y_true, oof_preds, target_names=label_names))

    # ── Confusion matrix ──────────────────────────────────────────────────────
    print("=== Confusion matrix (rows=actual, cols=predicted) ===")
    cm = confusion_matrix(y_true, oof_preds)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm_df)
    print()

    # ── Where are High predictions going wrong? ───────────────────────────────
    print("=== High class analysis ===")
    high_mask = y_true == 2
    high_probs = oof_probs[high_mask, 2]
    print(f"  True High samples: {high_mask.sum()}")
    print(f"  Correctly predicted High: {((oof_preds == 2) & high_mask).sum()}")
    print(f"  Missed High (predicted Low):    {((oof_preds == 0) & high_mask).sum()}")
    print(f"  Missed High (predicted Medium): {((oof_preds == 1) & high_mask).sum()}")
    print(f"\n  High probability stats for TRUE High samples:")
    print(f"    mean:   {high_probs.mean():.4f}")
    print(f"    median: {np.median(high_probs):.4f}")
    print(f"    min:    {high_probs.min():.4f}")
    print(f"    max:    {high_probs.max():.4f}")
    print(f"    pct above threshold ({thresholds[2]:.3f}): "
          f"{(high_probs >= thresholds[2]).mean():.3f}")

    # Prob distribution of missed High samples
    missed_high_probs = oof_probs[(y_true == 2) & (oof_preds != 2), 2]
    if len(missed_high_probs) > 0:
        print(f"\n  High prob of MISSED High samples:")
        print(f"    mean: {missed_high_probs.mean():.4f}")
        print(f"    max:  {missed_high_probs.max():.4f}")
        print(f"    pct between 0.3-0.69: "
              f"{((missed_high_probs >= 0.3) & (missed_high_probs < thresholds[2])).mean():.3f}")

    # ── False positives: what is being predicted as High incorrectly? ─────────
    print("\n=== False High predictions (predicted High but actually...) ===")
    false_high_mask = (oof_preds == 2) & (y_true != 2)
    print(f"  Total false High predictions: {false_high_mask.sum()}")
    if false_high_mask.sum() > 0:
        false_high_true = y_true[false_high_mask]
        print(f"  Actually Low:    {(false_high_true == 0).sum()}")
        print(f"  Actually Medium: {(false_high_true == 1).sum()}")

    # ── Per-country breakdown ─────────────────────────────────────────────────
    print("\n=== Per-country F1 breakdown ===")
    country_col = train["country"]
    country_map = {3: "eswatini", 1: "zimbabwe/malawi", 0: "lesotho"}

    for country_val, country_name in sorted(country_map.items()):
        mask = country_col == country_val
        if mask.sum() == 0:
            continue
        y_c = y_true[mask]
        p_c = oof_preds[mask]
        f1 = f1_score(y_c, p_c, average="macro")
        class_counts = {label_names[i]: (y_c == i).sum() for i in range(3)}
        print(f"  {country_name:<20} F1: {f1:.4f}  n={mask.sum()}  {class_counts}")

    # ── Medium vs Low boundary ────────────────────────────────────────────────
    print("\n=== Medium/Low boundary analysis ===")
    low_mask = y_true == 0
    med_mask = y_true == 1

    false_low = ((oof_preds == 0) & med_mask).sum()
    false_med = ((oof_preds == 1) & low_mask).sum()
    print(f"  True Medium predicted as Low:    {false_low} "
          f"({false_low/med_mask.sum():.3f} of all Medium)")
    print(f"  True Low predicted as Medium:    {false_med} "
          f"({false_med/low_mask.sum():.3f} of all Low)")

    # ── Probability calibration check ────────────────────────────────────────
    print("\n=== Probability ranges by true class ===")
    for cls, name in enumerate(label_names):
        mask = y_true == cls
        print(f"  {name}:")
        for prob_cls, prob_name in enumerate(label_names):
            vals = oof_probs[mask, prob_cls]
            print(f"    P({prob_name}): mean={vals.mean():.3f}  "
                  f"median={np.median(vals):.3f}  "
                  f"max={vals.max():.3f}")


if __name__ == "__main__":
    main()