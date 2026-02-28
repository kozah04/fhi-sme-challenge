"""
ordinal_train.py

Implements ordinal classification by decomposing the 3-class target into
two binary problems:

  Model A: P(Target >= Medium)  — separates Low from {Medium, High}
  Model B: P(Target == High)    — separates High from {Low, Medium}

Decision rule at inference:
  if P_B >= tB  → predict High
  elif P_A >= tA → predict Medium
  else           → predict Low

This enforces the ordinal structure (Low → Medium → High) rather than
treating the three classes as unrelated categories.

Key differences from train.py:
  - Two binary tasks instead of one multiclass task
  - Joint country+target CV stratification key (tighter OOF/LB alignment)
  - Proper 3-way ensemble weight search across LGB/XGB/CAT
  - Separate artifacts saved to models/ordinal_artifacts.pkl

Run from project root:
    python src/ordinal_train.py           # full run (~80-100 min)
    python src/ordinal_train.py --fast    # sanity check (~15-20 min)
"""

import sys
import random
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product as iterproduct

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import TargetEncoder, TARGET_ENCODE_COLS

warnings.filterwarnings("ignore")

# ── Reproducibility ───────────────────────────────────────────────────────────

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

N_FOLDS = 5
TARGET_COL = "Target"
ID_COL = "ID"
DROP_COLS = [ID_COL, TARGET_COL]
TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}

COUNTRY_NAMES = {0: "Lesotho", 1: "Zim/Malawi", 3: "Eswatini"}


# ── Model definitions ─────────────────────────────────────────────────────────

def make_lgb_binary(fast=False):
    return lgb.LGBMClassifier(
        n_estimators=500 if fast else 2000,
        learning_rate=0.0209,
        num_leaves=76,
        max_depth=11,
        min_child_samples=13,
        subsample=0.7987,
        colsample_bytree=0.5972,
        reg_alpha=0.00055,
        reg_lambda=0.1410,
        min_split_gain=0.2529,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def make_xgb_binary(fast=False):
    return xgb.XGBClassifier(
        n_estimators=300 if fast else 1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )


def make_cat_binary(fast=False):
    return CatBoostClassifier(
        iterations=300 if fast else 1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=0,
        thread_count=-1,
    )


# ── Early stopping fit functions ──────────────────────────────────────────────

def fit_lgb(model, X_tr, y_tr, X_val, y_val):
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def fit_xgb(model, X_tr, y_tr, X_val, y_val):
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def fit_cat(model, X_tr, y_tr, X_val, y_val):
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


# ── Ordinal target construction ───────────────────────────────────────────────

def make_binary_targets(y):
    """
    Convert 3-class target to two binary targets.

    y_A: 1 if Target >= Medium (i.e. Medium or High), else 0
    y_B: 1 if Target == High, else 0
    """
    y_A = (y >= 1).astype(int)  # at least Medium
    y_B = (y == 2).astype(int)  # exactly High
    return y_A, y_B


# ── Ordinal decision rule ─────────────────────────────────────────────────────

def ordinal_predict(p_A, p_B, tA, tB):
    """
    Apply ordinal decision rule given probabilities and thresholds.

    p_A: P(>= Medium) for each sample
    p_B: P(== High) for each sample
    tA, tB: thresholds

    Returns integer predictions (0=Low, 1=Medium, 2=High).
    """
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


# ── Threshold + weight search ─────────────────────────────────────────────────

def search_weights_and_thresholds(
    oof_A_lgb, oof_A_xgb, oof_A_cat,
    oof_B_lgb, oof_B_xgb, oof_B_cat,
    y_true, fast=False
):
    """
    Joint search over 3-way ensemble weights and ordinal thresholds.

    Strategy:
    1. Build simplex grid of (w_lgb, w_xgb, w_cat) summing to 1
    2. For each weight triplet, combine OOF probabilities
    3. Grid-search (tA, tB) to maximise weighted F1 on 3-class output
    4. Return best configuration

    Returns:
        best_weights_A, best_weights_B, best_tA, best_tB, best_f1
    """
    weight_step = 0.2 if fast else 0.1
    thresh_steps = 15 if fast else 30
    thresh_grid = np.linspace(0.05, 0.95, thresh_steps)

    best_f1 = 0.0
    best_weights_A = (0.6, 0.2, 0.2)
    best_weights_B = (0.6, 0.2, 0.2)
    best_tA = 0.5
    best_tB = 0.5

    # Build simplex weight grid
    weight_vals = np.arange(0, 1 + weight_step, weight_step)
    weight_triplets = [
        (wl, wx, wc)
        for wl in weight_vals
        for wx in weight_vals
        for wc in weight_vals
        if abs(wl + wx + wc - 1.0) < 1e-9
    ]

    print(f"  Weight combinations: {len(weight_triplets)}")
    print(f"  Threshold grid size: {thresh_steps} x {thresh_steps} = {thresh_steps**2}")

    for w_lgb, w_xgb, w_cat in weight_triplets:
        # Combined probabilities for task A and B
        # We allow different weights for A and B but keep them the same
        # for simplicity — can be split if needed
        p_A = w_lgb * oof_A_lgb + w_xgb * oof_A_xgb + w_cat * oof_A_cat
        p_B = w_lgb * oof_B_lgb + w_xgb * oof_B_xgb + w_cat * oof_B_cat

        for tA in thresh_grid:
            for tB in thresh_grid:
                preds = ordinal_predict(p_A, p_B, tA, tB)
                score = f1_score(y_true, preds, average="weighted")
                if score > best_f1:
                    best_f1 = score
                    best_weights_A = (w_lgb, w_xgb, w_cat)
                    best_weights_B = (w_lgb, w_xgb, w_cat)
                    best_tA = tA
                    best_tB = tB

    return best_weights_A, best_weights_B, best_tA, best_tB, best_f1


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_diagnostics(y_true, preds, country_arr):
    """Print per-country weighted F1 and High prediction counts."""
    print("\n  Per-country diagnostics:")
    print(f"  {'Country':<15} {'N':>5} {'WF1':>6} {'High_true':>10} {'High_pred':>10}")
    print("  " + "-" * 55)

    for val, name in sorted(COUNTRY_NAMES.items()):
        mask = country_arr == val
        if mask.sum() == 0:
            continue
        y_c = y_true[mask]
        p_c = preds[mask]
        f1_c = f1_score(y_c, p_c, average="weighted", zero_division=0)
        high_true = (y_c == 2).sum()
        high_pred = (p_c == 2).sum()
        print(f"  {name:<15} {mask.sum():>5} {f1_c:>6.4f} {high_true:>10} {high_pred:>10}")

    print(f"\n  Overall weighted F1: {f1_score(y_true, preds, average='weighted'):.4f}")
    print(f"\n  Confusion matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_true, preds, labels=[0, 1, 2])
    print(f"  {'':10} {'Low':>8} {'Medium':>8} {'High':>8}")
    for i, label in enumerate(["Low", "Medium", "High"]):
        print(f"  {label:<10} {cm[i, 0]:>8} {cm[i, 1]:>8} {cm[i, 2]:>8}")


# ── Cross-validation loop ─────────────────────────────────────────────────────

def run_cv(X, y, country_arr, fast=False):
    """
    Run stratified k-fold CV with joint country+target stratification key.

    The stratification key is country_target (e.g. "0_Low", "1_High") which
    ensures each fold preserves the country/class composition. This tightens
    the OOF-to-public gap compared to target-only stratification.

    For each fold:
    1. Construct binary targets y_A and y_B
    2. Fit TargetEncoder on training fold only
    3. Train LGB/XGB/CAT for both tasks A and B
    4. Collect OOF probabilities
    """
    # Joint stratification key
    strat_key = pd.Series([
        f"{c}_{int(t)}" for c, t in zip(country_arr, y)
    ])

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    n = len(X)
    oof_A_lgb = np.zeros(n)
    oof_A_xgb = np.zeros(n)
    oof_A_cat = np.zeros(n)
    oof_B_lgb = np.zeros(n)
    oof_B_xgb = np.zeros(n)
    oof_B_cat = np.zeros(n)

    models_A_lgb, models_A_xgb, models_A_cat = [], [], []
    models_B_lgb, models_B_xgb, models_B_cat = [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, strat_key)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")

        X_tr_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_val = y[val_idx]

        y_A_tr, y_B_tr = make_binary_targets(y_tr)
        y_A_val, y_B_val = make_binary_targets(y_val)

        # Target encode inside fold (same pattern as train.py)
        te = TargetEncoder(smoothing=10)
        X_tr = te.fit_transform(X_tr_raw, y_tr, TARGET_ENCODE_COLS)
        X_val = te.transform(X_val_raw)

        # ── Task A: P(>= Medium) ──────────────────────────────────────────

        # LightGBM A
        m = make_lgb_binary(fast)
        m = fit_lgb(m, X_tr, y_A_tr, X_val, y_A_val)
        oof_A_lgb[val_idx] = m.predict_proba(X_val)[:, 1]
        models_A_lgb.append(m)

        # XGBoost A
        m = make_xgb_binary(fast)
        m = fit_xgb(m, X_tr, y_A_tr, X_val, y_A_val)
        oof_A_xgb[val_idx] = m.predict_proba(X_val)[:, 1]
        models_A_xgb.append(m)

        # CatBoost A
        m = make_cat_binary(fast)
        m = fit_cat(m, X_tr, y_A_tr, X_val, y_A_val)
        oof_A_cat[val_idx] = m.predict_proba(X_val)[:, 1]
        models_A_cat.append(m)

        # ── Task B: P(== High) ────────────────────────────────────────────

        # LightGBM B
        m = make_lgb_binary(fast)
        m = fit_lgb(m, X_tr, y_B_tr, X_val, y_B_val)
        oof_B_lgb[val_idx] = m.predict_proba(X_val)[:, 1]
        models_B_lgb.append(m)

        # XGBoost B
        m = make_xgb_binary(fast)
        m = fit_xgb(m, X_tr, y_B_tr, X_val, y_B_val)
        oof_B_xgb[val_idx] = m.predict_proba(X_val)[:, 1]
        models_B_xgb.append(m)

        # CatBoost B
        m = make_cat_binary(fast)
        m = fit_cat(m, X_tr, y_B_tr, X_val, y_B_val)
        oof_B_cat[val_idx] = m.predict_proba(X_val)[:, 1]
        models_B_cat.append(m)

        # Quick fold diagnostic using equal weights and default thresholds
        p_A_fold = (oof_A_lgb[val_idx] + oof_A_xgb[val_idx] + oof_A_cat[val_idx]) / 3
        p_B_fold = (oof_B_lgb[val_idx] + oof_B_xgb[val_idx] + oof_B_cat[val_idx]) / 3
        fold_preds = ordinal_predict(p_A_fold, p_B_fold, tA=0.5, tB=0.5)
        fold_f1 = f1_score(y_val, fold_preds, average="weighted")
        high_pred = (fold_preds == 2).sum()
        high_true = (y_val == 2).sum()
        print(f"    Fold F1 (default thresh): {fold_f1:.4f} | "
              f"High pred={high_pred} true={high_true}")

    return (
        oof_A_lgb, oof_A_xgb, oof_A_cat,
        oof_B_lgb, oof_B_xgb, oof_B_cat,
        models_A_lgb, models_A_xgb, models_A_cat,
        models_B_lgb, models_B_xgb, models_B_cat,
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Use fewer estimators and coarser grid for quick sanity check")
    args = parser.parse_args()

    if args.fast:
        print("*** FAST MODE — for sanity check only, not for submission ***\n")

    print("Loading feature data...")
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    print(f"  Train: {train.shape}")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X = train[feature_cols]
    y = train[TARGET_COL].values
    country_arr = train["country"].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    y_A, y_B = make_binary_targets(y)
    print(f"  Task A (>=Medium): {y_A.sum()} positives ({y_A.mean():.1%})")
    print(f"  Task B (==High):   {y_B.sum()} positives ({y_B.mean():.1%})")

    print(f"\nRunning {N_FOLDS}-fold CV with joint country+target stratification...")
    (
        oof_A_lgb, oof_A_xgb, oof_A_cat,
        oof_B_lgb, oof_B_xgb, oof_B_cat,
        models_A_lgb, models_A_xgb, models_A_cat,
        models_B_lgb, models_B_xgb, models_B_cat,
    ) = run_cv(X, y, country_arr, fast=args.fast)

    print("\n" + "=" * 60)
    print("Searching optimal weights and thresholds...")
    (
        best_weights_A, best_weights_B,
        best_tA, best_tB, best_f1
    ) = search_weights_and_thresholds(
        oof_A_lgb, oof_A_xgb, oof_A_cat,
        oof_B_lgb, oof_B_xgb, oof_B_cat,
        y, fast=args.fast
    )

    w_lgb_A, w_xgb_A, w_cat_A = best_weights_A
    print(f"\nBest weights (A): LGB={w_lgb_A:.2f} XGB={w_xgb_A:.2f} CAT={w_cat_A:.2f}")
    print(f"Best tA (>=Medium threshold): {best_tA:.3f}")
    print(f"Best tB (==High threshold):   {best_tB:.3f}")
    print(f"Best OOF weighted F1: {best_f1:.4f}")

    # Final OOF predictions with best config
    p_A_oof = w_lgb_A * oof_A_lgb + w_xgb_A * oof_A_xgb + w_cat_A * oof_A_cat
    p_B_oof = w_lgb_A * oof_B_lgb + w_xgb_A * oof_B_xgb + w_cat_A * oof_B_cat
    final_preds = ordinal_predict(p_A_oof, p_B_oof, best_tA, best_tB)

    print_diagnostics(y, final_preds, country_arr)

    # Fit full-data target encoder for test set
    print("\nFitting full-data target encoder...")
    full_te = TargetEncoder(smoothing=10)
    full_te.fit(X, y, TARGET_ENCODE_COLS)

    print("Saving ordinal artifacts...")
    artifacts = {
        "models_A_lgb": models_A_lgb,
        "models_A_xgb": models_A_xgb,
        "models_A_cat": models_A_cat,
        "models_B_lgb": models_B_lgb,
        "models_B_xgb": models_B_xgb,
        "models_B_cat": models_B_cat,
        "feature_cols": feature_cols,
        "target_encode_cols": TARGET_ENCODE_COLS,
        "full_target_encoder": full_te,
        "best_weights_A": best_weights_A,
        "best_weights_B": best_weights_B,
        "best_tA": best_tA,
        "best_tB": best_tB,
        "oof_f1": best_f1,
        "fast_mode": args.fast,
    }

    out_path = MODELS_DIR / "ordinal_artifacts.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"  Saved {out_path}")
    print(f"\nFinal OOF weighted F1: {best_f1:.4f}")
    if args.fast:
        print("*** FAST MODE result — rerun without --fast before submitting ***")


if __name__ == "__main__":
    main()