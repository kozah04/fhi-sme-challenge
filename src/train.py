"""
train.py

Trains an ensemble of LightGBM and XGBoost models using stratified
k-fold cross-validation. Handles class imbalance via class weights.
Finds optimal ensemble weights via OOF search. Optimises per-class
probability thresholds post-prediction to maximise macro F1.

CatBoost has been removed from the ensemble after consistent underperformance
(0.768 vs LGB 0.804 and XGB 0.802). Including it was pulling the ensemble
below either individual model. It is retained as a trained artifact in case
it improves in future experiments.

Run from the project root:
    python src/train.py
"""

import random
import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

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
N_CLASSES = 3  # Low=0, Medium=1, High=2
DROP_COLS = [ID_COL, TARGET_COL]


# ── Class weights ─────────────────────────────────────────────────────────────

def get_class_weights(y):
    """
    Compute balanced class weights.

    With Low=65%, Medium=30%, High=5%, an unweighted model will almost
    entirely ignore the High class. Balanced weights inversely scale each
    class by its frequency, forcing the model to treat a High error as
    ~13x more costly than a Low error. This is critical for macro F1.
    """
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def get_sample_weights(y, class_weight_dict):
    """Map per-class weights to per-sample weights for XGBoost."""
    return np.array([class_weight_dict[yi] for yi in y])


# ── Model definitions ─────────────────────────────────────────────────────────

def make_lgb_model(class_weights):
    """
    LightGBM with Optuna-tuned hyperparameters.

    Key changes from defaults:
    - learning_rate lowered to 0.0209 (wants to learn slowly and carefully)
    - n_estimators raised to 2000 to match lower learning rate ceiling
    - colsample_bytree=0.597 (model benefits from feature subsampling,
      suggests some noise in the feature set)
    - min_split_gain=0.253 (suppresses noisy splits)
    Early stopping at 50 rounds controls actual tree count.
    """
    return lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.0209,
        num_leaves=76,
        max_depth=11,
        min_child_samples=13,
        subsample=0.7987,
        colsample_bytree=0.5972,
        reg_alpha=0.00055,
        reg_lambda=0.1410,
        min_split_gain=0.2529,
        class_weight=class_weights,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def make_xgb_model():
    """
    XGBoost classifier with sample weights passed at fit time.
    tree_method='hist' is fastest for CPU.
    """
    return xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
    )


def make_cat_model(class_weights):
    """
    CatBoost classifier. Trained and saved but excluded from ensemble
    due to consistent underperformance vs LGB and XGB.
    """
    weights_list = [class_weights[i] for i in range(N_CLASSES)]
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        class_weights=weights_list,
        random_seed=SEED,
        verbose=0,
        thread_count=-1,
    )


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimise_thresholds(oof_probs, y_true, n_steps=50):
    """
    Find per-class probability thresholds that maximise macro F1.

    The default argmax rule systematically under-predicts rare classes
    because their probabilities are naturally lower. We search for
    thresholds that give minority classes priority in the decision.

    Critically: thresholds are optimised on the SAME probability matrix
    that predict.py will use (the weighted ensemble). This ensures
    consistency between training and inference.
    """
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_f1 = 0.0
    threshold_grid = np.linspace(0.05, 0.95, n_steps)

    for t2 in threshold_grid:
        for t1 in threshold_grid:
            preds = apply_thresholds(oof_probs, {0: 0.5, 1: t1, 2: t2})
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_thresholds = {0: 0.5, 1: t1, 2: t2}

    return best_thresholds, best_f1


def apply_thresholds(probs, thresholds):
    """Apply per-class thresholds. High checked first (rarest class priority)."""
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


# ── Optimal ensemble weight search ───────────────────────────────────────────

def find_optimal_weights(oof_probs_lgb, oof_probs_xgb, y_true, n_steps=20):
    """
    Search for the optimal LGB/XGB weighting via OOF macro F1.

    Rather than guessing weights (e.g. 0.5/0.5), we search over a grid
    to find the combination that maximises OOF macro F1 after threshold
    optimisation. This is computed on OOF data so it is not leakage.

    Returns
    -------
    best_w_lgb : float  (weight for LGB, XGB weight = 1 - best_w_lgb)
    best_f1 : float
    best_thresholds : dict
    best_ensemble_probs : np.ndarray
    """
    best_w_lgb = 0.5
    best_f1 = 0.0
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_ensemble_probs = None

    weight_grid = np.linspace(0.3, 0.8, n_steps)

    for w_lgb in weight_grid:
        w_xgb = 1.0 - w_lgb
        ensemble = w_lgb * oof_probs_lgb + w_xgb * oof_probs_xgb
        thresholds, f1 = optimise_thresholds(ensemble, y_true, n_steps=30)
        if f1 > best_f1:
            best_f1 = f1
            best_w_lgb = w_lgb
            best_thresholds = thresholds
            best_ensemble_probs = ensemble

    return best_w_lgb, best_f1, best_thresholds, best_ensemble_probs


# ── Early stopping fit functions ─────────────────────────────────────────────

def fit_lgb(model, X_tr, y_tr, X_val, y_val):
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def fit_xgb(model, X_tr, y_tr, X_val, y_val, sample_weight):
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
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


# ── Cross-validation loop ─────────────────────────────────────────────────────

def run_cv(X, y, class_weights):
    """
    Run stratified k-fold CV for LightGBM, XGBoost, and CatBoost.

    All three model OOF probs are saved separately so ensemble weights
    can be searched post-training. CatBoost is trained for completeness
    but excluded from the primary ensemble.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_probs_lgb = np.zeros((len(X), N_CLASSES))
    oof_probs_xgb = np.zeros((len(X), N_CLASSES))
    oof_probs_cat = np.zeros((len(X), N_CLASSES))

    models_lgb, models_xgb, models_cat = [], [], []
    sample_weights_full = get_sample_weights(y, class_weights)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")

        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights_full[train_idx]

        # LightGBM
        lgb_model = make_lgb_model(class_weights)
        lgb_model = fit_lgb(lgb_model, X_tr, y_tr, X_val, y_val)
        oof_probs_lgb[val_idx] = lgb_model.predict_proba(X_val)
        models_lgb.append(lgb_model)
        lgb_f1 = f1_score(y_val, lgb_model.predict(X_val), average="macro")
        print(f"    LGB  F1: {lgb_f1:.4f} | best iter: {lgb_model.best_iteration_}")

        # XGBoost
        xgb_model = make_xgb_model()
        xgb_model = fit_xgb(xgb_model, X_tr, y_tr, X_val, y_val, sample_weight=sw_tr)
        oof_probs_xgb[val_idx] = xgb_model.predict_proba(X_val)
        models_xgb.append(xgb_model)
        xgb_f1 = f1_score(y_val, xgb_model.predict(X_val), average="macro")
        print(f"    XGB  F1: {xgb_f1:.4f}")

        # CatBoost (trained but not used in primary ensemble)
        cat_model = make_cat_model(class_weights)
        cat_model = fit_cat(cat_model, X_tr, y_tr, X_val, y_val)
        oof_probs_cat[val_idx] = cat_model.predict_proba(X_val)
        models_cat.append(cat_model)
        cat_f1 = f1_score(y_val, cat_model.predict(X_val), average="macro")
        print(f"    CAT  F1: {cat_f1:.4f} | best iter: {cat_model.best_iteration_}")

    return (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        models_lgb, models_xgb, models_cat,
    )


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(models_lgb, feature_names, top_n=20):
    importances = np.zeros(len(feature_names))
    for model in models_lgb:
        importances += model.feature_importances_
    importances /= len(models_lgb)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features (LightGBM average importance):")
    for _, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:<45} {row['importance']:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading feature data...")
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    print(f"  Shape: {train.shape}")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X = train[feature_cols]
    y = train[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    class_weights = get_class_weights(y)
    print(f"\nClass weights: { {k: round(v, 3) for k, v in class_weights.items()} }")
    print("  (0=Low, 1=Medium, 2=High)")

    print(f"\nRunning {N_FOLDS}-fold stratified CV...")
    (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        models_lgb, models_xgb, models_cat,
    ) = run_cv(X, y, class_weights)

    # Per-model OOF scores
    print("\n" + "=" * 50)
    print("Per-model OOF macro F1 (argmax, no threshold optimisation):")
    for name, probs in [
        ("LightGBM", oof_probs_lgb),
        ("XGBoost ", oof_probs_xgb),
        ("CatBoost", oof_probs_cat),
    ]:
        preds = np.argmax(probs, axis=1)
        score = f1_score(y, preds, average="macro")
        print(f"  {name}  {score:.4f}")

    # Find optimal LGB/XGB ensemble weights on OOF
    print("\nSearching for optimal LGB/XGB ensemble weights...")
    best_w_lgb, best_f1, best_thresholds, best_ensemble_probs = find_optimal_weights(
        oof_probs_lgb, oof_probs_xgb, y
    )
    best_w_xgb = 1.0 - best_w_lgb

    print(f"  Best weights: LGB={best_w_lgb:.2f}, XGB={best_w_xgb:.2f}")
    print(f"  Best thresholds: { {k: round(v, 3) for k, v in best_thresholds.items()} }")
    print(f"  Best OOF macro F1: {best_f1:.4f}")

    print_feature_importance(models_lgb, feature_cols)

    # Save all artifacts
    # best_ensemble_probs and best_thresholds are consistent with each other
    # predict.py will use the same weights to build test ensemble probs
    print("\nSaving models and metadata...")
    artifacts = {
        "models_lgb": models_lgb,
        "models_xgb": models_xgb,
        "models_cat": models_cat,
        "feature_cols": feature_cols,
        "best_thresholds": best_thresholds,
        "best_w_lgb": best_w_lgb,
        "best_w_xgb": best_w_xgb,
        "oof_probs_lgb": oof_probs_lgb,
        "oof_probs_xgb": oof_probs_xgb,
        "oof_probs_cat": oof_probs_cat,
        "oof_probs_ensemble": best_ensemble_probs,
        "oof_f1": best_f1,
    }

    with open(MODELS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print(f"  Saved models/artifacts.pkl")
    print(f"\nFinal OOF macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()