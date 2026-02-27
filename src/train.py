"""
train.py

Trains an ensemble of LightGBM, XGBoost, and CatBoost models using
stratified k-fold cross-validation. Handles class imbalance via class
weights. Optimizes per-class probability thresholds post-prediction to
maximise macro F1. Saves out-of-fold predictions and trained models.

Run from the project root:
    python src/train.py
"""

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

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_FOLDS = 5
TARGET_COL = "Target"
ID_COL = "ID"
N_CLASSES = 3  # Low=0, Medium=1, High=2

# Columns to drop before training - not features
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
    LightGBM classifier.

    Key decisions:
    - class_weight param accepts a dict directly
    - num_leaves=63 gives good depth without overfitting on this data size
    - min_child_samples=20 prevents tiny leaf nodes
    - colsample_bytree=0.8 adds feature randomness for better generalisation
    - verbose=-1 suppresses per-iteration output
    """
    return lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight=class_weights,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def make_xgb_model():
    """
    XGBoost classifier.

    XGBoost does not accept a class_weight dict directly for multi-class.
    We pass sample_weight per-sample during fit() instead.
    - tree_method='hist' is fastest for CPU
    - eval_metric='mlogloss' for multi-class
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
    CatBoost classifier.

    CatBoost handles missing values natively, which matters here since
    we have significant missingness even after adding flags.
    - class_weights param accepts a list ordered by class index
    - verbose=0 suppresses output
    - early_stopping_rounds handled manually via eval_set
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

    The default decision rule is argmax(probabilities), which picks the
    class with the highest probability. But this systematically under-predicts
    rare classes because their probabilities are naturally lower even when
    the model is confident about them.

    We search for thresholds t0, t1, t2 such that:
        predict High   if P(High) >= t2
        predict Medium if P(Medium) >= t1
        predict Low    otherwise

    The search applies thresholds in order of class rarity (rarest first)
    to give minority classes priority in the decision.

    Parameters
    ----------
    oof_probs : np.ndarray of shape (n_samples, 3)
        Out-of-fold predicted probabilities.
    y_true : np.ndarray
        True labels.
    n_steps : int
        Number of threshold values to try per class.

    Returns
    -------
    best_thresholds : dict {class_index: threshold}
    best_f1 : float
    """
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_f1 = 0.0

    threshold_grid = np.linspace(0.05, 0.95, n_steps)

    # Search over High threshold first (most impactful for minority class)
    for t2 in threshold_grid:
        for t1 in threshold_grid:
            preds = apply_thresholds(oof_probs, {0: 0.5, 1: t1, 2: t2})
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_thresholds = {0: 0.5, 1: t1, 2: t2}

    return best_thresholds, best_f1


def apply_thresholds(probs, thresholds):
    """
    Apply per-class thresholds to probability matrix.

    Decision order: High first, then Medium, then Low as default.
    This gives the rarest class first priority.
    """
    n = len(probs)
    preds = np.zeros(n, dtype=int)  # default: Low

    for i in range(n):
        if probs[i, 2] >= thresholds[2]:
            preds[i] = 2  # High
        elif probs[i, 1] >= thresholds[1]:
            preds[i] = 1  # Medium
        else:
            preds[i] = 0  # Low

    return preds


# ── Early stopping callback ───────────────────────────────────────────────────

def fit_with_early_stopping_lgb(model, X_tr, y_tr, X_val, y_val):
    """Fit LGB with early stopping on validation set."""
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def fit_with_early_stopping_xgb(model, X_tr, y_tr, X_val, y_val, sample_weight):
    """Fit XGB with early stopping on validation set."""
    model.fit(
        X_tr, y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def fit_with_early_stopping_cat(model, X_tr, y_tr, X_val, y_val):
    """Fit CatBoost with early stopping on validation set."""
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
    Run stratified k-fold cross-validation for all three models.

    For each fold:
    - Train LightGBM, XGBoost, CatBoost with early stopping
    - Collect out-of-fold probability predictions from each model
    - Average the three probability matrices (simple ensemble)

    Returns
    -------
    oof_probs_lgb, oof_probs_xgb, oof_probs_cat : np.ndarray (n, 3)
        Per-model out-of-fold probabilities.
    oof_probs_ensemble : np.ndarray (n, 3)
        Averaged ensemble probabilities.
    models_lgb, models_xgb, models_cat : list
        Trained model objects from each fold (used for test prediction).
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

        # ── LightGBM ──
        lgb_model = make_lgb_model(class_weights)
        lgb_model = fit_with_early_stopping_lgb(lgb_model, X_tr, y_tr, X_val, y_val)
        oof_probs_lgb[val_idx] = lgb_model.predict_proba(X_val)
        models_lgb.append(lgb_model)

        lgb_f1 = f1_score(y_val, lgb_model.predict(X_val), average="macro")
        print(f"    LGB  F1: {lgb_f1:.4f} | best iter: {lgb_model.best_iteration_}")

        # ── XGBoost ──
        xgb_model = make_xgb_model()
        xgb_model = fit_with_early_stopping_xgb(
            xgb_model, X_tr, y_tr, X_val, y_val, sample_weight=sw_tr
        )
        oof_probs_xgb[val_idx] = xgb_model.predict_proba(X_val)
        models_xgb.append(xgb_model)

        xgb_f1 = f1_score(y_val, xgb_model.predict(X_val), average="macro")
        print(f"    XGB  F1: {xgb_f1:.4f}")

        # ── CatBoost ──
        cat_model = make_cat_model(class_weights)
        cat_model = fit_with_early_stopping_cat(cat_model, X_tr, y_tr, X_val, y_val)
        oof_probs_cat[val_idx] = cat_model.predict_proba(X_val)
        models_cat.append(cat_model)

        cat_f1 = f1_score(y_val, cat_model.predict(X_val), average="macro")
        print(f"    CAT  F1: {cat_f1:.4f} | best iter: {cat_model.best_iteration_}")

    # Simple average ensemble
    oof_probs_ensemble = (oof_probs_lgb + oof_probs_xgb + oof_probs_cat) / 3.0

    return (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        oof_probs_ensemble,
        models_lgb, models_xgb, models_cat,
    )


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(models_lgb, feature_names, top_n=20):
    """Print average feature importance across LGB folds."""
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

    # Separate features and target
    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X = train[feature_cols]
    y = train[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Class weights
    class_weights = get_class_weights(y)
    print(f"\nClass weights: { {k: round(v, 3) for k, v in class_weights.items()} }")
    print("  (0=Low, 1=Medium, 2=High)")

    # Run cross-validation
    print(f"\nRunning {N_FOLDS}-fold stratified CV...")
    (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        oof_probs_ensemble,
        models_lgb, models_xgb, models_cat,
    ) = run_cv(X, y, class_weights)

    # Per-model OOF scores
    print("\n" + "=" * 50)
    print("OOF Results (before threshold optimisation):")

    for name, probs in [
        ("LightGBM ", oof_probs_lgb),
        ("XGBoost  ", oof_probs_xgb),
        ("CatBoost ", oof_probs_cat),
        ("Ensemble ", oof_probs_ensemble),
    ]:
        preds = np.argmax(probs, axis=1)
        score = f1_score(y, preds, average="macro")
        print(f"  {name} macro F1: {score:.4f}")

    # Threshold optimisation on ensemble OOF
    print("\nOptimising thresholds on ensemble OOF predictions...")
    best_thresholds, best_f1 = optimise_thresholds(oof_probs_ensemble, y)
    print(f"  Best thresholds: { {k: round(v, 3) for k, v in best_thresholds.items()} }")
    print(f"  Best OOF macro F1 after threshold optimisation: {best_f1:.4f}")

    # Feature importance
    print_feature_importance(models_lgb, feature_cols)

    # Save everything needed for predict.py
    print("\nSaving models and metadata...")
    artifacts = {
        "models_lgb": models_lgb,
        "models_xgb": models_xgb,
        "models_cat": models_cat,
        "feature_cols": feature_cols,
        "best_thresholds": best_thresholds,
        "oof_probs_ensemble": oof_probs_ensemble,
        "oof_f1": best_f1,
    }

    with open(MODELS_DIR / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    print(f"  Saved models/artifacts.pkl")
    print(f"\nFinal OOF macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()