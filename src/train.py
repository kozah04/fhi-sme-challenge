"""
train.py

Trains an ensemble of LightGBM and XGBoost models using stratified
k-fold cross-validation. Handles class imbalance via class weights.
Applies target encoding inside CV folds to prevent leakage. Finds
optimal ensemble weights via OOF search. Optimises per-class probability
thresholds post-prediction to maximise weighted F1.

Target encoding is computed on the training fold only and applied to the
validation fold and test set. This is the correct pattern - computing on
the full training set would leak validation target information into the
features, inflating OOF scores and hurting generalisation.

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
N_CLASSES = 3
DROP_COLS = [ID_COL, TARGET_COL]

# Columns to target encode.
# These are the ordinal-encoded categorical columns where the raw integer
# (0/1/2 or 0/1/3 for country) carries less information than the actual
# target distribution for each value.
# We do NOT target encode binary columns (0/1 only) as they carry little
# extra information from TE, and we avoid the continuous numeric columns.
TARGET_ENCODE_COLS = [
    "country",
    "has_credit_card",
    "has_loan_account",
    "has_internet_banking",
    "has_debit_card",
    "motor_vehicle_insurance",
    "medical_insurance",
    "funeral_insurance",
    "has_mobile_money",
    "uses_friends_family_savings",
    "uses_informal_lender",
    "keeps_financial_records",
    "offers_credit_to_customers",
    "compliance_income_tax",
]


# ── Target encoding ───────────────────────────────────────────────────────────

class TargetEncoder:
    """
    Encodes categorical columns as the mean target value per category,
    computed on the training fold only.

    For multi-class targets we produce one encoding per class:
        col_te_0 = P(Target=Low   | col=value)
        col_te_1 = P(Target=Medium | col=value)
        col_te_2 = P(Target=High  | col=value)

    Smoothing is applied to shrink rare categories toward the global mean,
    preventing overfit on categories with very few examples.

    Parameters
    ----------
    smoothing : float
        Controls how much rare categories are shrunk toward the global mean.
        Higher = more shrinkage. Default 10 works well for this dataset size.
    """

    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.encodings_ = {}   # col -> {class -> {value -> encoded_mean}}
        self.global_means_ = {}  # col -> {class -> global_mean}

    def fit(self, X, y, cols):
        """Compute target means per category on training data."""
        self.cols = cols
        df = X[cols].copy()
        df["__target__"] = y

        for col in cols:
            self.encodings_[col] = {}
            self.global_means_[col] = {}

            for cls in range(N_CLASSES):
                binary_target = (df["__target__"] == cls).astype(float)
                global_mean = binary_target.mean()
                self.global_means_[col][cls] = global_mean

                # Count and mean per category value
                stats = pd.DataFrame({
                    "value": df[col],
                    "target": binary_target,
                }).groupby("value")["target"].agg(["count", "mean"])

                # Smoothed encoding: blend category mean with global mean
                # Weight toward category mean as n increases
                smoothed = (
                    (stats["count"] * stats["mean"] + self.smoothing * global_mean)
                    / (stats["count"] + self.smoothing)
                )
                self.encodings_[col][cls] = smoothed.to_dict()

        return self

    def transform(self, X):
        """Apply fitted encoding to a dataframe. Unknown values get global mean."""
        X_out = X.copy()

        for col in self.cols:
            for cls in range(N_CLASSES):
                new_col = f"{col}_te_{cls}"
                global_mean = self.global_means_[col][cls]
                encoding_map = self.encodings_[col][cls]
                X_out[new_col] = X_out[col].map(encoding_map).fillna(global_mean)

        return X_out

    def fit_transform(self, X, y, cols):
        return self.fit(X, y, cols).transform(X)


# ── Class weights ─────────────────────────────────────────────────────────────

def get_class_weights(y):
    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return dict(zip(classes, weights))


def get_sample_weights(y, class_weight_dict):
    return np.array([class_weight_dict[yi] for yi in y])


# ── Model definitions ─────────────────────────────────────────────────────────

def make_lgb_model():
    """LightGBM with Optuna-tuned hyperparameters."""
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
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )


def make_xgb_model():
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


def make_cat_model():
    return CatBoostClassifier(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=SEED,
        verbose=0,
        thread_count=-1,
    )


# ── Threshold optimisation ────────────────────────────────────────────────────

def optimise_thresholds(oof_probs, y_true, n_steps=50):
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_f1 = 0.0
    threshold_grid = np.linspace(0.05, 0.95, n_steps)
    

    for t2 in threshold_grid:
        for t1 in threshold_grid:
            preds = apply_thresholds(oof_probs, {0: 0.5, 1: t1, 2: t2})
            score = f1_score(y_true, preds, average="weighted")
            if score > best_f1:
                best_f1 = score
                best_thresholds = {0: 0.5, 1: t1, 2: t2}

    return best_thresholds, best_f1


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


# ── Optimal ensemble weight search ───────────────────────────────────────────

def find_optimal_weights(oof_probs_lgb, oof_probs_xgb, y_true, n_steps=20):
    best_w_lgb = 0.5
    best_f1 = 0.0
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_ensemble_probs = None

    for w_lgb in np.linspace(0.3, 0.8, n_steps):
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
    Run stratified k-fold CV with target encoding applied inside each fold.

    For each fold:
    1. Fit TargetEncoder on training portion only
    2. Transform training and validation portions
    3. Train LGB, XGB, CatBoost on transformed features
    4. Collect OOF probabilities on the validation fold

    The fitted encoders from each fold are saved so predict.py can apply
    the full-data encoder (fitted on all training data) to the test set.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    oof_probs_lgb = np.zeros((len(X), N_CLASSES))
    oof_probs_xgb = np.zeros((len(X), N_CLASSES))
    oof_probs_cat = np.zeros((len(X), N_CLASSES))

    models_lgb, models_xgb, models_cat = [], [], []
    sample_weights_full = get_sample_weights(y, class_weights)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")

        X_tr_raw = X.iloc[train_idx].copy()
        X_val_raw = X.iloc[val_idx].copy()
        y_tr = y[train_idx]
        y_val = y[val_idx]
        sw_tr = sample_weights_full[train_idx]

        # Fit target encoder on training fold, apply to both
        te = TargetEncoder(smoothing=10)
        X_tr = te.fit_transform(X_tr_raw, y_tr, TARGET_ENCODE_COLS)
        X_val = te.transform(X_val_raw)

        # LightGBM
        lgb_model = make_lgb_model()
        lgb_model = fit_lgb(lgb_model, X_tr, y_tr, X_val, y_val)
        oof_probs_lgb[val_idx] = lgb_model.predict_proba(X_val)
        models_lgb.append(lgb_model)
        lgb_f1 = f1_score(y_val, lgb_model.predict(X_val), average="weighted")
        print(f"    LGB  F1: {lgb_f1:.4f} | best iter: {lgb_model.best_iteration_}")

        # XGBoost
        xgb_model = make_xgb_model()
        xgb_model = fit_xgb(xgb_model, X_tr, y_tr, X_val, y_val, sample_weight=None)
        oof_probs_xgb[val_idx] = xgb_model.predict_proba(X_val)
        models_xgb.append(xgb_model)
        xgb_f1 = f1_score(y_val, xgb_model.predict(X_val), average="weighted")
        print(f"    XGB  F1: {xgb_f1:.4f}")

        # CatBoost
        cat_model = make_cat_model()
        cat_model = fit_cat(cat_model, X_tr, y_tr, X_val, y_val)
        oof_probs_cat[val_idx] = cat_model.predict_proba(X_val)
        models_cat.append(cat_model)
        cat_f1 = f1_score(y_val, cat_model.predict(X_val), average="weighted")
        print(f"    CAT  F1: {cat_f1:.4f} | best iter: {cat_model.best_iteration_}")

    return (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        models_lgb, models_xgb, models_cat,
    )


# ── Feature importance ────────────────────────────────────────────────────────

def print_feature_importance(models_lgb, top_n=20):
    """Print feature importance from the first fold (representative)."""
    model = models_lgb[0]
    importances = model.feature_importances_
    feature_names = model.feature_name_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(f"\nTop {top_n} features (LightGBM fold 1 importance):")
    for _, row in importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:<45} {row['importance']:.1f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading feature data...")
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_features.csv")
    print(f"  Train: {train.shape}, Test: {test.shape}")

    feature_cols = [c for c in train.columns if c not in DROP_COLS]
    X = train[feature_cols]
    y = train[TARGET_COL].values
    X_test = test[feature_cols]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"  Target encoding: {len(TARGET_ENCODE_COLS)} cols x {N_CLASSES} classes "
          f"= {len(TARGET_ENCODE_COLS) * N_CLASSES} new features")

    class_weights = get_class_weights(y)
    print(f"\nClass weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    print(f"\nRunning {N_FOLDS}-fold stratified CV with target encoding...")
    (
        oof_probs_lgb, oof_probs_xgb, oof_probs_cat,
        models_lgb, models_xgb, models_cat,
    ) = run_cv(X, y, class_weights)

    print("\n" + "=" * 50)
    print("Per-model OOF weighted F1 (argmax):")
    for name, probs in [
        ("LightGBM", oof_probs_lgb),
        ("XGBoost ", oof_probs_xgb),
        ("CatBoost", oof_probs_cat),
    ]:
        preds = np.argmax(probs, axis=1)
        score = f1_score(y, preds, average="weighted")
        print(f"  {name}  {score:.4f}")

    print("\nSearching for optimal LGB/XGB ensemble weights...")
    best_w_lgb, best_f1, best_thresholds, best_ensemble_probs = find_optimal_weights(
        oof_probs_lgb, oof_probs_xgb, y
    )
    best_w_xgb = 1.0 - best_w_lgb
    print(f"  Best weights: LGB={best_w_lgb:.2f}, XGB={best_w_xgb:.2f}")
    print(f"  Best thresholds: { {k: round(v, 3) for k, v in best_thresholds.items()} }")
    print(f"  Best OOF weighted F1: {best_f1:.4f}")

    print_feature_importance(models_lgb)

    # Fit a full-data target encoder for transforming the test set.
    # This uses all training data (no fold split) to get the most stable
    # encoding estimates for the test set.
    print("\nFitting full-data target encoder for test set...")
    full_te = TargetEncoder(smoothing=10)
    full_te.fit(X, y, TARGET_ENCODE_COLS)

    print("Saving models and metadata...")
    artifacts = {
        "models_lgb": models_lgb,
        "models_xgb": models_xgb,
        "models_cat": models_cat,
        "feature_cols": feature_cols,
        "target_encode_cols": TARGET_ENCODE_COLS,
        "full_target_encoder": full_te,
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
    print(f"\nFinal OOF weighted F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()