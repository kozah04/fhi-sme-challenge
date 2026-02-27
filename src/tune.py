"""
tune.py

Uses Optuna to find optimal LightGBM hyperparameters via stratified
k-fold cross-validation. Optimises macro F1 as the objective.

Results are saved to models/best_params.pkl for use in train.py.

Run from the project root:
    python src/tune.py

This will take 20-40 minutes on CPU. Progress is printed every trial.
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path

import optuna
from optuna.samplers import TPESampler

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
N_FOLDS = 5
N_TRIALS = 100
TARGET_COL = "Target"
ID_COL = "ID"


# ── Threshold application (mirrors train.py) ──────────────────────────────────

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


def quick_threshold_search(oof_probs, y_true, n_steps=30):
    """Faster threshold search for use inside Optuna trials."""
    best_thresholds = {0: 0.5, 1: 0.5, 2: 0.5}
    best_f1 = 0.0
    grid = np.linspace(0.05, 0.95, n_steps)

    for t2 in grid:
        for t1 in grid:
            preds = apply_thresholds(oof_probs, {0: 0.5, 1: t1, 2: t2})
            score = f1_score(y_true, preds, average="macro")
            if score > best_f1:
                best_f1 = score
                best_thresholds = {0: 0.5, 1: t1, 2: t2}

    return best_thresholds, best_f1


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(X, y, class_weights):
    """
    Returns an Optuna objective function that:
    1. Samples LightGBM hyperparameters
    2. Runs 5-fold CV with early stopping
    3. Returns macro F1 after threshold optimisation

    We tune LightGBM only (not XGBoost or CatBoost) because:
    - LGB is the fastest to train, allowing more trials
    - LGB is currently our best single model
    - Once we have best LGB params, we retrain the full ensemble
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial):
        params = {
            "n_estimators": 2000,  # high ceiling, early stopping controls actual count
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "class_weight": class_weights,
            "random_state": SEED,
            "verbose": -1,
            "n_jobs": -1,
        }

        oof_probs = np.zeros((len(X), 3))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(-1),
                ],
            )
            oof_probs[val_idx] = model.predict_proba(X_val)

        _, best_f1 = quick_threshold_search(oof_probs, y)
        return best_f1

    return objective


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading feature data...")
    train = pd.read_csv(PROCESSED_DIR / "train_features.csv")
    feature_cols = [c for c in train.columns if c not in [ID_COL, TARGET_COL]]
    X = train[feature_cols]
    y = train[TARGET_COL].values

    print(f"  Features: {len(feature_cols)}, Samples: {len(X)}")

    classes = np.array([0, 1, 2])
    weights = compute_class_weight("balanced", classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    print(f"  Class weights: { {k: round(v, 3) for k, v in class_weights.items()} }")

    print(f"\nRunning Optuna: {N_TRIALS} trials, {N_FOLDS}-fold CV each")
    print("Progress printed every 10 trials...\n")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )

    # Track and print progress every 10 trials
    trial_scores = []

    def callback(study, trial):
        trial_scores.append(trial.value)
        if (trial.number + 1) % 10 == 0:
            best = study.best_value
            print(f"  Trial {trial.number + 1:>3}/{N_TRIALS}  "
                  f"current: {trial.value:.4f}  "
                  f"best so far: {best:.4f}")

    objective = make_objective(X, y, class_weights)
    study.optimize(objective, n_trials=N_TRIALS, callbacks=[callback])

    # Results
    print("\n" + "=" * 50)
    print(f"Best OOF macro F1: {study.best_value:.4f}")
    print("\nBest hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:<25} {v}")

    # Improvement over baseline
    baseline = 0.8086
    improvement = study.best_value - baseline
    print(f"\nImprovement over baseline: {improvement:+.4f}")

    # Save best params
    best_params = study.best_params
    best_params["best_f1"] = study.best_value

    with open(MODELS_DIR / "best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    print(f"\nSaved best params to models/best_params.pkl")
    print("Next step: update train.py to use these params, then retrain.")


if __name__ == "__main__":
    main()