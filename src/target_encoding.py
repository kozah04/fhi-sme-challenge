"""Shared target encoding utilities for the ordinal pipeline."""

import pandas as pd

N_CLASSES = 3

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


class TargetEncoder:
    """Fold-safe smoothed target encoder for multiclass targets."""

    def __init__(self, smoothing=10):
        self.smoothing = smoothing
        self.encodings_ = {}
        self.global_means_ = {}

    def fit(self, X, y, cols):
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

                stats = pd.DataFrame({
                    "value": df[col],
                    "target": binary_target,
                }).groupby("value")["target"].agg(["count", "mean"])

                smoothed = (
                    (stats["count"] * stats["mean"] + self.smoothing * global_mean)
                    / (stats["count"] + self.smoothing)
                )
                self.encodings_[col][cls] = smoothed.to_dict()

        return self

    def transform(self, X):
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
