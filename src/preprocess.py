"""
preprocess.py

Loads raw Train.csv and Test.csv, cleans dirty categorical values,
encodes features, and saves processed files to data/processed/.

Run from the project root:
    python src/preprocess.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ── Constants ────────────────────────────────────────────────────────────────

TARGET_COL = "Target"
ID_COL = "ID"
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
TARGET_INVERSE = {0: "Low", 1: "Medium", 2: "High"}

# Columns that follow the Have now / Used to have / Never had pattern.
# These are encoded as ordinal: Have now=2, Used to have=1, Never had=0.
# This ordering reflects a meaningful progression of financial inclusion.
PRODUCT_COLS = [
    "motor_vehicle_insurance",
    "has_mobile_money",
    "has_credit_card",
    "has_loan_account",
    "has_internet_banking",
    "has_debit_card",
    "medical_insurance",
    "funeral_insurance",
    "uses_friends_family_savings",
    "uses_informal_lender",
]

# Binary yes/no columns
BINARY_COLS = [
    "has_cellphone",
    "has_insurance",
    "future_risk_theft_stock",
    "problem_sourcing_money",
    "marketing_word_of_mouth",
    "motivation_make_more_money",
    "covid_essential_service",
]

# Columns with Yes/No/Don't know pattern - treated as ternary
TERNARY_COLS = [
    "attitude_stable_business_environment",
    "attitude_worried_shutdown",
    "attitude_satisfied_with_achievement",
    "attitude_more_successful_next_year",
    "perception_insurance_doesnt_cover_losses",
    "perception_cannot_afford_insurance",
    "perception_insurance_companies_dont_insure_businesses_like_yours",
    "perception_insurance_important",
    "compliance_income_tax",
    "current_problem_cash_flow",
]

# Numeric columns - kept as-is but will be log-transformed in features.py
NUMERIC_COLS = [
    "owner_age",
    "personal_income",
    "business_expenses",
    "business_turnover",
    "business_age_years",
    "business_age_months",
]


# ── Normalisation helpers ────────────────────────────────────────────────────

def normalise_string(val):
    """Lowercase, strip whitespace, normalise unicode apostrophes."""
    if pd.isna(val):
        return np.nan
    return (
        str(val)
        .strip()
        .lower()
        .replace("\u2019", "'")   # right single quotation mark
        .replace("\u200e", "")    # left-to-right mark (found in perception cols)
        .replace("?", "'")        # mangled apostrophe found in raw data
    )


def encode_product_col(series):
    """
    Encode the Have now / Used to have / Never had pattern as ordinal integers.

    Have now          -> 2  (currently financially included)
    Used to have      -> 1  (previously included, now excluded)
    Never had         -> 0  (never included)
    Don't know / NaN  -> NaN (genuinely unknown, handled later)

    The ordinal encoding preserves the meaningful direction: higher = more
    financially included. One-hot encoding would lose this information.
    """
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if "used to have" in v:
            return 1
        if "have now" in v:
            return 2
        if "never had" in v:
            return 0
        return np.nan  # don't know variants treated as missing

    return series.apply(_encode)


def encode_binary_col(series):
    """
    Encode Yes/No columns as 1/0.
    Anything else (Don't know, NaN) becomes NaN.
    """
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if v == "yes":
            return 1
        if v == "no":
            return 0
        return np.nan

    return series.apply(_encode)


def encode_ternary_col(series, col_name):
    """
    Encode Yes/No/Don't know columns.

    The encoding varies slightly by column based on what a positive answer means
    for financial health. For most attitude/perception columns, Yes=1, No=0.
    Don't know and refused are treated as missing.

    For compliance_income_tax: Yes=1, No=0 (compliance is positive signal).
    For current_problem_cash_flow: Yes=1 means a problem exists (negative signal,
    but we leave direction to the model - encoding is just numeric).
    """
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if v in ("yes", "1"):
            return 1
        if v == "no" or v == "0":
            return 0
        # Don't know, refused, N/A all become missing
        return np.nan

    return series.apply(_encode)


def encode_keeps_financial_records(series):
    """
    keeps_financial_records has a four-level ordinal structure:
    Yes, always > Yes, sometimes > Yes > No

    We encode this as 3/2/1/0 to preserve the ordinal information.
    The distinction between 'always' and 'sometimes' matters for financial health.
    """
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if "always" in v:
            return 3
        if "sometimes" in v:
            return 2
        if v == "yes":
            return 1
        if v == "no":
            return 0
        return np.nan

    return series.apply(_encode)


def encode_offers_credit(series):
    """
    offers_credit_to_customers: Yes always > Yes sometimes > No
    Encoded as 2/1/0.
    """
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if "always" in v:
            return 2
        if "sometimes" in v:
            return 1
        if v == "no":
            return 0
        return np.nan

    return series.apply(_encode)


def encode_owner_sex(series):
    """Encode Male=1, Female=0."""
    def _encode(val):
        v = normalise_string(val)
        if pd.isna(v):
            return np.nan
        if v == "male":
            return 1
        if v == "female":
            return 0
        return np.nan

    return series.apply(_encode)


def encode_country(series):
    """
    Ordinal encode country by median financial health in training data.
    Eswatini=3 (highest High rate), Malawi/Zimbabwe=1, Lesotho=0.
    Also keep as a categorical integer for embedding-style use in trees.

    We use a simple integer map here. Target encoding is done inside CV
    folds in train.py to prevent leakage.
    """
    country_map = {
        "eswatini": 3,
        "zimbabwe": 1,
        "malawi": 1,
        "lesotho": 0,
    }
    return series.str.lower().map(country_map)


# ── Main cleaning function ───────────────────────────────────────────────────

def clean(df, is_train=True):
    """
    Apply all cleaning and encoding steps to a raw dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Raw data as loaded from CSV.
    is_train : bool
        If True, also encode the Target column.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with all features numerically encoded.
        ID column is retained for joining purposes.
    """
    df = df.copy()

    # Encode target
    if is_train:
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)

    # Country
    df["country"] = encode_country(df["country"])

    # Financial product columns (Have now / Used to have / Never had)
    for col in PRODUCT_COLS:
        df[col] = encode_product_col(df[col])

    # Binary columns
    for col in BINARY_COLS:
        df[col] = encode_binary_col(df[col])

    # Ternary columns
    for col in TERNARY_COLS:
        df[col] = encode_ternary_col(df[col], col)

    # Special ordinal columns
    df["keeps_financial_records"] = encode_keeps_financial_records(df["keeps_financial_records"])
    df["offers_credit_to_customers"] = encode_offers_credit(df["offers_credit_to_customers"])
    df["owner_sex"] = encode_owner_sex(df["owner_sex"])

    # Numeric columns: coerce to float (handles any stray strings)
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ── Missingness flags ────────────────────────────────────────────────────────

def add_missing_flags(df):
    """
    Add binary indicator columns for features with high missingness.

    Missingness in this dataset is not random - it reflects which survey
    module was administered in each country. The pattern of what is missing
    is itself informative and should be given to the model explicitly.

    We add flags for any column missing more than 20% of values.
    """
    df = df.copy()
    feature_cols = [c for c in df.columns if c not in [ID_COL, TARGET_COL]]

    for col in feature_cols:
        missing_rate = df[col].isna().mean()
        if missing_rate > 0.20:
            df[f"{col}_missing"] = df[col].isna().astype(int)

    return df


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    print("Loading raw data...")
    train_raw = pd.read_csv(RAW_DIR / "Train.csv")
    test_raw = pd.read_csv(RAW_DIR / "Test.csv")

    print(f"  Train: {train_raw.shape}, Test: {test_raw.shape}")

    print("Cleaning train...")
    train_clean = clean(train_raw, is_train=True)
    train_clean = add_missing_flags(train_clean)

    print("Cleaning test...")
    test_clean = clean(test_raw, is_train=False)
    test_clean = add_missing_flags(test_clean)

    # Align columns - test should have same feature columns as train minus target
    train_features = [c for c in train_clean.columns if c != TARGET_COL]
    test_clean = test_clean.reindex(columns=train_features)

    print("Saving processed data...")
    train_clean.to_csv(PROCESSED_DIR / "train_clean.csv", index=False)
    test_clean.to_csv(PROCESSED_DIR / "test_clean.csv", index=False)

    print(f"  Saved train_clean.csv: {train_clean.shape}")
    print(f"  Saved test_clean.csv:  {test_clean.shape}")

    # Quick sanity check on target distribution
    print("\nTarget distribution after encoding:")
    print(train_clean[TARGET_COL].value_counts().sort_index())
    print("  (0=Low, 1=Medium, 2=High)")

    # Report missingness after encoding
    missing_after = train_clean.isnull().sum()
    missing_after = missing_after[missing_after > 0].sort_values(ascending=False)
    print(f"\nColumns still missing values after encoding ({len(missing_after)} cols):")
    print(missing_after.head(15))


if __name__ == "__main__":
    main()