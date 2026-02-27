"""
features.py

Loads cleaned data from data/processed/, engineers new features,
and saves the final feature matrices to data/processed/.

The feature engineering strategy is guided by the FHI construction logic:
the target was derived from the same survey data using four dimensions:
    1. Savings and assets
    2. Debt and repayment ability
    3. Resilience to shocks
    4. Access to credit and financial services

We create explicit composite scores for each dimension because the model
will find these much more useful than raw individual features alone.
The composite scores essentially help the model reconstruct the index
formula rather than having to discover it from scratch.

Run from the project root:
    python src/features.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ── Paths ────────────────────────────────────────────────────────────────────

PROCESSED_DIR = Path("data/processed")


# ── Column groups used in feature engineering ────────────────────────────────

# All ten financial product columns after ordinal encoding (0/1/2)
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

# Subset that specifically maps to formal financial inclusion (access dimension)
FORMAL_ACCESS_COLS = [
    "has_credit_card",
    "has_loan_account",
    "has_internet_banking",
    "has_debit_card",
    "motor_vehicle_insurance",
    "medical_insurance",
]

# Insurance-specific columns (resilience + assets dimension)
INSURANCE_COLS = [
    "motor_vehicle_insurance",
    "medical_insurance",
    "funeral_insurance",
    "has_insurance",  # binary flag for any insurance
]

# Informal finance columns (debt dimension - informal = weaker position)
INFORMAL_COLS = [
    "uses_friends_family_savings",
    "uses_informal_lender",
]

# Attitude and perception columns (all binary encoded: 1/0)
ATTITUDE_COLS = [
    "attitude_stable_business_environment",
    "attitude_satisfied_with_achievement",
    "attitude_more_successful_next_year",
]

NEGATIVE_ATTITUDE_COLS = [
    "attitude_worried_shutdown",
    "perception_cannot_afford_insurance",
    "perception_insurance_doesnt_cover_losses",
]


# ── Dimension 1: Access to credit and financial services ─────────────────────

def make_access_features(df):
    """
    Build features capturing formal financial inclusion.

    The access dimension asks: how many formal financial products does this
    business currently use, and how many has it ever used? A business with
    internet banking, a credit card, and a loan account is fundamentally
    different from one with only mobile money.

    We separate 'Have now' (score=2) from 'Used to have' (score=1) because
    currently active products signal present inclusion while lapsed products
    may signal volatility or inability to maintain access.
    """
    df = df.copy()

    # Count of formal products currently held (Have now = 2)
    df["formal_access_count"] = df[FORMAL_ACCESS_COLS].apply(
        lambda r: (r == 2).sum(), axis=1
    )

    # Count of all financial products currently held across all 10 cols
    df["total_products_now"] = df[PRODUCT_COLS].apply(
        lambda r: (r == 2).sum(), axis=1
    )

    # Count of products ever held (Have now or Used to have)
    df["total_products_ever"] = df[PRODUCT_COLS].apply(
        lambda r: (r >= 1).sum(), axis=1
    )

    # Sum of ordinal scores across all product cols (0+1+2 weighted)
    # Higher = more financially active overall
    df["product_score_total"] = df[PRODUCT_COLS].sum(axis=1)

    # Formal access score: sum of ordinal scores for formal products only
    df["formal_access_score"] = df[FORMAL_ACCESS_COLS].sum(axis=1)

    # Has any formal banking product right now (debit, credit, internet banking)
    banking_cols = ["has_credit_card", "has_internet_banking", "has_debit_card"]
    df["has_any_banking_now"] = df[banking_cols].apply(
        lambda r: int((r == 2).any()), axis=1
    )

    # Has both a loan and a banking product - strong formal inclusion signal
    df["has_loan_and_banking"] = (
        (df["has_loan_account"] == 2) &
        (df[["has_credit_card", "has_internet_banking", "has_debit_card"]] == 2).any(axis=1)
    ).astype(int)

    return df


# ── Dimension 2: Savings and assets ──────────────────────────────────────────

def make_savings_features(df):
    """
    Build features capturing savings behaviour and asset strength.

    Financial records keeping is a direct proxy for savings discipline.
    Personal income relative to business expenses signals whether the owner
    has a buffer. Insurance ownership (especially medical and vehicle) signals
    asset protection capacity.
    """
    df = df.copy()

    # Insurance coverage count (currently active insurance products)
    insurance_product_cols = ["motor_vehicle_insurance", "medical_insurance", "funeral_insurance"]
    df["insurance_count"] = df[insurance_product_cols].apply(
        lambda r: (r == 2).sum(), axis=1
    )

    # Combined insurance signal: has_insurance flag + product count
    # has_insurance=1 with zero product cols active would be an inconsistency
    # this combined score handles both
    df["insurance_strength"] = df["has_insurance"].fillna(0) + df["insurance_count"]

    # Financial records keeping is already ordinal (0-3), keep as-is
    # but create a binary: does the business keep any records at all?
    df["keeps_any_records"] = (df["keeps_financial_records"] > 0).astype(int)

    # Log-transform financial columns to compress extreme outliers
    # Adding 1 before log to handle zeros safely
    for col in ["personal_income", "business_expenses", "business_turnover"]:
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))

    # Income to expenses ratio - proxy for personal financial buffer
    # Clipped to avoid division by zero and extreme values
    df["income_expense_ratio"] = (
        df["personal_income"] / (df["business_expenses"] + 1)
    ).clip(upper=100)
    df["log_income_expense_ratio"] = np.log1p(df["income_expense_ratio"].clip(lower=0))

    # Turnover to expenses ratio - proxy for business profitability
    df["turnover_expense_ratio"] = (
        df["business_turnover"] / (df["business_expenses"] + 1)
    ).clip(upper=100)
    df["log_turnover_expense_ratio"] = np.log1p(df["turnover_expense_ratio"].clip(lower=0))

    return df


# ── Dimension 3: Debt and repayment ability ───────────────────────────────────

def make_debt_features(df):
    """
    Build features capturing debt profile and repayment capacity.

    Formal debt (loan account) is a positive signal in this context - it
    means the business qualified for formal credit, which requires some
    financial standing. Informal debt (informal lender, friends/family)
    signals exclusion from formal channels, which is a weaker position.

    The distinction between formal and informal borrowing is central to
    how the FHI access and debt dimensions were likely constructed.
    """
    df = df.copy()

    # Currently using informal finance sources
    df["uses_informal_now"] = df[INFORMAL_COLS].apply(
        lambda r: int((r == 2).any()), axis=1
    )

    # Has formal loan currently
    df["has_formal_loan_now"] = (df["has_loan_account"] == 2).astype(int)

    # Formal vs informal debt contrast
    # Positive = relying on formal channels, negative = informal only
    df["formal_vs_informal"] = df["has_formal_loan_now"] - df["uses_informal_now"]

    # Tax compliance is a strong signal of formal engagement
    df["is_tax_compliant"] = df["compliance_income_tax"].fillna(0)

    # Compliance + formal loan = strong repayment credibility signal
    df["compliance_and_loan"] = (
        (df["compliance_income_tax"] == 1) &
        (df["has_loan_account"] == 2)
    ).astype(int)

    return df


# ── Dimension 4: Resilience to shocks ────────────────────────────────────────

def make_resilience_features(df):
    """
    Build features capturing shock resilience.

    Resilience combines: having insurance (cushion against losses),
    positive owner attitude (forward-looking mindset), not facing current
    cash flow problems, and being a covid essential service (operational
    continuity signal).

    Owner attitude features are composite signals - a business owner who
    is satisfied, optimistic, and not worried about shutdown is more likely
    to be in a stable financial position. These attitudes correlate with
    the underlying financial reality.
    """
    df = df.copy()

    # Positive attitude composite score
    df["positive_attitude_score"] = df[ATTITUDE_COLS].fillna(0).sum(axis=1)

    # Negative attitude/perception score (higher = more vulnerable)
    df["negative_attitude_score"] = df[NEGATIVE_ATTITUDE_COLS].fillna(0).sum(axis=1)

    # Net attitude score: positive minus negative
    df["net_attitude_score"] = df["positive_attitude_score"] - df["negative_attitude_score"]

    # Cash flow stability: not having a cash flow problem is resilience signal
    # current_problem_cash_flow=1 means problem exists (bad), 0 means stable
    df["cash_flow_stable"] = (1 - df["current_problem_cash_flow"].fillna(0.5)).clip(0, 1)

    # Theft risk awareness (future_risk_theft_stock=1 means at risk)
    # Not directly bad but relevant to resilience dimension
    df["theft_risk"] = df["future_risk_theft_stock"].fillna(0)

    # Essential service during covid - signals operational resilience
    df["covid_resilient"] = df["covid_essential_service"].fillna(0)

    # Overall resilience composite
    df["resilience_score"] = (
        df["insurance_strength"].fillna(0) +
        df["positive_attitude_score"] +
        df["cash_flow_stable"] +
        df["covid_resilient"]
    )

    return df


# ── Business maturity features ────────────────────────────────────────────────

def make_business_features(df):
    """
    Build features from business characteristics.

    Business age and size are baseline signals. A business that has survived
    longer has demonstrated some resilience. The combination of age and
    financial product access tells a richer story than either alone.
    """
    df = df.copy()

    # Total business age in months
    df["business_age_total_months"] = (
        df["business_age_years"].fillna(0) * 12 +
        df["business_age_months"].fillna(0)
    )

    # Log business age to compress outliers
    df["log_business_age"] = np.log1p(df["business_age_total_months"])

    # Owner age bands (younger owners may be more risk-tolerant but less established)
    df["owner_age_band"] = pd.cut(
        df["owner_age"],
        bins=[0, 25, 35, 45, 55, 120],
        labels=False
    ).astype(float)

    # Marketing reach: word of mouth marketing = community-embedded business
    df["markets_actively"] = df["marketing_word_of_mouth"].fillna(0)

    # Offers credit to customers - signals some financial confidence
    # already ordinal 0/1/2 from preprocess
    df["extends_credit"] = df["offers_credit_to_customers"].fillna(0)

    return df


# ── Country-normalised financial features ────────────────────────────────────

def make_country_features(df):
    """
    Normalise financial figures within country.

    Personal income and business turnover are in local currency. Comparing
    raw figures across countries is meaningless - a turnover of 50,000 means
    something very different in Malawi (MWK) vs Eswatini (SZL). We compute
    each business's percentile rank within its own country so comparisons
    are meaningful.

    This is computed on the full dataset (train+test combined) to avoid
    different distributions between splits. The percentile rank itself
    does not leak target information.
    """
    df = df.copy()

    for col in ["personal_income", "business_turnover", "business_expenses"]:
        log_col = f"log_{col}"
        if log_col in df.columns:
            df[f"{col}_country_rank"] = df.groupby("country")[log_col].rank(pct=True)

    return df


# ── High-value interaction features ──────────────────────────────────────────

def make_interaction_features(df):
    """
    Create interaction features based on known high-signal combinations.

    From the diagnostic analysis, we know:
    - has_insurance=1 AND has_credit_card=2 -> 100% High in training data
    - has_insurance=1 AND has_internet_banking=2 -> 53.7% High
    - has_loan_account=2 AND compliance=1 -> strong High signal

    We encode these explicitly rather than hoping the model discovers them,
    because tree models can find interactions but explicit features make the
    signal immediately available at the root splits.
    """
    df = df.copy()

    # The single strongest known interaction
    df["insured_with_credit_card"] = (
        (df["has_insurance"] == 1) &
        (df["has_credit_card"] == 2)
    ).astype(int)

    # Insurance with internet banking
    df["insured_with_internet_banking"] = (
        (df["has_insurance"] == 1) &
        (df["has_internet_banking"] == 2)
    ).astype(int)

    # Insurance with any banking product
    df["insured_with_banking"] = (
        (df["has_insurance"] == 1) &
        (df[["has_credit_card", "has_internet_banking", "has_debit_card"]] == 2).any(axis=1)
    ).astype(int)

    # Formal loan with tax compliance
    df["loan_and_compliant"] = (
        (df["has_loan_account"] == 2) &
        (df["compliance_income_tax"] == 1)
    ).astype(int)

    # Full formal inclusion: insurance + banking + loan
    df["fully_formal"] = (
        (df["has_insurance"] == 1) &
        (df["has_any_banking_now"] == 1) &
        (df["has_loan_account"] == 2)
    ).astype(int)

    # Records keeping combined with formal access
    df["records_and_formal"] = (
        (df["keeps_financial_records"] >= 2) &
        (df["formal_access_count"] >= 1)
    ).astype(int)

    # Country-level access tier: country encoded value * access score
    # Captures that high access in a high-FHI country is a stronger signal
    df["country_x_access"] = df["country"] * df["formal_access_count"]

    return df


# ── Master pipeline ───────────────────────────────────────────────────────────

def engineer_features(df):
    """
    Run all feature engineering steps in order.

    Steps must run in this order because some functions depend on columns
    created by earlier functions (e.g. make_resilience_features uses
    insurance_strength created by make_savings_features).
    """
    df = make_access_features(df)
    df = make_savings_features(df)
    df = make_debt_features(df)
    df = make_resilience_features(df)
    df = make_business_features(df)
    df = make_country_features(df)
    df = make_interaction_features(df)
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("Loading cleaned data...")
    train = pd.read_csv(PROCESSED_DIR / "train_clean.csv")
    test = pd.read_csv(PROCESSED_DIR / "test_clean.csv")
    print(f"  Train: {train.shape}, Test: {test.shape}")

    # Combine for country-normalised features so percentile ranks are
    # computed on the full distribution, not split between train and test
    print("Engineering features (combined train+test for country normalisation)...")
    combined = pd.concat([train, test], axis=0, ignore_index=True)
    combined = engineer_features(combined)

    # Split back
    train_out = combined.iloc[:len(train)].copy()
    test_out = combined.iloc[len(train):].copy()

    # Save
    train_out.to_csv(PROCESSED_DIR / "train_features.csv", index=False)
    test_out.to_csv(PROCESSED_DIR / "test_features.csv", index=False)

    print(f"  Saved train_features.csv: {train_out.shape}")
    print(f"  Saved test_features.csv:  {test_out.shape}")

    # Report new features added
    original_cols = set(train.columns)
    new_cols = [c for c in train_out.columns if c not in original_cols]
    print(f"\n{len(new_cols)} new features added:")
    for col in new_cols:
        print(f"  {col}")

    # Quick sanity check on key interaction features
    print("\nKey interaction feature distributions (train only):")
    key_features = [
        "insured_with_credit_card",
        "fully_formal",
        "formal_access_count",
        "resilience_score",
        "product_score_total",
    ]
    for feat in key_features:
        if feat in train_out.columns:
            print(f"  {feat}: mean={train_out[feat].mean():.3f}, "
                  f"max={train_out[feat].max():.1f}, "
                  f"missing={train_out[feat].isna().sum()}")


if __name__ == "__main__":
    main()