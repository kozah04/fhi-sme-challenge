"""
test_features.py

Unit tests for src/features.py.

Run from the project root with:
    pytest tests/test_features.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import (
    make_access_features,
    make_savings_features,
    make_debt_features,
    make_resilience_features,
    make_business_features,
    make_country_features,
    make_interaction_features,
    engineer_features,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def minimal_df():
    """
    A minimal dataframe with all columns that features.py expects.
    Values are set to known quantities so we can assert exact outputs.
    This represents a single fully-formal High-tier business.
    """
    return pd.DataFrame([{
        # Identity
        "ID": "ID_TEST01",
        "country": 3,  # eswatini

        # Financial product cols (ordinal: 2=Have now, 1=Used to have, 0=Never had)
        "motor_vehicle_insurance": 2,
        "has_mobile_money": 2,
        "has_credit_card": 2,
        "has_loan_account": 2,
        "has_internet_banking": 2,
        "has_debit_card": 2,
        "medical_insurance": 2,
        "funeral_insurance": 0,
        "uses_friends_family_savings": 0,
        "uses_informal_lender": 0,

        # Binary cols
        "has_insurance": 1,
        "has_cellphone": 1,
        "future_risk_theft_stock": 0,
        "problem_sourcing_money": 0,
        "marketing_word_of_mouth": 1,
        "motivation_make_more_money": 1,
        "covid_essential_service": 1,

        # Ordinal cols
        "keeps_financial_records": 3,
        "offers_credit_to_customers": 2,
        "compliance_income_tax": 1,

        # Attitude cols
        "attitude_stable_business_environment": 1,
        "attitude_worried_shutdown": 0,
        "attitude_satisfied_with_achievement": 1,
        "attitude_more_successful_next_year": 1,
        "perception_insurance_doesnt_cover_losses": 0,
        "perception_cannot_afford_insurance": 0,
        "perception_insurance_companies_dont_insure_businesses_like_yours": 0,
        "perception_insurance_important": 1,

        # Cash flow
        "current_problem_cash_flow": 0,

        # Financials
        "personal_income": 10000.0,
        "business_expenses": 5000.0,
        "business_turnover": 20000.0,
        "owner_age": 40.0,
        "owner_sex": 1,
        "business_age_years": 8.0,
        "business_age_months": 6.0,

        # Target
        "Target": 2,
    }])


@pytest.fixture
def low_tier_df():
    """A minimal Low-tier business with no formal financial products."""
    return pd.DataFrame([{
        "ID": "ID_TEST02",
        "country": 0,  # lesotho

        "motor_vehicle_insurance": 0,
        "has_mobile_money": 0,
        "has_credit_card": 0,
        "has_loan_account": 0,
        "has_internet_banking": 0,
        "has_debit_card": 0,
        "medical_insurance": 0,
        "funeral_insurance": 0,
        "uses_friends_family_savings": 2,
        "uses_informal_lender": 2,

        "has_insurance": 0,
        "has_cellphone": 1,
        "future_risk_theft_stock": 1,
        "problem_sourcing_money": 1,
        "marketing_word_of_mouth": 0,
        "motivation_make_more_money": 1,
        "covid_essential_service": 0,

        "keeps_financial_records": 0,
        "offers_credit_to_customers": 0,
        "compliance_income_tax": 0,

        "attitude_stable_business_environment": 0,
        "attitude_worried_shutdown": 1,
        "attitude_satisfied_with_achievement": 0,
        "attitude_more_successful_next_year": 0,
        "perception_insurance_doesnt_cover_losses": 1,
        "perception_cannot_afford_insurance": 1,
        "perception_insurance_companies_dont_insure_businesses_like_yours": 1,
        "perception_insurance_important": 0,

        "current_problem_cash_flow": 1,

        "personal_income": 500.0,
        "business_expenses": 800.0,
        "business_turnover": 1000.0,
        "owner_age": 28.0,
        "owner_sex": 0,
        "business_age_years": 1.0,
        "business_age_months": 3.0,

        "Target": 0,
    }])


# ── TestMakeAccessFeatures ────────────────────────────────────────────────────

class TestMakeAccessFeatures:

    def test_formal_access_count_counts_have_now_only(self, minimal_df):
        result = make_access_features(minimal_df)
        # minimal_df has: credit_card=2, loan=2, internet_banking=2,
        # debit_card=2, motor_vehicle=2, medical=2 -> 6 formal products
        assert result["formal_access_count"].iloc[0] == 6

    def test_formal_access_count_zero_for_low_tier(self, low_tier_df):
        result = make_access_features(low_tier_df)
        assert result["formal_access_count"].iloc[0] == 0

    def test_total_products_now_counts_all_have_now(self, minimal_df):
        result = make_access_features(minimal_df)
        # minimal_df: motor_vehicle=2, mobile_money=2, credit_card=2,
        # loan=2, internet=2, debit=2, medical=2 -> 7 (funeral=0, informal=0)
        assert result["total_products_now"].iloc[0] == 7

    def test_total_products_ever_includes_used_to_have(self):
        df = pd.DataFrame([{
            "motor_vehicle_insurance": 1,  # used to have
            "has_mobile_money": 2,          # have now
            "has_credit_card": 0,
            "has_loan_account": 0,
            "has_internet_banking": 0,
            "has_debit_card": 0,
            "medical_insurance": 0,
            "funeral_insurance": 0,
            "uses_friends_family_savings": 0,
            "uses_informal_lender": 0,
        }])
        result = make_access_features(df)
        assert result["total_products_ever"].iloc[0] == 2

    def test_has_any_banking_now_is_1_when_banking_present(self, minimal_df):
        result = make_access_features(minimal_df)
        assert result["has_any_banking_now"].iloc[0] == 1

    def test_has_any_banking_now_is_0_when_no_banking(self, low_tier_df):
        result = make_access_features(low_tier_df)
        assert result["has_any_banking_now"].iloc[0] == 0

    def test_product_score_total_is_sum_of_ordinals(self, low_tier_df):
        result = make_access_features(low_tier_df)
        # low_tier has uses_friends=2, uses_informal=2, rest=0 -> sum=4
        assert result["product_score_total"].iloc[0] == 4

    def test_has_loan_and_banking_requires_both(self):
        # Loan only, no banking
        df = pd.DataFrame([{
            "has_loan_account": 2,
            "has_credit_card": 0,
            "has_internet_banking": 0,
            "has_debit_card": 0,
            "motor_vehicle_insurance": 0,
            "has_mobile_money": 0,
            "medical_insurance": 0,
            "funeral_insurance": 0,
            "uses_friends_family_savings": 0,
            "uses_informal_lender": 0,
        }])
        result = make_access_features(df)
        assert result["has_loan_and_banking"].iloc[0] == 0


# ── TestMakeSavingsFeatures ───────────────────────────────────────────────────

class TestMakeSavingsFeatures:

    def test_log_personal_income_is_positive_for_positive_income(self, minimal_df):
        df = make_access_features(minimal_df)
        result = make_savings_features(df)
        assert result["log_personal_income"].iloc[0] > 0

    def test_log_personal_income_handles_zero(self):
        df = pd.DataFrame([{
            "personal_income": 0.0,
            "business_expenses": 1000.0,
            "business_turnover": 2000.0,
            "motor_vehicle_insurance": 0, "has_mobile_money": 0,
            "has_credit_card": 0, "has_loan_account": 0,
            "has_internet_banking": 0, "has_debit_card": 0,
            "medical_insurance": 0, "funeral_insurance": 0,
            "uses_friends_family_savings": 0, "uses_informal_lender": 0,
            "has_insurance": 0, "keeps_financial_records": 0,
        }])
        df = make_access_features(df)
        result = make_savings_features(df)
        # log1p(0) = 0, should not raise or produce NaN
        assert result["log_personal_income"].iloc[0] == 0.0
        assert not pd.isna(result["log_personal_income"].iloc[0])

    def test_income_expense_ratio_is_clipped_at_100(self):
        df = pd.DataFrame([{
            "personal_income": 1_000_000.0,
            "business_expenses": 1.0,
            "business_turnover": 2000.0,
            "motor_vehicle_insurance": 0, "has_mobile_money": 0,
            "has_credit_card": 0, "has_loan_account": 0,
            "has_internet_banking": 0, "has_debit_card": 0,
            "medical_insurance": 0, "funeral_insurance": 0,
            "uses_friends_family_savings": 0, "uses_informal_lender": 0,
            "has_insurance": 0, "keeps_financial_records": 0,
        }])
        df = make_access_features(df)
        result = make_savings_features(df)
        assert result["income_expense_ratio"].iloc[0] <= 100

    def test_insurance_strength_combines_flag_and_count(self, minimal_df):
        df = make_access_features(minimal_df)
        result = make_savings_features(df)
        # has_insurance=1, insurance products: motor=2, medical=2, funeral=0 -> count=2
        # insurance_strength = 1 + 2 = 3
        assert result["insurance_strength"].iloc[0] == 3

    def test_keeps_any_records_is_1_when_records_kept(self, minimal_df):
        df = make_access_features(minimal_df)
        result = make_savings_features(df)
        assert result["keeps_any_records"].iloc[0] == 1

    def test_keeps_any_records_is_0_when_no_records(self, low_tier_df):
        df = make_access_features(low_tier_df)
        result = make_savings_features(df)
        assert result["keeps_any_records"].iloc[0] == 0


# ── TestMakeDebtFeatures ──────────────────────────────────────────────────────

class TestMakeDebtFeatures:

    def _prep(self, df):
        df = make_access_features(df)
        df = make_savings_features(df)
        return make_debt_features(df)

    def test_uses_informal_now_is_1_when_informal_lender_active(self, low_tier_df):
        result = self._prep(low_tier_df)
        assert result["uses_informal_now"].iloc[0] == 1

    def test_uses_informal_now_is_0_for_high_tier(self, minimal_df):
        result = self._prep(minimal_df)
        assert result["uses_informal_now"].iloc[0] == 0

    def test_has_formal_loan_now_is_1_when_loan_active(self, minimal_df):
        result = self._prep(minimal_df)
        assert result["has_formal_loan_now"].iloc[0] == 1

    def test_formal_vs_informal_is_positive_for_formal_borrower(self, minimal_df):
        result = self._prep(minimal_df)
        assert result["formal_vs_informal"].iloc[0] > 0

    def test_formal_vs_informal_is_negative_for_informal_only(self, low_tier_df):
        result = self._prep(low_tier_df)
        assert result["formal_vs_informal"].iloc[0] < 0

    def test_compliance_and_loan_requires_both(self, low_tier_df):
        # low_tier has no loan and no compliance
        result = self._prep(low_tier_df)
        assert result["compliance_and_loan"].iloc[0] == 0


# ── TestMakeResilienceFeatures ────────────────────────────────────────────────

class TestMakeResilienceFeatures:

    def _prep(self, df):
        df = make_access_features(df)
        df = make_savings_features(df)
        df = make_debt_features(df)
        return make_resilience_features(df)

    def test_positive_attitude_score_higher_for_high_tier(self, minimal_df, low_tier_df):
        high_result = self._prep(minimal_df)
        low_result = self._prep(low_tier_df)
        assert high_result["positive_attitude_score"].iloc[0] > low_result["positive_attitude_score"].iloc[0]

    def test_negative_attitude_score_higher_for_low_tier(self, minimal_df, low_tier_df):
        high_result = self._prep(minimal_df)
        low_result = self._prep(low_tier_df)
        assert low_result["negative_attitude_score"].iloc[0] > high_result["negative_attitude_score"].iloc[0]

    def test_cash_flow_stable_is_1_when_no_problem(self, minimal_df):
        result = self._prep(minimal_df)
        assert result["cash_flow_stable"].iloc[0] == 1.0

    def test_cash_flow_stable_is_0_when_problem_exists(self, low_tier_df):
        result = self._prep(low_tier_df)
        assert result["cash_flow_stable"].iloc[0] == 0.0

    def test_resilience_score_higher_for_high_tier(self, minimal_df, low_tier_df):
        high_result = self._prep(minimal_df)
        low_result = self._prep(low_tier_df)
        assert high_result["resilience_score"].iloc[0] > low_result["resilience_score"].iloc[0]


# ── TestMakeBusinessFeatures ──────────────────────────────────────────────────

class TestMakeBusinessFeatures:

    def _prep(self, df):
        df = make_access_features(df)
        df = make_savings_features(df)
        df = make_debt_features(df)
        df = make_resilience_features(df)
        return make_business_features(df)

    def test_business_age_total_months_combines_years_and_months(self, minimal_df):
        result = self._prep(minimal_df)
        # 8 years * 12 + 6 months = 102
        assert result["business_age_total_months"].iloc[0] == 102

    def test_log_business_age_is_positive(self, minimal_df):
        result = self._prep(minimal_df)
        assert result["log_business_age"].iloc[0] > 0

    def test_owner_age_band_is_not_null_for_valid_age(self, minimal_df):
        result = self._prep(minimal_df)
        assert not pd.isna(result["owner_age_band"].iloc[0])


# ── TestMakeInteractionFeatures ───────────────────────────────────────────────

class TestMakeInteractionFeatures:

    def _prep(self, df):
        df = make_access_features(df)
        df = make_savings_features(df)
        df = make_debt_features(df)
        df = make_resilience_features(df)
        df = make_business_features(df)
        return make_interaction_features(df)

    def test_insured_with_credit_card_is_1_when_both_present(self, minimal_df):
        result = self._prep(minimal_df)
        # has_insurance=1, has_credit_card=2
        assert result["insured_with_credit_card"].iloc[0] == 1

    def test_insured_with_credit_card_is_0_when_no_insurance(self, low_tier_df):
        result = self._prep(low_tier_df)
        assert result["insured_with_credit_card"].iloc[0] == 0

    def test_fully_formal_is_1_for_high_tier(self, minimal_df):
        result = self._prep(minimal_df)
        # has_insurance=1, has_any_banking_now=1, has_loan_account=2
        assert result["fully_formal"].iloc[0] == 1

    def test_fully_formal_is_0_for_low_tier(self, low_tier_df):
        result = self._prep(low_tier_df)
        assert result["fully_formal"].iloc[0] == 0

    def test_loan_and_compliant_requires_both(self, minimal_df):
        result = self._prep(minimal_df)
        # has_loan_account=2, compliance_income_tax=1
        assert result["loan_and_compliant"].iloc[0] == 1

    def test_records_and_formal_is_1_when_both_conditions_met(self, minimal_df):
        result = self._prep(minimal_df)
        # keeps_financial_records=3 (>=2), formal_access_count=6 (>=1)
        assert result["records_and_formal"].iloc[0] == 1


# ── TestEngineerFeatures ──────────────────────────────────────────────────────

class TestEngineerFeatures:

    def test_output_has_more_columns_than_input(self, minimal_df):
        result = engineer_features(minimal_df)
        assert result.shape[1] > minimal_df.shape[1]

    def test_no_new_string_columns_introduced(self, minimal_df):
        result = engineer_features(minimal_df)
        non_id = [c for c in result.columns if c != "ID"]
        string_cols = result[non_id].select_dtypes(include=["object"]).columns.tolist()
        assert len(string_cols) == 0, f"String columns found: {string_cols}"

    def test_original_columns_are_preserved(self, minimal_df):
        original_cols = set(minimal_df.columns)
        result = engineer_features(minimal_df)
        for col in original_cols:
            assert col in result.columns, f"Original column lost: {col}"

    def test_high_tier_scores_higher_than_low_tier(self, minimal_df, low_tier_df):
        high_result = engineer_features(minimal_df)
        low_result = engineer_features(low_tier_df)
        # Both composite scores should rank high tier above low tier
        assert high_result["product_score_total"].iloc[0] > low_result["product_score_total"].iloc[0]
        assert high_result["resilience_score"].iloc[0] > low_result["resilience_score"].iloc[0]
        assert high_result["formal_access_count"].iloc[0] > low_result["formal_access_count"].iloc[0]