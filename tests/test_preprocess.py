"""
test_preprocess.py

Unit tests for src/preprocess.py.

Run from the project root with:
    pytest tests/test_preprocess.py -v
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocess import (
    normalise_string,
    encode_product_col,
    encode_binary_col,
    encode_ternary_col,
    encode_keeps_financial_records,
    encode_offers_credit,
    encode_owner_sex,
    encode_country,
    clean,
    add_missing_flags,
    TARGET_MAP,
    PRODUCT_COLS,
    BINARY_COLS,
)


# ── TestNormaliseString ──────────────────────────────────────────────────────

class TestNormaliseString:

    def test_lowercases_input(self):
        assert normalise_string("YES") == "yes"

    def test_strips_whitespace(self):
        assert normalise_string("  No  ") == "no"

    def test_replaces_unicode_apostrophe(self):
        # Right single quotation mark (U+2019) found in raw data
        result = normalise_string("Don\u2019t know")
        assert "'" in result
        assert "\u2019" not in result

    def test_replaces_mangled_apostrophe(self):
        # Question mark used as apostrophe in some raw rows
        result = normalise_string("Don?t know")
        assert "'" in result

    def test_strips_left_to_right_mark(self):
        # U+200E found in perception_insurance_important column
        result = normalise_string("Do not know / N\u200e/A")
        assert "\u200e" not in result

    def test_returns_nan_for_nan_input(self):
        assert pd.isna(normalise_string(np.nan))


# ── TestEncodeProductCol ─────────────────────────────────────────────────────

class TestEncodeProductCol:

    def test_have_now_encodes_to_2(self):
        s = pd.Series(["Have now"])
        assert encode_product_col(s).iloc[0] == 2

    def test_used_to_have_encodes_to_1(self):
        s = pd.Series(["Used to have but don't have now"])
        assert encode_product_col(s).iloc[0] == 1

    def test_never_had_encodes_to_0(self):
        s = pd.Series(["Never had"])
        assert encode_product_col(s).iloc[0] == 0

    def test_dont_know_becomes_nan(self):
        s = pd.Series(["Don't know"])
        assert pd.isna(encode_product_col(s).iloc[0])

    def test_nan_input_stays_nan(self):
        s = pd.Series([np.nan])
        assert pd.isna(encode_product_col(s).iloc[0])

    def test_mixed_series_encodes_correctly(self):
        s = pd.Series(["Have now", "Never had", "Used to have but don't have now", np.nan])
        result = encode_product_col(s)
        assert result.iloc[0] == 2
        assert result.iloc[1] == 0
        assert result.iloc[2] == 1
        assert pd.isna(result.iloc[3])

    def test_output_is_numeric(self):
        s = pd.Series(["Have now", "Never had"])
        result = encode_product_col(s)
        assert pd.api.types.is_float_dtype(result) or pd.api.types.is_integer_dtype(result)


# ── TestEncodeBinaryCol ──────────────────────────────────────────────────────

class TestEncodeBinaryCol:

    def test_yes_encodes_to_1(self):
        s = pd.Series(["Yes"])
        assert encode_binary_col(s).iloc[0] == 1

    def test_no_encodes_to_0(self):
        s = pd.Series(["No"])
        assert encode_binary_col(s).iloc[0] == 0

    def test_case_insensitive(self):
        s = pd.Series(["YES", "NO"])
        result = encode_binary_col(s)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0

    def test_dont_know_becomes_nan(self):
        s = pd.Series(["Don't know"])
        assert pd.isna(encode_binary_col(s).iloc[0])

    def test_nan_stays_nan(self):
        s = pd.Series([np.nan])
        assert pd.isna(encode_binary_col(s).iloc[0])


# ── TestEncodeTernaryCol ─────────────────────────────────────────────────────

class TestEncodeTernaryCol:

    def test_yes_encodes_to_1(self):
        s = pd.Series(["Yes"])
        assert encode_ternary_col(s, "col").iloc[0] == 1

    def test_no_encodes_to_0(self):
        s = pd.Series(["No"])
        assert encode_ternary_col(s, "col").iloc[0] == 0

    def test_string_zero_encodes_to_0(self):
        # current_problem_cash_flow has stray "0" values in raw data
        s = pd.Series(["0"])
        assert encode_ternary_col(s, "current_problem_cash_flow").iloc[0] == 0

    def test_dont_know_becomes_nan(self):
        for val in ["Don't know", "Don't know or N/A", "Refused", "Don't Know"]:
            s = pd.Series([val])
            assert pd.isna(encode_ternary_col(s, "col").iloc[0]), f"Expected NaN for: {val}"

    def test_nan_stays_nan(self):
        s = pd.Series([np.nan])
        assert pd.isna(encode_ternary_col(s, "col").iloc[0])


# ── TestEncodeKeepsFinancialRecords ──────────────────────────────────────────

class TestEncodeKeepsFinancialRecords:

    def test_yes_always_encodes_to_3(self):
        s = pd.Series(["Yes, always"])
        assert encode_keeps_financial_records(s).iloc[0] == 3

    def test_yes_sometimes_encodes_to_2(self):
        s = pd.Series(["Yes, sometimes"])
        assert encode_keeps_financial_records(s).iloc[0] == 2

    def test_yes_encodes_to_1(self):
        s = pd.Series(["Yes"])
        assert encode_keeps_financial_records(s).iloc[0] == 1

    def test_no_encodes_to_0(self):
        s = pd.Series(["No"])
        assert encode_keeps_financial_records(s).iloc[0] == 0

    def test_ordinal_direction_is_correct(self):
        # Always > Sometimes > Yes > No
        s = pd.Series(["Yes, always", "Yes, sometimes", "Yes", "No"])
        result = encode_keeps_financial_records(s)
        assert result.iloc[0] > result.iloc[1] > result.iloc[2] > result.iloc[3]


# ── TestEncodeOffersCredit ───────────────────────────────────────────────────

class TestEncodeOffersCredit:

    def test_yes_always_encodes_to_2(self):
        s = pd.Series(["Yes, always"])
        assert encode_offers_credit(s).iloc[0] == 2

    def test_yes_sometimes_encodes_to_1(self):
        s = pd.Series(["Yes, sometimes"])
        assert encode_offers_credit(s).iloc[0] == 1

    def test_no_encodes_to_0(self):
        s = pd.Series(["No"])
        assert encode_offers_credit(s).iloc[0] == 0

    def test_ordinal_direction_is_correct(self):
        s = pd.Series(["Yes, always", "Yes, sometimes", "No"])
        result = encode_offers_credit(s)
        assert result.iloc[0] > result.iloc[1] > result.iloc[2]


# ── TestEncodeOwnerSex ───────────────────────────────────────────────────────

class TestEncodeOwnerSex:

    def test_male_encodes_to_1(self):
        s = pd.Series(["Male"])
        assert encode_owner_sex(s).iloc[0] == 1

    def test_female_encodes_to_0(self):
        s = pd.Series(["Female"])
        assert encode_owner_sex(s).iloc[0] == 0

    def test_case_insensitive(self):
        s = pd.Series(["MALE", "FEMALE"])
        result = encode_owner_sex(s)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 0


# ── TestEncodeCountry ────────────────────────────────────────────────────────

class TestEncodeCountry:

    def test_all_four_countries_map_to_integers(self):
        s = pd.Series(["eswatini", "lesotho", "malawi", "zimbabwe"])
        result = encode_country(s)
        assert result.notna().all()

    def test_eswatini_has_highest_value(self):
        # Eswatini has the highest High class rate so gets the highest ordinal
        s = pd.Series(["eswatini", "lesotho", "malawi", "zimbabwe"])
        result = encode_country(s)
        assert result.iloc[0] == result.max()

    def test_lesotho_has_lowest_value(self):
        s = pd.Series(["eswatini", "lesotho", "malawi", "zimbabwe"])
        result = encode_country(s)
        assert result.iloc[1] == result.min()


# ── TestClean ────────────────────────────────────────────────────────────────

class TestClean:

    @pytest.fixture
    def sample_train_row(self):
        """A minimal but realistic train row covering all encoding paths."""
        return pd.DataFrame([{
            "ID": "ID_TEST01",
            "country": "eswatini",
            "owner_age": 35.0,
            "attitude_stable_business_environment": "Yes",
            "attitude_worried_shutdown": "No",
            "compliance_income_tax": "Yes",
            "perception_insurance_doesnt_cover_losses": "No",
            "perception_cannot_afford_insurance": "Yes",
            "personal_income": 5000.0,
            "business_expenses": 2000.0,
            "business_turnover": 10000.0,
            "business_age_years": 5.0,
            "motor_vehicle_insurance": "Have now",
            "has_mobile_money": "Never had",
            "current_problem_cash_flow": "No",
            "has_cellphone": "Yes",
            "owner_sex": "Female",
            "offers_credit_to_customers": "Yes, sometimes",
            "attitude_satisfied_with_achievement": "Yes",
            "has_credit_card": "Have now",
            "keeps_financial_records": "Yes, always",
            "perception_insurance_companies_dont_insure_businesses_like_yours": "No",
            "perception_insurance_important": "Yes",
            "has_insurance": "Yes",
            "covid_essential_service": "Yes",
            "attitude_more_successful_next_year": "Yes",
            "problem_sourcing_money": "No",
            "marketing_word_of_mouth": "Yes",
            "has_loan_account": "Never had",
            "has_internet_banking": "Have now",
            "has_debit_card": "Have now",
            "future_risk_theft_stock": "No",
            "business_age_months": 6.0,
            "medical_insurance": "Have now",
            "funeral_insurance": "Never had",
            "motivation_make_more_money": "Yes",
            "uses_friends_family_savings": "Never had",
            "uses_informal_lender": "Never had",
            "Target": "High",
        }])

    def test_target_is_encoded_as_integer(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        assert result["Target"].iloc[0] == TARGET_MAP["High"]

    def test_no_string_columns_remain_after_cleaning(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        non_id_cols = [c for c in result.columns if c != "ID"]
        string_cols = result[non_id_cols].select_dtypes(include=["object"]).columns.tolist()
        assert len(string_cols) == 0, f"String columns still present: {string_cols}"

    def test_product_col_have_now_encodes_to_2(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        assert result["motor_vehicle_insurance"].iloc[0] == 2

    def test_product_col_never_had_encodes_to_0(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        assert result["has_mobile_money"].iloc[0] == 0

    def test_binary_yes_encodes_to_1(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        assert result["has_cellphone"].iloc[0] == 1

    def test_id_column_is_preserved(self, sample_train_row):
        result = clean(sample_train_row, is_train=True)
        assert "ID" in result.columns
        assert result["ID"].iloc[0] == "ID_TEST01"

    def test_is_train_false_does_not_encode_target(self, sample_train_row):
        test_row = sample_train_row.drop(columns=["Target"])
        result = clean(test_row, is_train=False)
        assert "Target" not in result.columns


# ── TestAddMissingFlags ──────────────────────────────────────────────────────

class TestAddMissingFlags:

    def test_flag_created_for_high_missing_column(self):
        # Column with >20% missing should get a _missing flag
        df = pd.DataFrame({
            "ID": range(10),
            "some_col": [np.nan] * 3 + [1.0] * 7,  # 30% missing
        })
        result = add_missing_flags(df)
        assert "some_col_missing" in result.columns

    def test_flag_not_created_for_low_missing_column(self):
        # Column with <=20% missing should not get a flag
        df = pd.DataFrame({
            "ID": range(10),
            "some_col": [np.nan] * 1 + [1.0] * 9,  # 10% missing
        })
        result = add_missing_flags(df)
        assert "some_col_missing" not in result.columns

    def test_flag_values_are_binary(self):
        df = pd.DataFrame({
            "ID": range(10),
            "some_col": [np.nan] * 5 + [1.0] * 5,
        })
        result = add_missing_flags(df)
        flag_vals = result["some_col_missing"].unique()
        assert set(flag_vals).issubset({0, 1})

    def test_flag_is_1_where_value_is_missing(self):
        df = pd.DataFrame({
            "ID": range(10),
            "some_col": [np.nan] * 5 + [1.0] * 5,
        })
        result = add_missing_flags(df)
        assert result.loc[result["some_col"].isna(), "some_col_missing"].eq(1).all()

    def test_original_columns_are_not_dropped(self):
        df = pd.DataFrame({
            "ID": range(10),
            "some_col": [np.nan] * 5 + [1.0] * 5,
        })
        result = add_missing_flags(df)
        assert "some_col" in result.columns