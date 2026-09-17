# ============================================================
# PYTEST UNIT TESTS — Health Insurance Cross-Sell ML System
# ============================================================
# Run with:  pytest tests/test_pipeline_core.py -v
#            pytest tests/ -v --cov=src --cov-report=term-missing
#
# Tests cover:
#   Clipper, build_preprocessors, detect_feature_types,
#   detect_leakage, tune_threshold, psi, recall_at_k,
#   lift_at_k, ks_statistic, get_propensity_band, score_customer,
#   config thresholds, feature engineering, cost evaluation
# ============================================================

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import pytest
import numpy as np
import pandas as pd

from src.preprocessing  import Clipper, build_preprocessors
from src.data_loader    import detect_feature_types, add_engineered_features, validate_input_data
from src.leakage_check  import detect_leakage
from src.metrics        import tune_threshold, psi, recall_at_k, lift_at_k, ks_statistic, cost_sensitive_evaluation
from src.lead_engine    import get_propensity_band, score_customer
from src.config         import (
    PROPENSITY_BANDS, PSI_MODERATE, PSI_HIGH,
    MIN_F1_IMPROVEMENT, MIN_ROCAUC_THRESHOLD, MAX_GENERALIZATION_GAP,
    PREVIOUSLY_INSURED_AUTO_SKIP, REQUIRE_DRIVING_LICENSE
)


# ============================================================
# CLIPPER TESTS  (6 tests)
# ============================================================

class TestClipper:

    def test_fit_transform_shape(self):
        X = np.array([[1.0], [1000.0], [2.0], [3.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape == X.shape

    def test_clips_outliers(self):
        X = np.array([[1.0], [2.0], [3.0], [9999.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).max() < 9999.0, "Outlier should have been clipped"

    def test_no_change_on_normal_data(self):
        X = np.array([[10.0], [11.0], [12.0], [13.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        np.testing.assert_array_almost_equal(X, clip.transform(X), decimal=3)

    def test_1d_input(self):
        X = np.array([1.0, 2.0, 3.0, 1000.0])
        clip = Clipper(fold=1.5)
        clip.fit(X)
        assert clip.transform(X).shape[0] == 4

    def test_get_feature_names_out(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        clip = Clipper()
        clip.fit(X)
        names = clip.get_feature_names_out(["a", "b"])
        assert list(names) == ["a", "b"]

    def test_fit_on_train_applied_to_test(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        X_test  = np.array([[100.0]])
        clip = Clipper(fold=1.5)
        clip.fit(X_train)
        clipped = clip.transform(X_test)
        assert clipped[0, 0] < 100.0, "Test outlier should be clipped to training bounds"


# ============================================================
# PREPROCESSOR TESTS  (5 tests)
# ============================================================

class TestBuildPreprocessors:

    def _sample_df(self):
        return pd.DataFrame({
            "age":                [25, 35, 45, 55, 65],
            "annual_premium":     [20000.0, 30000.0, 40000.0, 50000.0, 60000.0],
            "vehicle_age":        ["< 1 Year", "1-2 Year", "> 2 Years", "1-2 Year", "< 1 Year"],
            "region_code":        [1.0, 2.0, 3.0, 4.0, 5.0],
            "driving_license":    [1, 1, 1, 0, 1],
        })

    def test_returns_four_outputs(self):
        df = self._sample_df()
        result = build_preprocessors(["vehicle_age"], ["age", "annual_premium"], ["driving_license"], df)
        assert len(result) == 4

    def test_categorical_indices_are_list(self):
        df = self._sample_df()
        _, _, cat_idx, _ = build_preprocessors(["vehicle_age"], ["age", "annual_premium"], ["driving_license"], df)
        assert isinstance(cat_idx, list)

    def test_feature_order_coverage(self):
        df = self._sample_df()
        ord_cols  = ["vehicle_age"]
        cont_cols = ["age", "annual_premium"]
        bin_cols  = ["driving_license"]
        _, _, _, feat_order = build_preprocessors(ord_cols, cont_cols, bin_cols, df)
        for col in ord_cols + cont_cols + bin_cols:
            assert col in feat_order

    def test_scaled_preprocessor_transforms(self):
        df = self._sample_df()
        pre_scaled, _, _, _ = build_preprocessors(["vehicle_age"], ["age", "annual_premium"], ["driving_license"], df)
        out = pre_scaled.fit_transform(df)
        assert not np.isnan(out).any()

    def test_unscaled_preprocessor_transforms(self):
        df = self._sample_df()
        _, pre_unscaled, _, _ = build_preprocessors(["vehicle_age"], ["age", "annual_premium"], ["driving_license"], df)
        out = pre_unscaled.fit_transform(df)
        assert not np.isnan(out).any()


# ============================================================
# FEATURE TYPE DETECTION TESTS  (3 tests)
# ============================================================

class TestDetectFeatureTypes:

    def test_binary_detected(self):
        df = pd.DataFrame({
            "driving_license": [0, 1, 0, 1, 0],
            "response":        [0, 1, 0, 0, 1]
        })
        _, _, bin_cols = detect_feature_types(df, threshold=10)
        assert "driving_license" in bin_cols

    def test_target_excluded(self):
        df = pd.DataFrame({
            "annual_premium": [10.0, 20.0, 30.0, 40.0, 50.0],
            "response":       [0, 1, 0, 0, 1]
        })
        ord_cols, cont_cols, bin_cols = detect_feature_types(df, threshold=10)
        assert "response" not in ord_cols + cont_cols + bin_cols

    def test_continuous_detected(self):
        df = pd.DataFrame({
            "annual_premium": [10.5, 20.1, 55.3, 80.0, 110.5, 200.3,
                               15.0, 25.0, 35.0, 45.0, 55.0, 65.0],
            "response":       [0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        })
        _, cont_cols, _ = detect_feature_types(df, threshold=10)
        assert "annual_premium" in cont_cols


# ============================================================
# FEATURE ENGINEERING TESTS  (4 tests)
# ============================================================

class TestFeatureEngineering:

    def _raw_df(self):
        return pd.DataFrame({
            "gender": ["Male", "Female"],
            "age": [30, 45],
            "driving_license": [1, 1],
            "region_code": [28.0, 3.0],
            "previously_insured": [0, 1],
            "vehicle_age": ["1-2 Year", "< 1 Year"],
            "vehicle_damage": ["Yes", "No"],
            "annual_premium": [30000.0, 2630.0],
            "policy_sales_channel": [152.0, 26.0],
            "vintage": [150, 80],
            "response": [1, 0],
        })

    def test_adds_expected_columns(self):
        df = add_engineered_features(self._raw_df())
        for col in ["vehicle_damage_flag", "is_new_vehicle", "damaged_and_uninsured",
                    "premium_per_vintage_day", "age_bin", "high_premium_flag"]:
            assert col in df.columns

    def test_damaged_and_uninsured_logic(self):
        df = add_engineered_features(self._raw_df())
        assert df.loc[0, "damaged_and_uninsured"] == 1   # Yes damage + not previously insured
        assert df.loc[1, "damaged_and_uninsured"] == 0   # No damage

    def test_vehicle_damage_flag_binary(self):
        df = add_engineered_features(self._raw_df())
        assert set(df["vehicle_damage_flag"].unique()).issubset({0, 1})

    def test_no_nulls_introduced(self):
        df = add_engineered_features(self._raw_df())
        assert df.isnull().sum().sum() == 0


# ============================================================
# LEAKAGE CHECK TESTS  (4 tests)
# ============================================================

class TestDetectLeakage:

    def test_catches_identical_feature(self):
        X = pd.DataFrame({"a": [1, 0, 1, 0, 1]})
        y = pd.Series(        [1, 0, 1, 0, 1])
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) > 0
        assert any("a" in w for w in warnings)

    def test_no_false_positives_on_clean_data(self):
        np.random.seed(42)
        X = pd.DataFrame({"annual_premium": np.random.uniform(2630, 60000, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        warnings = detect_leakage(X, y, threshold_corr=0.99)
        assert len(warnings) == 0

    def test_catches_high_correlation(self):
        vals = np.arange(50, dtype=float)
        X    = pd.DataFrame({"leaky_feat": vals})
        y    = pd.Series((vals > 25).astype(int))
        warnings = detect_leakage(X, y, threshold_corr=0.85)
        assert len(warnings) > 0

    def test_empty_dataframe_no_crash(self):
        X = pd.DataFrame()
        y = pd.Series([0, 1, 0, 1])
        warnings = detect_leakage(X, y)
        assert isinstance(warnings, list)


# ============================================================
# THRESHOLD TUNING TESTS  (2 tests)
# ============================================================

class TestTuneThreshold:

    def test_returns_float_in_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200)
        y_prob = np.random.uniform(0, 1, 200)
        thr = tune_threshold(y_true, y_prob)
        assert isinstance(thr, float)
        assert 0.0 <= thr <= 1.0

    def test_perfect_prob_gives_low_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        thr = tune_threshold(y_true, y_prob)
        assert thr <= 0.9, f"Threshold {thr} should be <= 0.9 for perfectly separated data"


# ============================================================
# PSI TESTS  (3 tests)
# ============================================================

class TestPSI:

    def test_identical_distributions(self):
        x = np.random.normal(0, 1, 500)
        assert psi(x, x) < 0.05

    def test_shifted_distribution_higher_psi(self):
        rng = np.random.RandomState(42)
        ref = rng.normal(0, 1, 1000)
        new = rng.normal(3, 1, 1000)
        assert psi(ref, new) > psi(ref, ref)

    def test_uses_reference_edges_not_actual(self):
        rng = np.random.RandomState(0)
        ref = rng.normal(0, 1, 500)
        new = rng.normal(5, 1, 500)
        score = psi(ref, new)
        assert score > 0.20, f"Completely shifted distribution should have PSI >> 0.20, got {score:.4f}"


# ============================================================
# RECALL @ K TESTS  (2 tests)
# ============================================================

class TestRecallAtK:

    def test_top_scores_captured(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        r = recall_at_k(y, p, k=0.3)
        assert r >= 0.66

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(recall_at_k(y, p), float)


# ============================================================
# LIFT @ K TESTS  (2 tests)
# ============================================================

class TestLiftAtK:

    def test_lift_above_one_for_good_model(self):
        y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 1])
        p = np.array([0.9, 0.1, 0.2, 0.8, 0.3, 0.05, 0.1, 0.15, 0.2, 0.7])
        assert lift_at_k(y, p, k=0.3) > 1.0

    def test_returns_float(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.1, 0.9, 0.3, 0.8])
        assert isinstance(lift_at_k(y, p), float)


# ============================================================
# KS STATISTIC TESTS  (2 tests)
# ============================================================

class TestKSStatistic:

    def test_returns_float_in_range(self):
        np.random.seed(1)
        y = np.random.randint(0, 2, 100)
        p = np.random.uniform(0, 1, 100)
        ks = ks_statistic(y, p)
        assert isinstance(ks, float)
        assert 0.0 <= ks <= 1.0

    def test_perfect_model_high_ks(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        p = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
        assert ks_statistic(y, p) > 0.7


# ============================================================
# COST-SENSITIVE EVALUATION TESTS  (2 tests)
# ============================================================

class TestCostSensitiveEvaluation:

    def test_returns_expected_keys(self):
        X = pd.DataFrame({"annual_premium": [30000.0, 25000.0, 40000.0, 20000.0]})
        y_true = pd.Series([1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 0])
        result = cost_sensitive_evaluation(X, y_true, y_pred)
        for key in ("false_negative_count", "false_positive_count", "true_positive_count",
                    "estimated_missed_revenue", "wasted_call_cost", "net_estimated_value"):
            assert key in result

    def test_no_calls_zero_cost(self):
        X = pd.DataFrame({"annual_premium": [30000.0, 25000.0]})
        y_true = pd.Series([0, 0])
        y_pred = np.array([0, 0])
        result = cost_sensitive_evaluation(X, y_true, y_pred)
        assert result["wasted_call_cost"] == 0.0


# ============================================================
# LEAD ENGINE TESTS  (8 tests)
# ============================================================

class TestLeadEngine:

    def test_get_propensity_band_low(self):
        assert get_propensity_band(0.10) == "LOW"

    def test_get_propensity_band_medium(self):
        assert get_propensity_band(0.45) == "MEDIUM"

    def test_get_propensity_band_high(self):
        assert get_propensity_band(0.75) == "HIGH"

    def test_score_customer_cold_lead_low_prob(self):
        row    = {"previously_insured": 0, "driving_license": 1}
        result = score_customer(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "COLD_LEAD"
        assert result["propensity_band"] == "LOW"

    def test_score_customer_previously_insured_rule(self):
        """Already-insured customer must trigger the hard COLD_LEAD rule."""
        row    = {"previously_insured": 1, "driving_license": 1}
        result = score_customer(row, prob=0.05, threshold=0.5)
        assert result["decision"] == "COLD_LEAD"
        assert result["rule_triggered"] == "ALREADY_VEHICLE_INSURED"

    def test_score_customer_no_license_rule(self):
        """No driving license must trigger the hard COLD_LEAD rule."""
        row    = {"previously_insured": 0, "driving_license": 0}
        result = score_customer(row, prob=0.9, threshold=0.5)
        assert result["decision"] == "COLD_LEAD"
        assert result["rule_triggered"] == "NO_DRIVING_LICENSE"

    def test_score_customer_warm_lead(self):
        row    = {"previously_insured": 0, "driving_license": 1}
        result = score_customer(row, prob=0.35, threshold=0.5)
        assert result["decision"] == "WARM_LEAD"

    def test_score_customer_hot_lead(self):
        row    = {"previously_insured": 0, "driving_license": 1}
        result = score_customer(row, prob=0.9, threshold=0.5)
        assert result["decision"] == "HOT_LEAD"
        assert result["rule_triggered"] is None

    def test_output_keys_complete(self):
        row    = {"previously_insured": 0, "driving_license": 1}
        result = score_customer(row, prob=0.2, threshold=0.5)
        for key in ("cross_sell_probability", "propensity_band", "decision", "rule_triggered"):
            assert key in result


# ============================================================
# CONFIG TESTS  (5 tests)
# ============================================================

class TestConfig:

    def test_psi_thresholds_ordered(self):
        assert PSI_MODERATE < PSI_HIGH

    def test_challenger_gates_reasonable(self):
        assert 0.0 < MIN_F1_IMPROVEMENT    < 0.1
        assert 0.5 < MIN_ROCAUC_THRESHOLD  < 1.0
        assert 0.0 < MAX_GENERALIZATION_GAP < 1.0

    def test_propensity_band_boundaries_valid(self):
        for band, (low, high) in PROPENSITY_BANDS.items():
            assert 0.0 <= low  <= 1.0, f"{band} low boundary invalid"
            assert 0.0 <= high <= 1.1, f"{band} high boundary invalid"

    def test_business_rule_flags_are_bool(self):
        assert isinstance(PREVIOUSLY_INSURED_AUTO_SKIP, bool)
        assert isinstance(REQUIRE_DRIVING_LICENSE, bool)

    def test_business_rules_enabled_by_default(self):
        assert PREVIOUSLY_INSURED_AUTO_SKIP is True
        assert REQUIRE_DRIVING_LICENSE is True


# ============================================================
# DATA VALIDATION TESTS  (3 tests)
# ============================================================

class TestValidateInputData:

    def _valid_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3] * 40,
            "Gender": ["Male", "Female", "Male"] * 40,
            "Age": [25, 35, 45] * 40,
            "Driving_License": [1, 1, 1] * 40,
            "Region_Code": [1.0, 2.0, 3.0] * 40,
            "Previously_Insured": [0, 1, 0] * 40,
            "Vehicle_Age": ["< 1 Year", "1-2 Year", "> 2 Years"] * 40,
            "Vehicle_Damage": ["Yes", "No", "Yes"] * 40,
            "Annual_Premium": [30000.0, 2630.0, 45000.0] * 40,
            "Policy_Sales_Channel": [152.0, 26.0, 124.0] * 40,
            "Vintage": [100, 150, 200] * 40,
            "Response": [1, 0, 0] * 40,
        })

    def test_normalizes_column_names(self):
        df = validate_input_data(self._valid_df())
        assert "annual_premium" in df.columns
        assert "Annual_Premium" not in df.columns

    def test_drops_id_column(self):
        df = validate_input_data(self._valid_df())
        assert "id" not in df.columns

    def test_raises_on_missing_column(self):
        df = self._valid_df().drop(columns=["Vehicle_Damage"])
        with pytest.raises(ValueError):
            validate_input_data(df)
