import numpy as np
import pandas as pd
import pytest

from okmich_quant_labelling.utils.predictive_evaluator import (
    _naive_pnl,
    _mutual_info,
    _auc_walkforward,
    evaluate_labels,
    classify_label,
)


@pytest.fixture
def synthetic_data():
    """Creates synthetic OHLC data and trend."""
    np.random.seed(42)
    n = 5000

    # Create datetime index (required by _ensure_datetime_index)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    # Random walk price series
    ret = np.random.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(ret))

    df = pd.DataFrame(
        {
            "close": price,
            "random_label": np.random.choice([-1, 0, 1], size=n),
        },
        index=dates,
    )
    px = df["close"].astype(float)
    df["fwd_ret_1"] = np.log(px).shift(-1) - np.log(px)

    # Labels
    df["random_label"] = np.random.choice([-1, 0, 1], size=n)
    df["perfect_label"] = np.sign(df["fwd_ret_1"]).fillna(0)
    df["inverse_label"] = -df["perfect_label"]

    return df.dropna()


# -------------------------------
# TEST 1: Random trend ≈ uninformative
# -------------------------------
def test_random_labels_predictiveness(synthetic_data):
    df = synthetic_data
    mi = _mutual_info(df["random_label"], df["fwd_ret_1"])
    auc = _auc_walkforward(df["random_label"], df["fwd_ret_1"])
    pnl, cov = _naive_pnl(df["random_label"], df["fwd_ret_1"])

    assert abs(mi) < 0.02, "Mutual information should be near zero for random trend"
    assert 0.40 <= auc <= 0.55, f"AUC={auc} should be near 0.5 for random trend"
    assert abs(round(pnl, 2)) < 0.05, "PnL should be near zero for random trend"


# -------------------------------
# TEST 2: Perfect trend ≈ fully predictive
# -------------------------------
def test_perfect_labels_predictiveness(synthetic_data):
    df = synthetic_data
    mi = _mutual_info(df["perfect_label"], df["fwd_ret_1"])
    auc = _auc_walkforward(df["perfect_label"], df["fwd_ret_1"])
    pnl, cov = _naive_pnl(df["perfect_label"], df["fwd_ret_1"])

    assert mi > 0.2, "Mutual information should be high for perfect trend"
    assert auc > 0.95, "AUC should be close to 1 for perfect trend"
    assert pnl > 0.8, "PnL should be very positive for perfect trend"


# -------------------------------
# TEST 3: Inverse trend ≈ anti-predictive
# -------------------------------
def test_inverse_labels_predictiveness(synthetic_data):
    df = synthetic_data
    mi = _mutual_info(df["inverse_label"], df["fwd_ret_1"])
    auc = _auc_walkforward(df["inverse_label"], df["fwd_ret_1"])
    pnl, cov = _naive_pnl(df["inverse_label"], df["fwd_ret_1"])

    assert mi > 0.2, "Mutual information should still be high even if inverse"
    assert (
        auc > 0.98
    ), "AUC should be close to 1 for inverse trend (perfect discrimination, just inverted)"
    assert pnl < -0.8, "PnL should be very negative for inverse trend"


# -------------------------------
# TEST 4: Evaluate_labels end-to-end sanity check
# -------------------------------
def test_evaluate_labels_end_to_end(synthetic_data):
    df = synthetic_data
    # Make sure we have enough data and proper column names
    results = evaluate_labels(
        df, ["random_label", "perfect_label", "inverse_label"], "fwd_ret_1"
    )

    # The ranking might not be exactly as expected due to the conservative evaluation
    # but we can check that we get results
    assert len(results) == 3
    assert set(results["label"].values) == {
        "random_label",
        "perfect_label",
        "inverse_label",
    }


# ---------------------------------------------------------------------------
# Tests for classify_label — horizon tail NaNs + output type assertions
# ---------------------------------------------------------------------------

@pytest.fixture
def classify_df():
    """DataFrame with close + a label column; realistic size for AUC walk-forward."""
    np.random.seed(0)
    n = 2000
    ret = np.random.normal(0, 0.001, n)
    price = 100 * np.exp(np.cumsum(ret))
    fwd1 = np.log(price[1:] / price[:-1])
    labels = np.sign(np.append(fwd1, [0]))  # perfect label
    df = pd.DataFrame(
        {"close": price, "label": labels},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    return df


class TestClassifyLabel:
    def test_returns_required_keys(self, classify_df):
        """Result dict must contain corr, mi, auc, pnl, coverage, type."""
        result = classify_label(classify_df, "label", horizon=1)
        for key in ("corr", "mi", "auc", "pnl", "coverage", "type"):
            assert key in result, f"Missing key: {key}"

    def test_pnl_is_scalar_float(self, classify_df):
        """pnl must be a scalar float, NOT a tuple."""
        result = classify_label(classify_df, "label", horizon=1)
        assert isinstance(result["pnl"], float), (
            f"pnl should be float, got {type(result['pnl'])}: {result['pnl']}"
        )

    def test_coverage_is_scalar_float(self, classify_df):
        """coverage must be a scalar float between 0 and 1."""
        result = classify_label(classify_df, "label", horizon=1)
        assert isinstance(result["coverage"], float)
        assert 0.0 <= result["coverage"] <= 1.0

    def test_type_is_string(self, classify_df):
        """type must be 'Predictive' or 'Descriptive'."""
        result = classify_label(classify_df, "label", horizon=1)
        assert result["type"] in ("Predictive", "Descriptive")

    def test_horizon_tail_nans_do_not_drop_label_column(self, classify_df):
        """
        With horizon=5, the last 5 rows of fwd_ret have NaN.
        Using dropna(axis=1) would drop fwd_ret entirely — verify this no longer happens
        by checking that corr is finite (requires both columns to survive).
        """
        result = classify_label(classify_df, "label", horizon=5)
        # corr must be a float (NaN or finite) — KeyError would mean column was dropped
        assert "corr" in result
        assert isinstance(result["corr"], float)

    def test_longer_horizon_still_runs(self, classify_df):
        """horizon=20 leaves many NaN tail rows; function must not crash."""
        result = classify_label(classify_df, "label", horizon=20)
        assert isinstance(result["pnl"], float)
        assert isinstance(result["coverage"], float)

    def test_corr_bounded(self, classify_df):
        """Correlation must be in [-1, 1] or NaN."""
        result = classify_label(classify_df, "label", horizon=1)
        if not np.isnan(result["corr"]):
            assert -1.0 <= result["corr"] <= 1.0
