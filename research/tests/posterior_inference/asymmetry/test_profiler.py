import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.posterior_inference.asymmetry import (
    ForwardOutcome,
    bartlett_hac_variance,
    forward_outcome_by_state,
)
from okmich_quant_research.posterior_inference.asymmetry.profiler import _weighted_quantiles


def _two_state_split_probs(T: int, p_hi: float = 0.9) -> np.ndarray:
    """State 0 dominates the second half, state 1 the first half."""
    p = np.empty((T, 2), dtype=float)
    half = T // 2
    p[:half, 0] = 1.0 - p_hi
    p[half:, 0] = p_hi
    p[:, 1] = 1.0 - p[:, 0]
    return p


def _uniform_probs(T: int, K: int = 3) -> np.ndarray:
    return np.full((T, K), 1.0 / K, dtype=float)


# --- bartlett_hac_variance ----------------------------------------------------

def test_hac_bandwidth_zero_is_sum_of_squares() -> None:
    x = np.array([0.5, -1.0, 2.0, -0.25])
    assert bartlett_hac_variance(x, bandwidth=0) == pytest.approx(float(x @ x))


def test_hac_positive_autocorrelation_inflates_variance() -> None:
    x = np.ones(50)  # perfectly positively autocorrelated
    var0 = bartlett_hac_variance(x, bandwidth=0)
    var10 = bartlett_hac_variance(x, bandwidth=10)
    assert var10 > var0


def test_hac_is_nonnegative_and_handles_empty() -> None:
    alt = np.array([1.0, -1.0] * 20)  # strong negative autocorrelation
    assert bartlett_hac_variance(alt, bandwidth=5) >= 0.0
    assert np.isnan(bartlett_hac_variance(np.array([]), bandwidth=3))


# --- _weighted_quantiles ------------------------------------------------------

def test_weighted_quantiles_uniform_weights_match_ordering() -> None:
    values = np.array([10.0, 0.0, 5.0, 20.0])
    weights = np.ones(4)
    q25, q50, q75 = _weighted_quantiles(values, weights, np.array([0.25, 0.5, 0.75]))
    assert q25 <= q50 <= q75
    assert 0.0 <= q25 and q75 <= 20.0


def test_weighted_quantiles_zero_total_weight_is_nan() -> None:
    out = _weighted_quantiles(np.array([1.0, 2.0]), np.zeros(2), np.array([0.5]))
    assert np.isnan(out).all()


def test_weighted_quantiles_ignore_zero_weight_observations() -> None:
    # The state owns no mass at value 0; every quantile must be 100, not an interpolated value.
    out = _weighted_quantiles(np.array([0.0, 100.0]), np.array([0.0, 1.0]), np.array([0.25, 0.5, 0.75]))
    np.testing.assert_array_equal(out, np.array([100.0, 100.0, 100.0]))


def test_weighted_quantiles_negligible_weight_does_not_anchor() -> None:
    # Near-zero mass at value 0 must not drag the quantile down to ~50.
    out = _weighted_quantiles(np.array([0.0, 100.0]), np.array([1e-12, 1.0]), np.array([0.25, 0.5, 0.75]))
    np.testing.assert_array_equal(out, np.array([100.0, 100.0, 100.0]))


# --- forward_outcome_by_state: validation -------------------------------------

def test_rejects_non_matrix_posterior() -> None:
    with pytest.raises(ValueError):
        forward_outcome_by_state(np.ones(10), {"trend": ForwardOutcome(np.zeros(10), 1)})


def test_rejects_empty_outcomes() -> None:
    with pytest.raises(ValueError, match="at least one axis"):
        forward_outcome_by_state(_uniform_probs(10), {})


def test_rejects_outcome_length_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        forward_outcome_by_state(_uniform_probs(10), {"trend": ForwardOutcome(np.zeros(9), 1)})


def test_rejects_unnormalized_rows() -> None:
    bad = np.full((10, 3), 0.5)  # rows sum to 1.5 — logits/scores, not a posterior
    with pytest.raises(ValueError, match="sum to 1"):
        forward_outcome_by_state(bad, {"trend": ForwardOutcome(np.zeros(10), 1)})


def test_accepts_rows_within_tolerance() -> None:
    probs = _uniform_probs(20, K=3) + 1e-9  # tiny drift, within default row_sum_tol
    df = forward_outcome_by_state(probs, {"trend": ForwardOutcome(np.linspace(-1, 1, 20), 2)}, min_coverage=1.0)
    assert len(df) == 3


def test_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError, match="horizon"):
        forward_outcome_by_state(_uniform_probs(10), {"trend": ForwardOutcome(np.zeros(10), 0)})


def test_rejects_state_names_length_mismatch() -> None:
    with pytest.raises(ValueError, match="state_names"):
        forward_outcome_by_state(_uniform_probs(10, K=3), {"trend": ForwardOutcome(np.zeros(10), 1)},
                                 state_names=["a", "b"])


# --- forward_outcome_by_state: structure --------------------------------------

def test_one_row_per_axis_state_with_expected_columns() -> None:
    T = 40
    probs = _uniform_probs(T, K=3)
    outcomes = {"trend": ForwardOutcome(np.linspace(-1, 1, T), 4), "vol": ForwardOutcome(np.abs(np.linspace(-1, 1, T)), 4)}
    df = forward_outcome_by_state(probs, outcomes, state_names=["bear", "neutral", "bull"], min_coverage=1.0)
    assert len(df) == 2 * 3
    assert set(df["axis"]) == {"trend", "vol"}
    assert list(df.columns) == ["axis", "state", "state_label", "horizon", "n_valid", "n_map", "coverage", "n_eff",
                                "w_mean", "q25", "q50", "q75", "pooled_mean", "delta_vs_pooled", "se_hac", "t_hac",
                                "low_coverage"]
    assert set(df["state_label"]) == {"bear", "neutral", "bull"}


# --- forward_outcome_by_state: numerics ---------------------------------------

def test_delta_equals_w_mean_minus_pooled() -> None:
    rng = np.random.default_rng(0)
    T = 200
    raw = rng.random((T, 3))
    probs = raw / raw.sum(axis=1, keepdims=True)
    outcomes = {"trend": ForwardOutcome(rng.standard_normal(T), 5)}
    df = forward_outcome_by_state(probs, outcomes, min_coverage=1.0)
    np.testing.assert_allclose(df["delta_vs_pooled"], df["w_mean"] - df["pooled_mean"], rtol=1e-9, atol=1e-12)


def test_uniform_posterior_gives_zero_contrast() -> None:
    T = 100
    probs = _uniform_probs(T, K=3)
    outcomes = {"trend": ForwardOutcome(np.linspace(0, 10, T), 3)}
    df = forward_outcome_by_state(probs, outcomes, min_coverage=1.0)
    # Every state has identical (uniform) weights -> conditional mean equals pooled mean.
    np.testing.assert_allclose(df["delta_vs_pooled"], 0.0, atol=1e-9)


def test_separating_posterior_recovers_sign_of_contrast() -> None:
    T = 200
    probs = _two_state_split_probs(T, p_hi=0.9)
    outcome = np.arange(T, dtype=float)  # low early, high late
    df = forward_outcome_by_state(probs, {"trend": ForwardOutcome(outcome, 1)}, min_coverage=1.0)
    s0 = df[df["state"] == 0].iloc[0]
    s1 = df[df["state"] == 1].iloc[0]
    # State 0 concentrates mass on the high-outcome second half -> positive contrast; state 1 the opposite.
    assert s0["delta_vs_pooled"] > 0.0
    assert s1["delta_vs_pooled"] < 0.0
    assert s0["pooled_mean"] == pytest.approx(outcome.mean())


def test_nan_tail_is_excluded_from_valid_count() -> None:
    T, horizon = 50, 6
    probs = _uniform_probs(T, K=2)
    values = np.arange(T, dtype=float)
    values[-horizon:] = np.nan  # forward window unavailable at the tail
    df = forward_outcome_by_state(probs, {"trend": ForwardOutcome(values, horizon)}, min_coverage=1.0)
    assert (df["n_valid"] == T - horizon).all()


def test_low_coverage_flags_and_nans_significance() -> None:
    T = 100
    probs = _uniform_probs(T, K=2)  # coverage per state ~ T/2 = 50
    outcomes = {"trend": ForwardOutcome(np.linspace(-1, 1, T), 2)}
    df = forward_outcome_by_state(probs, outcomes, min_coverage=1000.0)  # force low coverage
    assert df["low_coverage"].all()
    assert df["se_hac"].isna().all()
    assert df["t_hac"].isna().all()
    # point estimates are still reported
    assert df["w_mean"].notna().all()


def test_hac_bandwidth_ties_to_horizon_minus_one() -> None:
    # With horizon=1 (bandwidth 0), se_hac must equal the white-noise SE of the contribution series.
    rng = np.random.default_rng(7)
    T = 300
    probs = _two_state_split_probs(T, p_hi=0.8)
    outcome = rng.standard_normal(T)
    df = forward_outcome_by_state(probs, {"trend": ForwardOutcome(outcome, 1)}, min_coverage=1.0)
    finite = np.isfinite(outcome)
    pooled = outcome[finite].mean()
    n_valid = finite.sum()
    for k in (0, 1):
        w = probs[finite, k]
        cov = w.sum()
        contrib = (w / cov - 1.0 / n_valid) * (outcome[finite] - pooled)
        expected_se = np.sqrt(bartlett_hac_variance(contrib, bandwidth=0))
        got = df[df["state"] == k].iloc[0]["se_hac"]
        assert got == pytest.approx(expected_se, rel=1e-9)
