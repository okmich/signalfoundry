import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.posterior_inference.asymmetry import (
    AxisProbe,
    MarketAxis,
    PosteriorStream,
    ValidationVerdict,
    incremental_residual,
    validate_outcomes,
    validate_stream,
)


def _one_hot_stream(states: np.ndarray, T: int, K: int = 2, fold_ids: np.ndarray | None = None) -> PosteriorStream:
    probs = np.zeros((T, K))
    probs[np.arange(T), states] = 1.0
    idx = pd.date_range("2020-01-01", periods=T, freq="5min")
    return PosteriorStream(probs=probs, state_names=[f"s{k}" for k in range(K)], index=idx, fold_ids=fold_ids)


# --- incremental_residual -----------------------------------------------------

def test_incremental_residual_removes_linear_baseline() -> None:
    rng = np.random.default_rng(3)
    n = 1000
    b = rng.standard_normal(n)
    f = 3.0 * b + 5.0 + 0.01 * rng.standard_normal(n)
    resid = incremental_residual(f, b)
    assert abs(np.corrcoef(resid, b)[0, 1]) < 0.05   # baseline component removed
    assert abs(np.nanmean(resid)) < 0.05


def test_incremental_residual_preserves_nan() -> None:
    f = np.array([1.0, np.nan, 3.0, 4.0])
    b = np.array([1.0, 2.0, np.nan, 4.0])
    resid = incremental_residual(f, b)
    assert np.isnan(resid[1]) and np.isnan(resid[2])
    assert np.isfinite(resid[0]) and np.isfinite(resid[3])


# --- PosteriorStream ----------------------------------------------------------

def test_posterior_stream_rejects_bad_state_names() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="5min")
    with pytest.raises(ValueError, match="state_names"):
        PosteriorStream(probs=np.full((3, 2), 0.5), state_names=["only_one"], index=idx)


# --- validate_outcomes: the lesson encoded ------------------------------------

def test_rejects_baseline_only_separation() -> None:
    # forward = baseline signal + independent noise; the state tracks the baseline, so it separates raw forward (the trap),
    # but the residual is state-independent noise -> no incremental edge. Folds backstop any single pooled fluke.
    rng = np.random.default_rng(0)
    T = 4000
    b = rng.standard_normal(T)
    states = (b > np.median(b)).astype(int)         # state correlated with the baseline
    forward = 2.0 * b + rng.standard_normal(T)      # residual is genuine noise, independent of the state
    fold_ids = np.arange(T) // 1000                 # 4 folds for the stability backstop
    stream = _one_hot_stream(states, T, fold_ids=fold_ids)
    report = validate_outcomes(stream, [AxisProbe("vol", 1, True, forward, b)], min_coverage=100.0)
    assert report.verdicts["vol"] == ValidationVerdict.REJECTED
    raw = report.table[(report.table.kind == "raw") & (report.table.market_axis == "vol")]
    assert raw.t_hac.abs().max() > 3.0              # raw separation is real — and still rejected


def test_confirms_incremental_separation_and_fold_stable() -> None:
    rng = np.random.default_rng(1)
    T = 4000
    b = rng.standard_normal(T)
    states = rng.integers(0, 2, T)             # state INDEPENDENT of the baseline
    forward = 2.0 * b + 0.5 * states + 1e-3 * rng.standard_normal(T)
    fold_ids = np.arange(T) // 1000            # 4 folds
    stream = _one_hot_stream(states, T, fold_ids=fold_ids)
    report = validate_outcomes(stream, [AxisProbe("vol", 1, True, forward, b)], min_coverage=100.0)
    assert report.verdicts["vol"] == ValidationVerdict.CONFIRMED
    assert not report.per_fold.empty
    assert (report.per_fold.t_hac.abs() > 2.0).mean() >= 0.6


def test_inconclusive_when_coverage_too_low() -> None:
    rng = np.random.default_rng(2)
    T = 500
    b = rng.standard_normal(T)
    states = rng.integers(0, 2, T)
    forward = 2.0 * b + 0.5 * states
    stream = _one_hot_stream(states, T)
    report = validate_outcomes(stream, [AxisProbe("vol", 1, True, forward, b)], min_coverage=1e9)
    assert report.verdicts["vol"] == ValidationVerdict.INCONCLUSIVE


def test_inconclusive_when_incremental_coverage_thin_but_raw_is_fine() -> None:
    # Raw forward is well-covered and separated; the baseline is mostly NaN, so the residual (hence the incremental
    # contrast) is thin -> "can't tell", not "rejected".
    rng = np.random.default_rng(5)
    T = 1000
    states = rng.integers(0, 2, T)
    forward = 0.5 * states + 0.01 * rng.standard_normal(T)   # raw separation present, full coverage
    baseline = np.full(T, np.nan)
    baseline[-80:] = rng.standard_normal(80)                 # baseline finite on only 80 rows -> thin residual
    stream = _one_hot_stream(states, T)
    report = validate_outcomes(stream, [AxisProbe("vol", 1, True, forward, baseline)], min_coverage=200.0)
    assert report.verdicts["vol"] == ValidationVerdict.INCONCLUSIVE


def test_fold_ids_accepts_python_list() -> None:
    # A list (not ndarray) of fold ids must slice correctly, not collapse to a scalar False.
    rng = np.random.default_rng(6)
    T = 4000
    b = rng.standard_normal(T)
    states = rng.integers(0, 2, T)
    forward = 2.0 * b + 0.5 * states + 1e-3 * rng.standard_normal(T)
    fold_ids = list(np.arange(T) // 1000)                    # plain Python list
    stream = _one_hot_stream(states, T, fold_ids=fold_ids)
    report = validate_outcomes(stream, [AxisProbe("vol", 1, True, forward, b)], min_coverage=100.0)
    assert report.verdicts["vol"] == ValidationVerdict.CONFIRMED
    assert sorted(report.per_fold.fold.unique().tolist()) == [0, 1, 2, 3]


# --- validate_stream wiring ---------------------------------------------------

def test_validate_stream_builds_probes_and_judges() -> None:
    rng = np.random.default_rng(4)
    T = 2000
    idx = pd.date_range("2020-01-01", periods=T, freq="5min")
    close = 100.0 * np.exp(np.cumsum(0.001 * rng.standard_normal(T)))
    prices = pd.DataFrame({"close": close, "tick_volume": rng.integers(1, 100, T)}, index=idx)
    states = rng.integers(0, 3, T)
    probs = np.zeros((T, 3))
    probs[np.arange(T), states] = 1.0
    stream = PosteriorStream(probs=probs, state_names=["low", "mid", "high"], index=idx)
    report = validate_stream(stream, prices, axes=[MarketAxis.VOLATILITY], horizons=[2, 4], min_coverage=50.0)
    assert "volatility" in report.verdicts
    assert set(report.table.kind) == {"raw", "incremental"}
