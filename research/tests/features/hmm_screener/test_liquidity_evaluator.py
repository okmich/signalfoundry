"""Tests for the liquidity-axis evaluator.

The evaluator computes forward summed volume over ``horizons[0]`` bars per row,
then ranks states by median forward volume via ``map_regime_to_volatility_score``.
Polarity: bucket 0 = lowest forward volume = LEAST liquid; bucket k = MOST
liquid. (The earlier Amihud-style metric was dropped to remove explicit
volatility entanglement; forward median volume is the cleaner proxy at 5-min
OHLCV resolution.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from okmich_quant_research.features.hmm_screener import (
    AXIS_EVALUATORS,
    AxisEvaluation,
    HmmFeatureScreener,
    HmmScreenerConfig,
    ParetoStatus,
    ScreenStrategy,
    evaluate_liquidity,
)


def _make_synthetic_liquidity_ohlcv(T: int = 2000, seed: int = 11) -> pd.DataFrame:
    """Two regimes: 'liquid' (low |ret|, high volume) and 'illiquid' (high |ret|, low volume).

    The two halves are concatenated so a 2-state HMM has the structural opportunity
    to discover the regime split when handed the right feature set.
    """
    rng = np.random.default_rng(seed)
    half = T // 2
    # Liquid half — small price moves, large volumes.
    rets_liquid = rng.normal(0.0, 0.0003, half)
    vol_liquid = rng.uniform(800.0, 1200.0, half)
    # Illiquid half — large price moves, small volumes.
    rets_illiquid = rng.normal(0.0, 0.003, half)
    vol_illiquid = rng.uniform(50.0, 150.0, half)
    log_rets = np.concatenate([rets_liquid, rets_illiquid])
    volume = np.concatenate([vol_liquid, vol_illiquid])
    close = 100.0 * np.exp(np.cumsum(log_rets))
    high = close * (1.0 + rng.uniform(0.0, 0.001, T))
    low = close * (1.0 - rng.uniform(0.0, 0.001, T))
    return pd.DataFrame({"close": close, "high": high, "low": low, "tick_volume": volume})


def test_liquidity_in_axis_evaluators_dispatch() -> None:
    assert "liquidity" in AXIS_EVALUATORS
    assert AXIS_EVALUATORS["liquidity"] is evaluate_liquidity


def test_evaluate_liquidity_returns_error_when_volume_column_missing() -> None:
    T = 200
    close = np.full(T, 100.0)
    raw = pd.DataFrame({"close": close, "high": close * 1.001, "low": close * 0.999})
    state_labels = np.zeros(T, dtype=np.int64)
    state_labels[T // 2:] = 1
    out = evaluate_liquidity(gamma=np.zeros((T, 2)), state_labels=state_labels,
                             raw_data=raw, horizons=(12,))
    assert isinstance(out, AxisEvaluation)
    assert out.axis_separation == 0.0
    assert "volume" in out.raw_details.get("error", "")


def test_evaluate_liquidity_accepts_either_tick_volume_or_volume() -> None:
    raw_tv = _make_synthetic_liquidity_ohlcv(T=400)
    raw_v = raw_tv.rename(columns={"tick_volume": "volume"})
    state_labels = np.concatenate([np.zeros(200, dtype=np.int64), np.ones(200, dtype=np.int64)])

    out_tv = evaluate_liquidity(gamma=np.zeros((400, 2)), state_labels=state_labels,
                                raw_data=raw_tv, horizons=(12,))
    out_v = evaluate_liquidity(gamma=np.zeros((400, 2)), state_labels=state_labels,
                               raw_data=raw_v, horizons=(12,))
    assert out_tv.raw_details["volume_col"] == "tick_volume"
    assert out_v.raw_details["volume_col"] == "volume"
    # The illiquidity numbers should match between the two — only the column name changes.
    assert out_tv.axis_separation == pytest.approx(out_v.axis_separation, rel=1e-12)


def test_evaluate_liquidity_finds_separation_on_constructed_two_regime_data() -> None:
    """Hand-crafted two-state Viterbi over a deliberately bifurcated liquidity signal.

    Liquid half (state 0): small |returns| + high volume -> low illiquidity.
    Illiquid half (state 1): big |returns| + low volume   -> high illiquidity.
    The evaluator should report a clear axis_separation and two distinct buckets.
    """
    T = 2000
    raw = _make_synthetic_liquidity_ohlcv(T=T)
    # Perfect state labels matching the regime split.
    state_labels = np.concatenate([np.zeros(T // 2, dtype=np.int64), np.ones(T // 2, dtype=np.int64)])
    out = evaluate_liquidity(gamma=np.zeros((T, 2)), state_labels=state_labels,
                             raw_data=raw, horizons=(12,))
    assert out.axis_separation > 0.0
    assert out.secondary_robustness >= 2
    assert out.secondary_label == "n_distinct_buckets"
    # State 0 (liquid: high volume) should map to a HIGHER bucket than state 1 (illiquid: low volume),
    # because the ascending-by-median ranking puts highest forward volume in the highest bucket.
    # See raw_details['polarity']: bucket 0 = LEAST liquid, bucket k = MOST liquid.
    mapping = out.raw_details["mapping"]
    assert mapping[0] > mapping[1]


@pytest.mark.slow
def test_screener_end_to_end_liquidity_axis() -> None:
    """Smoke test: the screener can drive a fit + liquidity evaluation end-to-end."""
    raw = _make_synthetic_liquidity_ohlcv(T=2000)

    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        log_rets = np.log(out["close"] / out["close"].shift(1))
        out["abs_log_rets"] = log_rets.abs()
        out["vol_log"] = np.log(out["tick_volume"].clip(lower=1.0))
        return out

    config = HmmScreenerConfig(
        signal_type="liquidity",
        algo="hmm_lambda",
        n_states=2,
        data_size=2000,
        random_state=42,
    )
    screener = HmmFeatureScreener(config, raw, feature_engineering)
    result = screener.screen(["abs_log_rets", "vol_log"], strategy=ScreenStrategy.ABLATION)

    assert len(result.evaluations) >= 1
    assert all(e.pareto_status in {ParetoStatus.KEEPER, ParetoStatus.TRAP,
                                   ParetoStatus.FRAGILE, ParetoStatus.DOMINATED}
               for e in result.evaluations)
    non_error = [e for e in result.evaluations if e.error is None]
    assert len(non_error) >= 1
