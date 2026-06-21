"""Streaming-Batch Equivalence Test for CTL.

The streaming FSM (`CTLState` + `ctl_step`, exercised here via `ctl_streaming_replay`) must produce labels
**bar-for-bar identical** to the offline `continuous_trend_labeling` over the same input series with the same
`omega`.

CTL is deterministic — there is no source of stochasticity, so equivalence is **exact**, not approximate. Any
divergence is a bug. The batch function emits NaN during pre-trigger warmup while the streaming FSM emits 0;
NaN -> 0 mapping preserves the integer-equivalence contract this test enforces.

Two layers of test:

1. Synthetic series (cheap, fast, runs as part of pytest collection): random walks, monotone trends,
   oscillating series, edge omegas. Catches FSM transcription bugs without needing a data folder.

2. Panel series (heavier; auto-skipped if the data folder is absent): each panel symbol at three representative
   omegas, on the 5m series tail. Point at the parquet folder via the CTL_PANEL_DATA_DIR env var.

Run with: pytest features/tests/trend/test_ctl_streaming_equivalence.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.trend import (CTLState, continuous_trend_labeling, ctl_step, ctl_streaming_replay,
                                          ctl_warm_up)


# ----------------------------------------------------------------------------
# Synthetic series
# ----------------------------------------------------------------------------

def _assert_exact_equivalence(prices: np.ndarray, omega: float, label: str = "") -> None:
    # CTL emits NaN during pre-trigger warmup; the streaming FSM emits 0. Map NaN -> 0 to
    # preserve the integer-equivalence contract this test enforces.
    batch_raw = np.asarray(continuous_trend_labeling(prices, omega=omega), dtype=np.float64)
    batch = np.where(np.isnan(batch_raw), 0, batch_raw).astype(np.int64)
    streaming = ctl_streaming_replay(prices, omega=omega)
    diffs = np.flatnonzero(batch != streaming)
    if len(diffs) > 0:
        first = diffs[0]
        sample = []
        for i in diffs[:5]:
            sample.append(f"i={int(i)}: batch={int(batch[i])} stream={int(streaming[i])} price={prices[i]:.6f}")
        raise AssertionError(
            f"[{label}] streaming != batch at {len(diffs)} bar(s); first divergence i={int(first)}\n"
            + "\n".join(sample)
        )


def _random_walk(n: int, seed: int, drift: float = 0.0, vol: float = 0.001, start: float = 100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=drift, scale=vol, size=n)
    return start * np.exp(np.cumsum(log_returns))


@pytest.mark.parametrize("n,seed,vol,omega", [
    (1_000, 1, 0.001, 0.005),
    (1_000, 2, 0.002, 0.01),
    (5_000, 3, 0.001, 0.02),
    (5_000, 4, 0.003, 0.05),
    (10_000, 5, 0.0005, 0.001),
    (10_000, 6, 0.005, 0.10),
])
def test_random_walk_equivalence(n: int, seed: int, vol: float, omega: float) -> None:
    prices = _random_walk(n=n, seed=seed, vol=vol)
    _assert_exact_equivalence(prices, omega, label=f"rw n={n} seed={seed} vol={vol} omega={omega}")


def test_monotone_uptrend() -> None:
    prices = np.linspace(100.0, 200.0, num=2_000)
    _assert_exact_equivalence(prices, omega=0.01, label="monotone-up")


def test_monotone_downtrend() -> None:
    prices = np.linspace(200.0, 100.0, num=2_000)
    _assert_exact_equivalence(prices, omega=0.01, label="monotone-down")


def test_flat_series_no_trigger() -> None:
    # All zero label expected; FSM and batch must agree on the all-zero output.
    prices = np.full(500, 100.0)
    _assert_exact_equivalence(prices, omega=0.01, label="flat")


def test_oscillating_series() -> None:
    # Sawtooth that crosses omega both ways repeatedly — exercises flip guards.
    t = np.arange(2_000)
    prices = 100.0 + 5.0 * np.sin(t * 2 * np.pi / 50.0)
    _assert_exact_equivalence(prices, omega=0.02, label="oscillating")


def test_extreme_small_omega() -> None:
    prices = _random_walk(n=2_000, seed=42, vol=0.001)
    _assert_exact_equivalence(prices, omega=0.0001, label="tiny-omega")


def test_extreme_large_omega() -> None:
    prices = _random_walk(n=2_000, seed=43, vol=0.001)
    # Most series will never cross 50% -> all-zero labels expected, both sides.
    _assert_exact_equivalence(prices, omega=0.50, label="huge-omega")


def test_pandas_series_input_parity() -> None:
    """Batch accepts pd.Series; streaming accepts either. Both should agree."""
    arr = _random_walk(n=3_000, seed=7, vol=0.002)
    series = pd.Series(arr, index=pd.date_range("2024-01-01", periods=len(arr), freq="5min"))
    batch_raw = np.asarray(continuous_trend_labeling(series, omega=0.015), dtype=np.float64)
    batch_from_series = np.where(np.isnan(batch_raw), 0, batch_raw).astype(np.int64)
    streaming_from_array = ctl_streaming_replay(arr, omega=0.015)
    np.testing.assert_array_equal(batch_from_series, streaming_from_array)


# ----------------------------------------------------------------------------
# Input validation / live robustness
# ----------------------------------------------------------------------------

def test_ctlstate_rejects_nonfinite_omega():
    for bad in (0.0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="omega must be > 0"):
            CTLState(omega=bad)


@pytest.mark.parametrize("fn", [ctl_streaming_replay, ctl_warm_up])
def test_array_entrypoints_reject_nonfinite_prices(fn):
    """Array entry points mirror the batch function: clean warm-up data is required."""
    with pytest.raises(ValueError, match="NaN or infinite"):
        fn(np.array([100.0, np.nan, 110.0]), omega=0.05)
    with pytest.raises(ValueError, match="NaN or infinite"):
        fn(np.array([100.0, np.inf, 110.0]), omega=0.05)


def test_ctl_step_holds_label_on_nonfinite_tick():
    """A single bad/missing live tick must not update state or force a spurious flat."""
    # Drive into a confirmed uptrend.
    state = CTLState(omega=0.05)
    for i, p in enumerate([100.0, 101.0, 110.0]):  # bar 2 triggers up
        label = ctl_step(state, p, i)
    assert label == 1
    snapshot = (state.direction, state.x_high, state.t_high, state.x_low, state.t_low)

    # NaN/inf ticks: state frozen, current direction held.
    assert ctl_step(state, float("nan"), 3) == 1
    assert ctl_step(state, float("inf"), 4) == 1
    assert (state.direction, state.x_high, state.t_high, state.x_low, state.t_low) == snapshot

    # A valid bar after the gap resumes normally (new high extends the trend).
    assert ctl_step(state, 115.0, 5) == 1
    assert state.x_high == 115.0 and state.t_high == 5


# ----------------------------------------------------------------------------
# Panel series (5m parquet tail) — heavier, optional
# ----------------------------------------------------------------------------

PANEL_DATA_FOLDER = os.environ.get("CTL_PANEL_DATA_DIR", r"D:\data_dump\market_data\labelled\FXPIG-Server\5")
PANEL_SYMBOLS = [
    "EURUSD.r", "USDJPY.r", "GBPUSD.r", "AUDUSD.r", "EURJPY.r",  # FX (5)
    "US500.r", "USTEC.r", "DE30.r",                              # Indices (3)
    "BTCUSDT.r",                                                 # Crypto (1)
    "XAUUSD.r",                                                  # Metals (1)
]
PANEL_OMEGAS = [0.001, 0.01, 0.05]
PANEL_TAIL_ROWS = 600_000  # ~5y of 5m bars


def _panel_data_available() -> bool:
    return os.path.isdir(PANEL_DATA_FOLDER) and any(
        os.path.exists(os.path.join(PANEL_DATA_FOLDER, f"{s}.parquet")) for s in PANEL_SYMBOLS
    )


@pytest.mark.skipif(not _panel_data_available(), reason="CTL panel data folder not present (set CTL_PANEL_DATA_DIR)")
@pytest.mark.parametrize("symbol", PANEL_SYMBOLS)
@pytest.mark.parametrize("omega", PANEL_OMEGAS)
def test_panel_symbol_equivalence(symbol: str, omega: float) -> None:
    filepath = os.path.join(PANEL_DATA_FOLDER, f"{symbol}.parquet")
    if not os.path.exists(filepath):
        pytest.skip(f"missing parquet: {filepath}")
    df = pd.read_parquet(filepath, columns=["close"]).iloc[-PANEL_TAIL_ROWS:]
    prices = df["close"].to_numpy()
    _assert_exact_equivalence(prices, omega=omega, label=f"{symbol} omega={omega}")
