"""Per-signal-type axis evaluators.

Each ``evaluate_*`` function takes a fitted HMM's outputs plus the raw OHLC data and returns an ``AxisEvaluation`` carrying:
  * ``axis_separation`` — the primary axis-matched spread across states (forward-return spread for direction/momentum,
                          forward-vol spread for volatility, choppiness-index spread for path structure).
  * ``secondary_robustness`` — an axis-appropriate count / monotonicity stat.

Wraps the existing ``okmich_quant_labelling.utils.label_util`` mapper functions with ``return_diagnostics=True`` so the
heavy lifting of statistical scoring lives in one place.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from okmich_quant_labelling.utils.label_util import (
    map_label_to_momentum_score,
    map_label_to_trend_direction,
    map_regime_to_path_structure_score,
    map_regime_to_volatility_score,
)

from ._result import AxisEvaluation


AxisEvaluator = Callable[..., AxisEvaluation]


def _build_forward_log_returns(close: NDArray, state_labels: NDArray, horizon: int) -> pd.DataFrame:
    """Build a per-row DataFrame with state + forward log-return at ``horizon``."""
    log_close = np.log(close)
    fwd = np.full(len(close), np.nan)
    if len(close) > horizon:
        fwd[:-horizon] = log_close[horizon:] - log_close[:-horizon]
    return pd.DataFrame({"state": state_labels[:len(close)], "returns": fwd}).dropna()


def evaluate_direction(*, gamma: NDArray, state_labels: NDArray, raw_data: pd.DataFrame,
                       horizons: tuple[int, ...], **_ignored) -> AxisEvaluation:
    """Direction axis: forward log-return mean spread + count of conservatively-significant states."""
    primary_horizon = horizons[0] if horizons else 12
    fwd_df = _build_forward_log_returns(raw_data["close"].values, state_labels, primary_horizon)
    if len(fwd_df) < 30:
        return AxisEvaluation(0.0, 0.0, "n_significant_states",
                              raw_details={"error": "fewer than 30 valid forward returns"})

    mapping, diag = map_label_to_trend_direction(fwd_df, state_col="state", return_col="returns",
        method="conservative", return_diagnostics=True)
    n_significant = sum(1 for v in mapping.values() if v != 0)

    valid = diag.dropna(subset=["mean"]) if "mean" in diag.columns else diag.iloc[0:0]
    sep = float(valid["mean"].max() - valid["mean"].min()) if len(valid) >= 2 else 0.0
    return AxisEvaluation(
        axis_separation=sep,
        secondary_robustness=float(n_significant),
        secondary_label="n_significant_states",
        raw_details={"horizon": primary_horizon, "mapping": {int(k): int(v) for k, v in mapping.items()}},
    )


def evaluate_momentum(*, gamma: NDArray, state_labels: NDArray, raw_data: pd.DataFrame,
                     horizons: tuple[int, ...], is_directional: bool = True, **_ignored) -> AxisEvaluation:
    """Momentum axis: forward log-return rank spread + count of distinct momentum scores.

    ``is_directional`` selects between signed momentum (rank by signed median) and magnitude-only momentum
    (rank by ``|median|``). The screener decides by inspecting the baseline features' registry-declared ``directional`` flag.
    """
    primary_horizon = horizons[0] if horizons else 12
    fwd_df = _build_forward_log_returns(raw_data["close"].values, state_labels, primary_horizon)
    if len(fwd_df) < 30:
        return AxisEvaluation(0.0, 0.0, "n_distinct_scores",
                              raw_details={"error": "fewer than 30 valid forward returns"})

    fwd_df = fwd_df.rename(columns={"state": "regime"})
    mapping, diag = map_label_to_momentum_score(
        fwd_df, regime_col="regime", ret_col="returns",
        method="robust", is_directional=is_directional, return_diagnostics=True,
    )

    if "insufficient_data" in diag.columns:
        valid = diag[~diag["insufficient_data"]]
    else:
        valid = diag.dropna(subset=["median"]) if "median" in diag.columns else diag.iloc[0:0]

    if len(valid) >= 2 and "median" in valid.columns:
        if is_directional:
            sep = float(valid["median"].max() - valid["median"].min())
        else:
            sep = float(valid["median"].abs().max() - valid["median"].abs().min())
    else:
        sep = 0.0

    n_distinct = len(set(mapping.values()))
    return AxisEvaluation(
        axis_separation=sep,
        secondary_robustness=float(n_distinct),
        secondary_label="n_distinct_scores",
        raw_details={"horizon": primary_horizon, "is_directional": is_directional,
                     "mapping": {int(k): int(v) for k, v in mapping.items()}},
    )


def evaluate_volatility(*, gamma: NDArray, state_labels: NDArray, raw_data: pd.DataFrame,
                       horizons: tuple[int, ...], **_ignored) -> AxisEvaluation:
    """Volatility axis: forward realized-vol median spread + distinct bucket count."""
    primary_horizon = horizons[0] if horizons else 12
    close = raw_data["close"].values
    if len(close) <= primary_horizon:
        return AxisEvaluation(0.0, 0.0, "n_distinct_buckets",
                              raw_details={"error": "not enough bars for forward vol"})

    log_close = np.log(close)
    log_rets = np.diff(log_close, prepend=log_close[0])
    # Trailing forward window RMS of log-rets — proxy for realized vol over horizon h.
    fwd_vol = np.full(len(close), np.nan)
    # Use convolution for speed: cumulative sum of squared log-rets, windowed.
    sq = log_rets ** 2
    cumsum_sq = np.cumsum(sq)
    for t in range(len(close) - primary_horizon):
        window_sumsq = cumsum_sq[t + primary_horizon] - cumsum_sq[t]
        fwd_vol[t] = float(np.sqrt(window_sumsq / primary_horizon))

    fwd_df = pd.DataFrame({"regime": state_labels[:len(close)], "realized_vol": fwd_vol}).dropna()
    if len(fwd_df) < 30:
        return AxisEvaluation(0.0, 0.0, "n_distinct_buckets",
                              raw_details={"error": "fewer than 30 valid forward-vol rows"})

    mapping, diag = map_regime_to_volatility_score(
        fwd_df, regime_col="regime", vol_proxy_col="realized_vol",
        method="median", return_diagnostics=True,
    )

    if "median_vol" in diag.columns:
        valid = diag.dropna(subset=["median_vol"])
    else:
        valid = diag.iloc[0:0]

    sep = float(valid["median_vol"].max() - valid["median_vol"].min()) if len(valid) >= 2 else 0.0
    n_distinct = len(set(mapping.values()))
    return AxisEvaluation(
        axis_separation=sep,
        secondary_robustness=float(n_distinct),
        secondary_label="n_distinct_buckets",
        raw_details={"horizon": primary_horizon, "mapping": {int(k): int(v) for k, v in mapping.items()}},
    )


def evaluate_path_structure(*, gamma: NDArray, state_labels: NDArray, raw_data: pd.DataFrame,
                           horizons: tuple[int, ...], **_ignored) -> AxisEvaluation:
    """Efficiency / path-structure axis: choppiness-index median spread + distinct scores."""
    required = {"high", "low", "close"}
    missing = required - set(raw_data.columns)
    if missing:
        return AxisEvaluation(0.0, 0.0, "n_distinct_scores",
                              raw_details={"error": f"path-structure requires {required}; missing {missing}"})

    df = raw_data.copy()
    df["regime"] = state_labels[:len(df)]
    mapping, diag = map_regime_to_path_structure_score(df, regime_col="regime", method="id_chop",
                                                       return_diagnostics=True)

    if "median_chop" in diag.columns:
        valid = diag.dropna(subset=["median_chop"])
    else:
        valid = diag.iloc[0:0]

    sep = float(valid["median_chop"].max() - valid["median_chop"].min()) if len(valid) >= 2 else 0.0
    n_distinct = len(set(mapping.values()))
    return AxisEvaluation(
        axis_separation=sep,
        secondary_robustness=float(n_distinct),
        secondary_label="n_distinct_scores",
        raw_details={"mapping": {int(k): int(v) for k, v in mapping.items()}},
    )


def evaluate_liquidity(*, gamma: NDArray, state_labels: NDArray, raw_data: pd.DataFrame,
                       horizons: tuple[int, ...], **_ignored) -> AxisEvaluation:
    """Liquidity axis: forward median volume spread + bucket monotonicity.

    Computes the forward cumulative volume over ``horizons[0]`` bars, then ranks states by median forward volume via
    ``map_regime_to_volatility_score`` (mechanically generic — any non-negative scalar proxy works). Higher buckets
    = more active / liquid (higher forward volume per period).

    **Why forward volume, not Amihud.** The classical Amihud measure ``|forward return| / forward volume`` directly
    captures price impact per unit traded, but introduces explicit volatility contamination via the
    return-magnitude numerator: a volatile feature subset can look "illiquid" just because returns are large.
    Forward median volume retains the activity-based liquidity intuition (active = liquid) without algebraic volatility
    coupling. Residual entanglement remains (volatile periods often have high volume on FX/equity) but it is structural, not formulaic.

    Polarity: with ``map_regime_to_volatility_score`` ranking ascending by median, bucket 0 = lowest forward volume = LEAST liquid;
    bucket k = highest forward volume = MOST liquid. Surfaced in ``raw_details['polarity']``.

    Requires ``raw_data`` to contain ``close`` and a volume column. Accepts ``tick_volume`` or ``volume`` (in that order of preference);
    if neither is present, returns a zero-separation result with a clear error message in ``raw_details``.
    """
    primary_horizon = horizons[0] if horizons else 12

    vol_col: str | None = None
    for c in ("tick_volume", "volume"):
        if c in raw_data.columns:
            vol_col = c
            break
    if vol_col is None:
        return AxisEvaluation(0.0, 0.0, "n_distinct_buckets",
                              raw_details={"error": "liquidity axis requires a 'tick_volume' or 'volume' column"})

    close = raw_data["close"].values
    volume = np.asarray(raw_data[vol_col].values, dtype=float)
    if len(close) <= primary_horizon:
        return AxisEvaluation(0.0, 0.0, "n_distinct_buckets",
                              raw_details={"error": "not enough bars for forward volume"})

    # Forward cumulative volume over horizon h. Guard the trailing window with NaN.
    cumsum_vol = np.cumsum(volume)
    fwd_vol = np.full(len(close), np.nan)
    fwd_vol[:-primary_horizon] = cumsum_vol[primary_horizon:] - cumsum_vol[:-primary_horizon]
    # Treat negative volumes (shouldn't happen for real data but defensive) as missing
    # rather than producing nonsense ranks.
    fwd_vol[fwd_vol < 0] = np.nan

    fwd_df = pd.DataFrame({"regime": state_labels[:len(close)], "fwd_volume": fwd_vol}).dropna()
    if len(fwd_df) < 30:
        return AxisEvaluation(0.0, 0.0, "n_distinct_buckets",
                              raw_details={"error": "fewer than 30 valid forward-volume rows"})

    mapping, diag = map_regime_to_volatility_score(
        fwd_df, regime_col="regime", vol_proxy_col="fwd_volume",
        method="median", return_diagnostics=True,
    )

    valid = diag.dropna(subset=["median_vol"]) if "median_vol" in diag.columns else diag.iloc[0:0]
    sep = float(valid["median_vol"].max() - valid["median_vol"].min()) if len(valid) >= 2 else 0.0
    n_distinct = len(set(mapping.values()))
    return AxisEvaluation(
        axis_separation=sep,
        secondary_robustness=float(n_distinct),
        secondary_label="n_distinct_buckets",
        raw_details={"horizon": primary_horizon, "volume_col": vol_col,
                     "polarity": "ascending = bucket 0 LEAST liquid (lowest forward volume), "
                                 "bucket k MOST liquid",
                     "mapping": {int(k): int(v) for k, v in mapping.items()}},
    )


def _not_implemented_factory(signal_type: str) -> AxisEvaluator:
    """Return a stub evaluator that raises a clear message until a real one lands."""
    def _impl(**_kwargs) -> AxisEvaluation:
        raise NotImplementedError(
            f"No HMM evaluator implemented for signal_type={signal_type!r}. "
            f"Add one in hmm_screener._evaluators when this axis is needed."
        )
    return _impl


# Dispatch from registry SIGNAL_TYPES -> evaluator. Stubs raise on call rather
# than at registration time so that the screener's __init__ doesn't trip on
# unrelated axes.
AXIS_EVALUATORS: dict[str, AxisEvaluator] = {
    "trend": evaluate_direction,
    "momentum": evaluate_momentum,
    "volatility": evaluate_volatility,
    "price_structure": evaluate_path_structure,
    "liquidity": evaluate_liquidity,
    "toxicity": _not_implemented_factory("toxicity"),
    "order_flow": _not_implemented_factory("order_flow"),
    "volume_structure": _not_implemented_factory("volume_structure"),
    "information": _not_implemented_factory("information"),
    "composite": _not_implemented_factory("composite"),
    "regime": _not_implemented_factory("regime"),
    "temporal": _not_implemented_factory("temporal"),
}


def get_evaluator(signal_type: str) -> AxisEvaluator:
    """Look up the evaluator for a registry signal_type."""
    if signal_type not in AXIS_EVALUATORS:
        raise KeyError(f"No evaluator registered for signal_type={signal_type!r}. Known: {sorted(AXIS_EVALUATORS)}")
    return AXIS_EVALUATORS[signal_type]
