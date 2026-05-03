"""Triple-barrier label generation. Path-directional, log-return, side-agnostic.

Label semantics: +1 if upper barrier hit (high reaches it), -1 if lower hit
(low reaches it), 0 on vertical expiry. The caller's primary signal direction
is NOT involved — it is combined with these labels at the meta-labelling layer.

Volatility contract: unitless return-volatility (vol_kind=RETURN). Barriers are
computed multiplicatively against entry price (see
`features.tbm.barriers.compute_barrier_levels`). Volatility is forward-filled
onto `prices.index` before lookup; this is look-ahead-safe because the input
vol series is itself a backward-looking estimate. If `volatility.attrs['vol_kind']`
is set and not 'return', `get_labels` raises (prevents passing price-unit ATR
or annualized vol).

Barrier-touch semantics: an upper barrier is hit on bar `k` iff `high[k]` >=
`upper`; lower iff `low[k]` <= `lower`. Exit price for return calculation is
the BARRIER PRICE on horizontal hits (consistent fill convention; avoids the
wick-touch-with-close-far-from-barrier pathology where label and ret disagree
in sign). Vertical exit price is `close[t1_iloc]`.

Same-bar tie-break: when both barriers are touched in the same bar (high >=
upper AND low <= lower) the label is determined by `same_bar_policy`:
  - WORST_CASE (default): resolve to LOWER (downside-dominant; the safest
    backtest convention because it does not optimistically assume the upper
    target was reached first).
  - UPPER_FIRST: resolve to UPPER (legacy convention, biases longs).
  - LOWER_FIRST: resolve to LOWER.

Entry price is `close[t0]`. No slippage modelled.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from enum import Enum
from math import log
from typing import List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from okmich_quant_features.tbm.barriers import BarrierHit

logger = logging.getLogger(__name__)

VOL_KIND_ATTR = "vol_kind"
ALLOWED_VOL_KIND = "return"

try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False


class BarrierTiePolicy(str, Enum):
    UPPER_FIRST = "upper_first"
    LOWER_FIRST = "lower_first"
    WORST_CASE = "worst_case"


# Numeric encoding for the numba kernel
_POLICY_UPPER = 0
_POLICY_LOWER = 1
_POLICY_WORST = 2

_POLICY_TO_INT = {
    BarrierTiePolicy.UPPER_FIRST: _POLICY_UPPER,
    BarrierTiePolicy.LOWER_FIRST: _POLICY_LOWER,
    BarrierTiePolicy.WORST_CASE: _POLICY_WORST,
}


def _walk_path_python(high_arr: np.ndarray, low_arr: np.ndarray, t0_iloc: int, t1_iloc: int,
                      upper: float, lower: float, upper_active: bool, lower_active: bool,
                      tie_policy: int) -> Tuple[int, int]:
    """Scan bars (t0_iloc, t1_iloc] for the first barrier touch.

    Returns (hit_iloc, label) with hit_iloc == -1 meaning no hit. When both
    barriers are touched in the same bar, `tie_policy` decides:
      0 -> upper wins (label=1), 1 -> lower wins (label=-1),
      2 -> worst-case (lower wins, label=-1).
    """
    for k in range(t0_iloc + 1, t1_iloc + 1):
        u = upper_active and high_arr[k] >= upper
        l = lower_active and low_arr[k] <= lower
        if u and l:
            if tie_policy == 0:
                return k, 1
            return k, -1
        if u:
            return k, 1
        if l:
            return k, -1
    return -1, 0


if _NUMBA_AVAILABLE:
    _walk_path = njit(cache=True)(_walk_path_python)
else:
    _walk_path = _walk_path_python


_WORKER_HIGH_ARR: np.ndarray = None
_WORKER_LOW_ARR: np.ndarray = None
_WORKER_CLOSE_ARR: np.ndarray = None
_WORKER_VOL_ARR: np.ndarray = None
_WORKER_INDEX: pd.DatetimeIndex = None


def _pool_init(high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, vol_arr: np.ndarray,
               index: pd.DatetimeIndex) -> None:
    global _WORKER_HIGH_ARR, _WORKER_LOW_ARR, _WORKER_CLOSE_ARR, _WORKER_VOL_ARR, _WORKER_INDEX
    _WORKER_HIGH_ARR = high_arr
    _WORKER_LOW_ARR = low_arr
    _WORKER_CLOSE_ARR = close_arr
    _WORKER_VOL_ARR = vol_arr
    _WORKER_INDEX = index


def get_labels(events: pd.Series, prices: pd.DataFrame, pt_sl: List[float], volatility: pd.Series,
               min_ret: float = 0.0, num_threads: int = 1,
               same_bar_policy: BarrierTiePolicy = BarrierTiePolicy.WORST_CASE) -> pd.DataFrame:
    """Apply triple-barrier labelling using H/L for barrier touches and the
    BARRIER PRICE for horizontal exit (close[t1] only on vertical expiry).

    See module docstring for vol_kind contract and same-bar tie-break policy.

    Returns DataFrame indexed by t0 with columns: t1, ret, label, barrier.
    """
    if len(pt_sl) != 2:
        raise ValueError(f"pt_sl must have length 2, got {pt_sl}")
    pt, sl = float(pt_sl[0]), float(pt_sl[1])
    if pt < 0 or sl < 0:
        raise ValueError(f"pt_sl multipliers must be >= 0, got {pt_sl}")
    if pt == 0 and sl == 0:
        raise ValueError("at least one of pt_sl must be > 0")

    if not isinstance(same_bar_policy, BarrierTiePolicy):
        # Allow string for convenience
        same_bar_policy = BarrierTiePolicy(same_bar_policy)
    tie_policy_int = _POLICY_TO_INT[same_bar_policy]

    required_cols = {"high", "low", "close"}
    missing = required_cols - set(prices.columns)
    if missing:
        raise ValueError(f"prices missing columns: {sorted(missing)}")

    for col in ("high", "low", "close"):
        if not is_numeric_dtype(prices[col]):
            raise ValueError(f"prices['{col}'] must be numeric dtype, got {prices[col].dtype}")

    if not prices.index.is_monotonic_increasing:
        raise ValueError("prices.index must be monotonic increasing")
    if not prices.index.is_unique:
        raise ValueError("prices.index must be unique")
    if len(prices) == 0:
        raise ValueError("prices must be non-empty")

    # vol_kind contract check
    declared_kind = volatility.attrs.get(VOL_KIND_ATTR) if hasattr(volatility, "attrs") else None
    if declared_kind is not None and declared_kind != ALLOWED_VOL_KIND:
        raise ValueError(
            f"volatility has vol_kind={declared_kind!r}; get_labels requires "
            f"vol_kind={ALLOWED_VOL_KIND!r} (unitless return-volatility). "
            f"Convert price-unit or annualized vol before passing."
        )

    if events.empty:
        return _empty_labels()

    if not events.index.is_unique:
        raise ValueError("events.index must be unique")

    _check_tz_consistency(prices.index, events.index, "prices.index", "events.index")
    if isinstance(volatility.index, pd.DatetimeIndex):
        _check_tz_consistency(prices.index, volatility.index, "prices.index", "volatility.index")
        if len(volatility.index.intersection(prices.index)) == 0:
            raise ValueError("volatility.index does not overlap prices.index")

    high_arr = prices["high"].to_numpy(np.float64)
    low_arr = prices["low"].to_numpy(np.float64)
    close_arr = prices["close"].to_numpy(np.float64)
    vol_arr = volatility.reindex(prices.index, method="ffill").to_numpy(np.float64)
    close_index = prices.index

    t0_ilocs, t1_ilocs, valid_mask = _resolve_event_ilocs(events, close_index)

    if num_threads <= 1:
        rows = _process_block(events, t0_ilocs, t1_ilocs, valid_mask,
                              high_arr, low_arr, close_arr, vol_arr, close_index,
                              pt, sl, min_ret, tie_policy_int)
    else:
        chunks = _split_event_block(events, t0_ilocs, t1_ilocs, valid_mask, num_threads)
        rows = []
        with ProcessPoolExecutor(max_workers=num_threads, initializer=_pool_init,
                                 initargs=(high_arr, low_arr, close_arr, vol_arr, close_index)) as pool:
            futures = [pool.submit(_process_block_worker, ev, t0i, t1i, vm, pt, sl, min_ret, tie_policy_int)
                       for ev, t0i, t1i, vm in chunks]
            for fut in futures:
                rows.extend(fut.result())

    if not rows:
        return _empty_labels()

    df = pd.DataFrame(rows, columns=["t0", "t1", "ret", "label", "barrier"]).set_index("t0").sort_index()
    df.index.name = None
    return df


def apply_min_return_filter(labels: pd.DataFrame, min_ret: float) -> pd.DataFrame:
    """Retroactively zero out small-return labels. Does not mutate input. Leaves
    the `barrier` column untouched."""
    if min_ret <= 0:
        return labels
    out = labels.copy()
    mask = out["ret"].abs() < min_ret
    out.loc[mask, "label"] = 0
    return out


def _check_tz_consistency(idx_a: pd.Index, idx_b: pd.Index, name_a: str, name_b: str) -> None:
    a_tz = getattr(idx_a, "tz", None)
    b_tz = getattr(idx_b, "tz", None)
    if (a_tz is None) != (b_tz is None):
        raise ValueError(f"{name_a} (tz={a_tz}) and {name_b} (tz={b_tz}) tz-awareness mismatch")


def _resolve_event_ilocs(events: pd.Series, close_index: pd.DatetimeIndex
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t0_arr = pd.DatetimeIndex(events.index).to_numpy()
    t1_arr = pd.DatetimeIndex(events.values).to_numpy()

    t0_ilocs = close_index.get_indexer(t0_arr)
    last_iloc = len(close_index) - 1

    t1_ilocs = close_index.searchsorted(t1_arr, side="left").astype(np.int64)
    t1_ilocs = np.minimum(t1_ilocs, last_iloc)

    valid_mask = (t0_ilocs >= 0) & (t1_ilocs > t0_ilocs)
    return t0_ilocs.astype(np.int64), t1_ilocs, valid_mask


def _process_block(events: pd.Series, t0_ilocs: np.ndarray, t1_ilocs: np.ndarray, valid_mask: np.ndarray,
                   high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray, vol_arr: np.ndarray,
                   close_index: pd.DatetimeIndex, pt: float, sl: float, min_ret: float,
                   tie_policy: int) -> List[Tuple]:
    rows: List[Tuple] = []
    event_t0 = events.index
    pt_active = pt > 0
    sl_active = sl > 0

    for i in range(len(events)):
        if not valid_mask[i]:
            t0 = event_t0[i]
            if t0_ilocs[i] < 0:
                logger.warning("event t0=%s not in close index; skipping", t0)
            continue

        t0_iloc = int(t0_ilocs[i])
        t1_iloc = int(t1_ilocs[i])
        vol = float(vol_arr[t0_iloc])
        if not np.isfinite(vol) or vol <= 0:
            logger.warning("event t0=%s has invalid vol=%s; skipping", event_t0[i], vol)
            continue

        entry_price = float(close_arr[t0_iloc])
        if not np.isfinite(entry_price) or entry_price <= 0:
            logger.warning("event t0=%s has invalid entry_price=%s; skipping", event_t0[i], entry_price)
            continue

        # Reject configurations producing a non-positive lower barrier.
        if sl_active and sl * vol >= 1.0:
            logger.warning("event t0=%s sl*vol=%.4f >= 1 (lower barrier non-positive); skipping",
                           event_t0[i], sl * vol)
            continue

        upper = entry_price * (1.0 + pt * vol)
        lower = entry_price * (1.0 - sl * vol)

        hit_iloc, hit_label = _walk_path(high_arr, low_arr, t0_iloc, t1_iloc, upper, lower,
                                         pt_active, sl_active, tie_policy)

        if hit_iloc == -1:
            exit_iloc = t1_iloc
            exit_price = float(close_arr[exit_iloc])
            barrier = BarrierHit.VERTICAL.value
            label = 0
        else:
            exit_iloc = hit_iloc
            # Exit at barrier price, not bar close — avoids the wick-touch / close-far-from-barrier
            # pathology where label and ret disagree in sign.
            exit_price = upper if hit_label == 1 else lower
            barrier = BarrierHit.UPPER.value if hit_label == 1 else BarrierHit.LOWER.value
            label = hit_label

        if not np.isfinite(exit_price) or exit_price <= 0:
            logger.warning("event t0=%s has invalid exit_price=%s; skipping", event_t0[i], exit_price)
            continue

        ret = log(exit_price / entry_price)
        if min_ret > 0 and abs(ret) < min_ret:
            label = 0

        rows.append((event_t0[i], close_index[exit_iloc], ret, label, barrier))

    return rows


def _process_block_worker(events: pd.Series, t0_ilocs: np.ndarray, t1_ilocs: np.ndarray,
                          valid_mask: np.ndarray, pt: float, sl: float, min_ret: float,
                          tie_policy: int) -> List[Tuple]:
    return _process_block(events, t0_ilocs, t1_ilocs, valid_mask,
                          _WORKER_HIGH_ARR, _WORKER_LOW_ARR, _WORKER_CLOSE_ARR, _WORKER_VOL_ARR,
                          _WORKER_INDEX, pt, sl, min_ret, tie_policy)


def _split_event_block(events: pd.Series, t0_ilocs: np.ndarray, t1_ilocs: np.ndarray,
                       valid_mask: np.ndarray, num_threads: int
                       ) -> List[Tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray]]:
    n = len(events)
    chunk_size = max(1, n // num_threads)
    out = []
    for i in range(0, n, chunk_size):
        j = min(i + chunk_size, n)
        out.append((events.iloc[i:j], t0_ilocs[i:j], t1_ilocs[i:j], valid_mask[i:j]))
    return out


def _empty_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=["t1", "ret", "label", "barrier"])
