"""
ExcursionLens - were the entries good, did the exits catch the move, and where should the stop go?

Price behaviour, not the clock, should decide exits, and an exit can only capture or waste what the entry offered.
So the lens scores entries and exits SEPARATELY and puts every number next to what random chance would have produced.

Per-trade definitions (``side`` = +1 long / -1 short; every excursion is >= 0):
    MFE        peak unrealized profit reached while the trade was open
    MAE        largest unrealized loss reached before the trade closed
    realized   what the trade banked (from its own entry/exit prices)
    giveback   MFE - realized: profit handed back before the exit
    potential  peak favourable move FROM THE ENTRY, followed past the exit until the directional move ended. The move ends
               at the first bar after the exit whose adverse extreme sits ``k`` ATR below the best favourable price since
               entry (``k`` = the trade's ``move_end_col`` distance if given, else its stop distance, else
               ``leg_retrace_atr``), or at
               the cap: end of the stamped day, ``max_leg_bars`` after entry, or a data break. potential >= MFE.
    extension  potential - MFE: how far the move kept going after the exit
    missed     potential - realized = giveback + extension ("you missed another 800 pts")
    trailing   what a ``k``-ATR trailing stop from the same entry would have banked: the live-achievable benchmark that
               splits ``missed`` into what a better exit rule could catch and what only hindsight could catch

Three questions, each against a random baseline:
    entry_quality()  MFE/MAE ratio over FIXED windows after entry (so the exit cannot flatter the entry) and the share of
                     entries that reach +k ATR before -k ATR. Baseline: random entries matched on hour of day,
                     volatility tercile and long/short mix.
    exit_quality()   realized vs giveback vs extension, the trailing-stop benchmark, and the drift AFTER the exit.
                     Baseline: random exits drawn from the strategy's own holding times.
    stop_analysis()  MAE vs final result, the recovery curve P(win | MAE >= x), and stop candidates re-simulated on the
                     trade paths. The suggested stop is fitted on earlier trades and judged on later ones (walk-forward).

Causality: ATR is taken from the bar BEFORE the entry bar. ``potential`` and ``extension`` use hindsight by design. They are
evaluation measures only and must never feed a strategy. Excursions use bar highs/lows; a short's path is the bid series plus
``spread`` (pass ``spread`` only when the trade prices include it, so realized and path agree).

Dual-mode, like its siblings:
    from okmich_quant_research.backtesting.excursion_lens import ExcursionLens

    el = ExcursionLens(trades_df, ohlc)                    # any trade list (vectorbt records_readable or simple names)
    el = ExcursionLens.from_portfolio(pf, ohlc)            # completed backtest
    el = ExcursionLens.from_signal(ohlc, signal_fn)        # alpha hunting, no backtest needed
    el.show_dashboard(output_html="excursions.html")
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import cached_property

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba import njit
from plotly.subplots import make_subplots


class EntryFill(StrEnum):
    """Which price of its bar a trade fills at; it decides where the in-trade path starts."""
    CLOSE = "close"  # fill at the bar's close: the in-trade path starts on the NEXT bar (vectorbt's default)
    OPEN = "open"  # fill at the bar's open: the fill bar itself is inside the trade


@dataclass(frozen=True)
class ExcursionConfig:
    """Settings for ``ExcursionLens``. Bar counts are in bars of the ``ohlc`` frame."""
    atr_period: int = 14
    entry_fill: EntryFill = EntryFill.CLOSE
    leg_retrace_atr: float = 2.0  # move-end retrace when trades carry no stop price
    max_leg_bars: int = 500  # safety cap on how far past the entry the potential is followed
    cap_at_day_end: bool = True  # the potential stops at the end of the entry's stamped day (set False for daily bars)
    horizons: tuple[int, ...] | None = None  # entry windows; None -> median hold x (0.25, 0.5, 1, 2, 4)
    barriers_atr: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    post_exit_bars: int | None = None  # after-exit window; None -> median holding bars
    stop_grid_atr: tuple[float, ...] = tuple(np.round(np.arange(0.25, 5.01, 0.25), 2))
    n_stop_folds: int = 5
    n_null: int = 200
    break_gap_multiple: float = 10.0  # a bar gap above this x the median spacing is a data break (weekend)
    seed: int = 0


def _midrank(real: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Share of null draws below ``real``, ties counted half: a null identical to ``real`` sits at 0.5, not 0."""
    real = np.atleast_1d(real)
    null = null.reshape(len(null), -1)
    return np.mean(null < real[None, :], axis=0) + 0.5 * np.mean(null == real[None, :], axis=0)


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Row-wise mean ignoring NaN; an all-NaN row gives NaN without numpy's empty-slice warning."""
    counts = np.sum(np.isfinite(values), axis=1)
    totals = np.nansum(values, axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


# ---------------------------------------------------------------------------
# Numba kernels (price units; "F" = favourable excursion, "A" = adverse excursion, both relative to the entry price)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _fav_adv(side, high_j, low_j, spr_j, p0):
    if side > 0:
        return high_j - p0, p0 - low_j
    return p0 - (low_j + spr_j), (high_j + spr_j) - p0


@njit(cache=True)
def _price_f(side, price_j, spr_j, p0):
    if side > 0:
        return price_j - p0
    return p0 - (price_j + spr_j)


@njit(cache=True)
def _trade_kernel(e_idx, x_idx, side, p0, p1, atr, k_leg, cap_end, offset, open_, high, low, close, spr, seg, stop_grid):
    m, nq, n = e_idx.shape[0], stop_grid.shape[0], high.shape[0]
    mfe, mae, potential = np.zeros(m), np.zeros(m), np.zeros(m)
    leg_end, trail_end = np.full(m, -1), np.full(m, -1)
    trail = np.zeros(m)
    stop_out = np.empty((m, nq))
    stop_hit = np.zeros((m, nq), dtype=np.bool_)
    for i in range(m):
        s, e, x, a, ref = side[i], e_idx[i], x_idx[i], atr[i], p0[i]
        realized = s * (p1[i] - ref)
        first, last_in = e + offset, x + offset - 1
        best, worst = 0.0, 0.0
        for q in range(nq):
            stop_out[i, q] = realized
        for j in range(first, min(last_in, n - 1) + 1):
            f, adv = _fav_adv(s, high[j], low[j], spr[j], ref)
            for q in range(nq):  # a bar meets the stop before it may extend MFE: the order inside a bar is unknown
                if not stop_hit[i, q] and adv >= stop_grid[q] * a:
                    stop_out[i, q] = min(_price_f(s, open_[j], spr[j], ref), -stop_grid[q] * a)
                    stop_hit[i, q] = True
            best = max(best, f)
            worst = max(worst, adv)
        mfe[i] = max(best, realized, 0.0)
        mae[i] = max(worst, -realized, 0.0)

        b = mfe[i]
        end = max(last_in, e)
        if b - realized < k_leg[i] * a:  # the move had not already ended at the exit: follow it
            j = x + offset
            exit_seg = seg[min(max(last_in, e), n - 1)]
            while j <= cap_end[i] and j < n and seg[j] == exit_seg:
                f, adv = _fav_adv(s, high[j], low[j], spr[j], ref)
                if b + adv >= k_leg[i] * a:
                    break
                b = max(b, f)
                end = j
                j += 1
        potential[i] = b
        leg_end[i] = end

        tb, j, done = 0.0, first, False
        while j < n and j <= cap_end[i] and seg[j] == seg[min(first, n - 1)]:
            f, adv = _fav_adv(s, high[j], low[j], spr[j], ref)
            level = tb - k_leg[i] * a
            if -adv <= level:
                trail[i] = min(_price_f(s, open_[j], spr[j], ref), level)
                trail_end[i] = j
                done = True
                break
            tb = max(tb, f)
            j += 1
        if not done:
            last = j - 1
            if last >= first:
                trail[i] = _price_f(s, close[last], spr[last], ref)
                trail_end[i] = last
            else:
                trail_end[i] = e
    return mfe, mae, potential, leg_end, trail, trail_end, stop_out, stop_hit


@njit(cache=True)
def _window_excursions(ref, high, low, spr, seg, offset, horizons):
    """Per bar and horizon: long MFE, long MAE, short MFE, short MAE (price units). NaN when a break cuts the window."""
    n, nh = ref.shape[0], horizons.shape[0]
    out = np.full((4, nh, n), np.nan)
    for t in range(n):
        if not np.isfinite(ref[t]):
            continue
        hi, lo, his, los = -np.inf, np.inf, -np.inf, np.inf
        steps, k, j = 0, 0, t + offset
        while k < nh and j < n and seg[j] == seg[t]:
            hi, lo = max(hi, high[j]), min(lo, low[j])
            his, los = max(his, high[j] + spr[j]), min(los, low[j] + spr[j])
            steps += 1
            j += 1
            if steps == horizons[k]:
                out[0, k, t] = max(hi - ref[t], 0.0)
                out[1, k, t] = max(ref[t] - lo, 0.0)
                out[2, k, t] = max(ref[t] - los, 0.0)
                out[3, k, t] = max(his - ref[t], 0.0)
                k += 1
    return out


@njit(cache=True)
def _first_passage(ref, high, low, spr, atr, seg, offset, barriers, cap):
    """Per bar and barrier k: 1 if +k ATR came first, 0 if -k ATR did, 0.5 if both in one bar, NaN if neither by the cap."""
    n, nk = ref.shape[0], barriers.shape[0]
    out = np.full((2, nk, n), np.nan)
    for t in range(n):
        if not (np.isfinite(ref[t]) and np.isfinite(atr[t]) and atr[t] > 0):
            continue
        for q in range(nk):
            d = barriers[q] * atr[t]
            res_l, res_s = np.nan, np.nan
            j, steps = t + offset, 0
            while steps < cap and j < n and seg[j] == seg[t] and (np.isnan(res_l) or np.isnan(res_s)):
                if np.isnan(res_l):
                    up, dn = high[j] >= ref[t] + d, low[j] <= ref[t] - d
                    res_l = 0.5 if (up and dn) else (1.0 if up else (0.0 if dn else np.nan))
                if np.isnan(res_s):
                    fav, adv = low[j] + spr[j] <= ref[t] - d, high[j] + spr[j] >= ref[t] + d
                    res_s = 0.5 if (fav and adv) else (1.0 if fav else (0.0 if adv else np.nan))
                j += 1
                steps += 1
            out[0, q, t], out[1, q, t] = res_l, res_s
    return out


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ExcursionLens:
    """MAE / MFE / directional-potential analysis of a trade list against its OHLC path. See the module docstring."""

    _BG = "#0d1117"
    _PANEL = "#161b22"
    _BORDER = "#30363d"
    _TEXT = "#e6edf3"
    _SUB = "#8b949e"
    _GREEN = "#3fb950"
    _RED = "#f85149"
    _BLUE = "#58a6ff"
    _ORANGE = "#d29922"
    _PURPLE = "#bc8cff"

    _ALIASES = {
        "entry_time": ("Entry Timestamp", "Entry Index", "entry_time"),
        "exit_time": ("Exit Timestamp", "Exit Index", "exit_time"),
        "entry_price": ("Avg Entry Price", "entry_price"),
        "exit_price": ("Avg Exit Price", "exit_price"),
        "side": ("Direction", "side"),
    }

    def __init__(self, trades_df: pd.DataFrame, ohlc: pd.DataFrame, spread: pd.Series | None = None,
                 stop_col: str | None = None, move_end_col: str | None = None, config: ExcursionConfig | None = None,
                 **config_overrides):
        """
        Parameters
        ----------
        trades_df    : vectorbt ``pf.trades.records_readable`` or a frame with entry_time, exit_time, side (+1/-1 or
                       Long/Short) and optionally entry_price / exit_price. Open trades (``Status == "Open"``) are
                       dropped.
        ohlc         : open/high/low/close bars; every trade timestamp must be one of its index stamps.
        spread       : optional bid-ask spread in PRICE units, aligned to ``ohlc``; added to a short's path.
        stop_col     : optional trades column with each trade's stop price. It enables R units and, unless
                       ``move_end_col`` is given, sets the move-end retrace ``k`` to the trade's own stop distance.
        move_end_col : optional trades column with each trade's move-end retrace distance in PRICE units. Use it when
                       the exit itself is a retrace of the stop distance (e.g. a CTL flip): with ``k`` equal to the
                       exit rule, the move ends exactly at the exit and ``extension`` is zero by construction.
        config       : ``ExcursionConfig``; keyword overrides (e.g. ``n_null=100``) are applied on top.
        """
        self.config = replace(config or ExcursionConfig(), **config_overrides)
        self._ohlc = self._prepare_ohlc(ohlc)
        n = len(self._ohlc)
        self._spr = (np.zeros(n) if spread is None
                     else pd.Series(spread).reindex(self._ohlc.index).fillna(0.0).to_numpy(dtype=float))
        self._seg, self._seg_end, self._day_end = self._segments(self._ohlc.index)
        self._atr_ref = self._atr(self._ohlc, self.config.atr_period).shift(1).to_numpy()
        self._offset = 1 if self.config.entry_fill is EntryFill.CLOSE else 0
        self._ref_price = self._ohlc[self.config.entry_fill.value].to_numpy(dtype=float)
        self.trades = self._prepare_trades(trades_df, stop_col, move_end_col)
        self._compute_trade_metrics()

    # ------------------------------------------------------------------
    # Alternate constructors (dual-mode ingestion)
    # ------------------------------------------------------------------

    @classmethod
    def from_portfolio(cls, portfolio, ohlc: pd.DataFrame, **kwargs) -> "ExcursionLens":
        """Backtest mode: build from a vectorbt ``Portfolio`` via ``portfolio.trades.records_readable``."""
        return cls(portfolio.trades.records_readable, ohlc, **kwargs)

    @classmethod
    def from_signal(cls, ohlc: pd.DataFrame, signal_fn, close_col: str = "close", vbt_kwargs: dict | None = None,
                    **kwargs) -> "ExcursionLens":
        """Alpha-hunting mode: ``signal_fn(ohlc)`` returns signed positions ({-1, 0, +1}); trades come from vectorbt."""
        from .signal_adapter import signal_to_portfolio
        pf = signal_to_portfolio(ohlc, signal_fn, close_col=close_col, **(vbt_kwargs or {}))
        return cls.from_portfolio(pf, ohlc, **kwargs)

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_ohlc(ohlc: pd.DataFrame) -> pd.DataFrame:
        frame = ohlc.copy()
        frame.columns = [str(c).strip().lower() for c in frame.columns]
        missing = {"open", "high", "low", "close"} - set(frame.columns)
        if missing:
            raise ValueError(f"ohlc is missing columns {sorted(missing)}; it needs open, high, low and close.")
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        return frame[["open", "high", "low", "close"]].astype(float)

    def _segments(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gaps = index.to_series().diff()
        spacing = gaps.median()
        breaks = (gaps > spacing * self.config.break_gap_multiple).to_numpy()
        seg = np.cumsum(breaks).astype(np.int64)
        positions = np.arange(len(index))
        seg_end = pd.Series(positions).groupby(seg).transform("max").to_numpy()
        day_end = pd.Series(positions).groupby(np.asarray(index.normalize())).transform("max").to_numpy()
        return seg, seg_end, day_end

    @staticmethod
    def _atr(ohlc: pd.DataFrame, period: int) -> pd.Series:
        prev_close = ohlc["close"].shift(1)
        true_range = pd.concat([ohlc["high"] - ohlc["low"], (ohlc["high"] - prev_close).abs(),
                                (ohlc["low"] - prev_close).abs()], axis=1).max(axis=1)
        return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _column(self, df: pd.DataFrame, key: str, required: bool = True) -> pd.Series | None:
        for name in self._ALIASES[key]:
            if name in df.columns:
                return df[name]
        if required:
            raise ValueError(f"trades_df needs one of the columns {self._ALIASES[key]}.")
        return None

    def _prepare_trades(self, trades_df: pd.DataFrame, stop_col: str | None, move_end_col: str | None) -> pd.DataFrame:
        df = trades_df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        if "Status" in df.columns:
            open_trades = (df["Status"].astype(str) == "Open").to_numpy()
            if open_trades.any():
                warnings.warn(f"ExcursionLens: dropped {int(open_trades.sum())} open trade(s) (no exit yet).")
            df = df[~open_trades]
        side_raw = self._column(df, "side")
        side = (np.sign(side_raw.astype(float)) if pd.api.types.is_numeric_dtype(side_raw)
                else side_raw.astype(str).str.strip().str.lower().map({"long": 1, "short": -1}))
        entry_time = pd.to_datetime(self._column(df, "entry_time"))
        exit_time = pd.to_datetime(self._column(df, "exit_time"))
        index = self._ohlc.index
        e_idx, x_idx = index.get_indexer(entry_time), index.get_indexer(exit_time)
        unmatched = int(((e_idx < 0) | (x_idx < 0)).sum())
        if unmatched:
            raise ValueError(f"{unmatched} trade timestamp(s) are not bars of `ohlc`. Pass the same bars the trades were "
                             "generated on (timestamps must match exactly).")
        out = pd.DataFrame({"entry_time": entry_time.to_numpy(), "exit_time": exit_time.to_numpy(),
                            "side": np.asarray(side, dtype=float), "e_idx": e_idx, "x_idx": x_idx})
        entry_price = self._column(df, "entry_price", required=False)
        exit_price = self._column(df, "exit_price", required=False)
        exit_ref = self._ohlc[self.config.entry_fill.value].to_numpy()
        out["entry_price"] = (entry_price.to_numpy(dtype=float) if entry_price is not None
                              else self._ref_price[e_idx])
        out["exit_price"] = exit_price.to_numpy(dtype=float) if exit_price is not None else exit_ref[x_idx]
        out["stop_price"] = df[stop_col].to_numpy(dtype=float) if stop_col else np.nan
        out["move_end"] = df[move_end_col].to_numpy(dtype=float) if move_end_col else np.nan
        bad = (out["side"].abs() != 1) | (out["x_idx"] < out["e_idx"]) | ~np.isfinite(self._atr_ref[e_idx])
        if bad.any():
            warnings.warn(f"ExcursionLens: dropped {int(bad.sum())} trade(s) with no side, exit before entry, or no ATR "
                          "history yet.")
        return out[~bad].sort_values("entry_time").reset_index(drop=True)

    def _cap_end(self, e_idx: np.ndarray) -> np.ndarray:
        first = np.minimum(e_idx + self._offset, len(self._ohlc) - 1)
        cap = np.minimum(e_idx + self._offset + self.config.max_leg_bars - 1, self._seg_end[first])
        if self.config.cap_at_day_end:
            cap = np.minimum(cap, self._day_end[e_idx])
        return cap.astype(np.int64)

    def _compute_trade_metrics(self) -> None:
        t, o = self.trades, self._ohlc
        e_idx, x_idx = t["e_idx"].to_numpy(np.int64), t["x_idx"].to_numpy(np.int64)
        side, p0, p1 = t["side"].to_numpy(float), t["entry_price"].to_numpy(float), t["exit_price"].to_numpy(float)
        atr = self._atr_ref[e_idx]
        stop_dist = np.abs(p0 - t["stop_price"].to_numpy(float))
        move_end = t["move_end"].to_numpy(float)
        k_leg = np.where(np.isfinite(stop_dist) & (stop_dist > 0), stop_dist / atr, self.config.leg_retrace_atr)
        k_leg = np.where(np.isfinite(move_end) & (move_end > 0), move_end / atr, k_leg)
        grid = np.asarray(self.config.stop_grid_atr, dtype=float)
        mfe, mae, potential, leg_end, trail, trail_end, stop_out, stop_hit = _trade_kernel(
            e_idx, x_idx, side, p0, p1, atr, k_leg, self._cap_end(e_idx), self._offset, o["open"].to_numpy(),
            o["high"].to_numpy(), o["low"].to_numpy(), o["close"].to_numpy(), self._spr, self._seg, grid)
        realized = side * (p1 - p0)
        t["hold_bars"] = x_idx - e_idx
        t["atr_entry"], t["k_leg_atr"] = atr, k_leg
        t["realized"], t["mfe"], t["mae"], t["potential"], t["trailing"] = realized, mfe, mae, potential, trail
        t["giveback"], t["extension"], t["missed"] = mfe - realized, potential - mfe, potential - realized
        for name in ("realized", "mfe", "mae", "potential", "trailing", "giveback", "extension", "missed"):
            t[f"{name}_atr"] = t[name] / atr
            t[f"{name}_bp"] = 1e4 * t[name] / p0
            t[f"{name}_r"] = t[name] / stop_dist
        with np.errstate(divide="ignore", invalid="ignore"):
            t["capture"] = np.where(mfe > 0, realized / mfe, np.nan)
            t["potential_capture"] = np.where(potential > 0, realized / potential, np.nan)
        t["leg_end_time"] = o.index[np.maximum(leg_end, 0)]
        t["trailing_exit_time"] = o.index[np.maximum(trail_end, 0)]
        self._stop_out_atr = stop_out / atr[:, None]
        self._stop_hit = stop_hit
        self._holds = np.maximum(x_idx - e_idx, 1)

    # ------------------------------------------------------------------
    # Entry quality (exit-agnostic, fixed windows, random-entry null)
    # ------------------------------------------------------------------

    @cached_property
    def horizons(self) -> np.ndarray:
        if self.config.horizons:
            return np.unique(np.asarray(self.config.horizons, dtype=np.int64))
        median = max(int(np.median(self._holds)), 1)
        return np.unique(np.maximum(np.round(median * np.array([0.25, 0.5, 1, 2, 4])), 1).astype(np.int64))

    @cached_property
    def _bar_excursions(self) -> tuple[np.ndarray, np.ndarray]:
        o = self._ohlc
        win = _window_excursions(self._ref_price, o["high"].to_numpy(), o["low"].to_numpy(), self._spr, self._seg,
                                 self._offset, self.horizons)
        fp = _first_passage(self._ref_price, o["high"].to_numpy(), o["low"].to_numpy(), self._spr, self._atr_ref,
                            self._seg, self._offset, np.asarray(self.config.barriers_atr, dtype=float),
                            self.config.max_leg_bars)
        return win / self._atr_ref, fp

    def _entry_stats(self, bars: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        win, fp = self._bar_excursions
        is_long = side > 0
        mfe = np.where(is_long, win[0][:, bars], win[2][:, bars])
        mae = np.where(is_long, win[1][:, bars], win[3][:, bars])
        hits = np.where(is_long, fp[0][:, bars], fp[1][:, bars])
        return _nanmean(mfe), _nanmean(mae), _nanmean(hits)

    @cached_property
    def _null_strata(self) -> tuple[np.ndarray, dict]:
        """Stratum per bar: hour of day x volatility tercile (full-sample, evaluation-only); pool = bars with ATR."""
        o = self._ohlc
        rel_vol = self._atr_ref / o["close"].to_numpy()
        valid = np.isfinite(rel_vol)
        lo, hi = np.nanquantile(rel_vol[valid], [1 / 3, 2 / 3])
        tercile = np.where(rel_vol < lo, 0, np.where(rel_vol > hi, 2, 1))
        stratum = np.asarray(o.index.hour) * 3 + tercile
        stratum[~valid] = -1
        pools = {key: np.flatnonzero(stratum == key) for key in np.unique(stratum[valid])}
        return stratum, pools

    def _sample_like(self, anchors: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """One random bar per anchor bar, drawn from the anchor's stratum (hour of day x volatility tercile)."""
        stratum, pools = self._null_strata
        all_bars = np.concatenate(list(pools.values()))
        sampled = np.empty(len(anchors), dtype=np.int64)
        for key in np.unique(stratum[anchors]):
            rows = np.flatnonzero(stratum[anchors] == key)
            pool = pools.get(key, all_bars)
            sampled[rows] = pool[rng.integers(0, len(pool), len(rows))]
        return sampled

    @cached_property
    def _entry_null(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed)
        side, e_idx = self.trades["side"].to_numpy(), self.trades["e_idx"].to_numpy()
        draws = [self._entry_stats(self._sample_like(e_idx, rng), side) for _ in range(self.config.n_null)]
        return tuple(np.stack(part) for part in zip(*draws))

    def entry_quality(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(by_horizon, by_barrier): real statistic, random-entry 2.5/50/97.5 percentiles and the real one's percentile."""
        side = self.trades["side"].to_numpy()
        mfe, mae, hits = self._entry_stats(self.trades["e_idx"].to_numpy(), side)
        null_mfe, null_mae, null_hits = self._entry_null
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio, null_ratio = mfe / mae, null_mfe / null_mae
        by_horizon = pd.DataFrame({"horizon_bars": self.horizons, "mfe_atr": mfe, "mae_atr": mae, "mfe_mae_ratio": ratio})
        by_horizon = by_horizon.join(self._band(ratio, null_ratio))
        by_barrier = pd.DataFrame({"barrier_atr": self.config.barriers_atr, "p_up_first": hits})
        by_barrier = by_barrier.join(self._band(hits, null_hits))
        fp = self._bar_excursions[1]
        bars = self.trades["e_idx"].to_numpy()
        by_barrier["resolved_share"] = np.mean(np.isfinite(np.where(side > 0, fp[0][:, bars], fp[1][:, bars])), axis=1)
        return by_horizon, by_barrier

    @staticmethod
    def _band(real: np.ndarray, null: np.ndarray) -> pd.DataFrame:
        lo, med, hi = np.nanpercentile(null, [2.5, 50, 97.5], axis=0)
        return pd.DataFrame({"random_lo": lo, "random_median": med, "random_hi": hi,
                             "percentile_vs_random": _midrank(real, null)})

    # ------------------------------------------------------------------
    # Exit quality (entries fixed)
    #   timing: your exits vs random exits drawn from your own holding times (same entries)
    #   after-exit drift: did the trade's direction still have edge after the exit? vs the same direction from random
    #   bars matched on the exit bar's hour and volatility tercile (a direction with no edge left)
    # ------------------------------------------------------------------

    @cached_property
    def post_exit_offsets(self) -> np.ndarray:
        w = self.config.post_exit_bars or max(int(np.median(self._holds)), 1)
        return np.unique(np.maximum(np.round(w * np.array([0.25, 0.5, 1, 1.5, 2])), 1).astype(np.int64))

    @property
    def fixed_hold(self) -> bool:
        """True when every trade held the same number of bars: random-exit timing then equals the real exits."""
        return np.unique(self._holds).size == 1

    def _drift_after(self, bars: np.ndarray, side: np.ndarray, ref_price: np.ndarray) -> np.ndarray:
        """Mean move in ``side``'s favour ``d`` bars after ``bars``, from ``ref_price``, in that bar's ATR."""
        n = len(self._ohlc)
        close = self._ohlc["close"].to_numpy()
        atr = self._atr_ref[np.minimum(bars, n - 1)]
        out = np.empty(len(self.post_exit_offsets))
        for k, d in enumerate(self.post_exit_offsets):
            later = bars + d
            ok = (later < n) & (later <= self._seg_end[np.minimum(bars, n - 1)])
            later = np.minimum(later, n - 1)
            move = np.where(side > 0, close[later] - ref_price, ref_price - (close[later] + self._spr[later]))
            out[k] = _nanmean(np.where(ok, move / atr, np.nan)[None, :])[0]
        return out

    def _random_exit_bars(self, rng: np.random.Generator) -> np.ndarray:
        e_idx = self.trades["e_idx"].to_numpy()
        hold = rng.choice(self._holds, len(e_idx))
        return np.minimum(e_idx + hold, self._seg_end[np.minimum(e_idx + self._offset, len(self._ohlc) - 1)])

    @cached_property
    def _exit_null(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed + 1)
        t = self.trades
        side, p0, atr = t["side"].to_numpy(), t["entry_price"].to_numpy(), t["atr_entry"].to_numpy()
        x_idx = t["x_idx"].to_numpy()
        exit_px = self._ohlc[self.config.entry_fill.value].to_numpy()
        close = self._ohlc["close"].to_numpy()
        realized, drift = [], []
        for _ in range(self.config.n_null):
            bars = self._random_exit_bars(rng)
            exit_f = np.where(side > 0, exit_px[bars] - p0, p0 - (exit_px[bars] + self._spr[bars]))
            realized.append(_nanmean((exit_f / atr)[None, :])[0])
            anchors = self._sample_like(x_idx, rng)
            drift.append(self._drift_after(anchors, side, np.where(side > 0, close[anchors],
                                                                    close[anchors] + self._spr[anchors])))
        return np.asarray(realized), np.stack(drift)

    def exit_quality(self) -> tuple[pd.Series, pd.DataFrame]:
        """(summary, after_exit_drift). Summary means are per trade in entry ATR; drift is in ATR at the exit bar."""
        t = self.trades
        null_realized, null_drift = self._exit_null
        real_drift = self._drift_after(t["x_idx"].to_numpy(), t["side"].to_numpy(), t["exit_price"].to_numpy())
        realized_mean = t["realized_atr"].mean()
        summary = pd.Series({
            "trades": len(t), "median_hold_bars": float(np.median(self._holds)), "fixed_hold": float(self.fixed_hold),
            "realized_atr": realized_mean, "mfe_atr": t["mfe_atr"].mean(),
            "giveback_atr": t["giveback_atr"].mean(), "extension_atr": t["extension_atr"].mean(),
            "potential_atr": t["potential_atr"].mean(), "missed_atr": t["missed_atr"].mean(),
            "trailing_atr": t["trailing_atr"].mean(), "median_capture": t["capture"].median(),
            "median_potential_capture": t["potential_capture"].median(),
            "random_exit_lo": np.percentile(null_realized, 2.5), "random_exit_median": np.median(null_realized),
            "random_exit_hi": np.percentile(null_realized, 97.5),
            "exit_percentile_vs_random": float(_midrank(np.array([realized_mean]), null_realized[:, None])[0]),
        })
        drift = pd.DataFrame({"bars_after_exit": self.post_exit_offsets, "drift_atr": real_drift})
        return summary, drift.join(self._band(real_drift, null_drift))

    # ------------------------------------------------------------------
    # Stop analysis (MAE-based, walk-forward)
    # ------------------------------------------------------------------

    def stop_analysis(self) -> dict:
        """Recovery curve, in-sample stop curve, walk-forward stop choice and stop-fill slippage."""
        t = self.trades
        grid = np.asarray(self.config.stop_grid_atr, dtype=float)
        realized, mae = t["realized_atr"].to_numpy(), t["mae_atr"].to_numpy()
        recovery = pd.DataFrame({"mae_at_least_atr": grid})
        recovery["trades_reaching"] = [(mae >= x).sum() for x in grid]
        recovery["p_still_wins"] = [np.mean(realized[mae >= x] > 0) if (mae >= x).any() else np.nan for x in grid]
        recovery["mean_final_atr"] = [realized[mae >= x].mean() if (mae >= x).any() else np.nan for x in grid]
        curve = pd.DataFrame({"stop_atr": grid, "mean_result_atr": self._stop_out_atr.mean(axis=0),
                              "share_stopped": self._stop_hit.mean(axis=0)})
        baseline = realized.mean()

        folds = np.array_split(np.arange(len(t)), self.config.n_stop_folds)
        walk = []
        for f in range(1, len(folds)):
            train, test = np.concatenate(folds[:f]), folds[f]
            train_means = self._stop_out_atr[train].mean(axis=0)
            best = int(np.argmax(train_means))
            use_stop = train_means[best] > realized[train].mean()
            test_result = self._stop_out_atr[test, best].mean() if use_stop else realized[test].mean()
            walk.append({"fold": f, "train_trades": len(train), "test_trades": len(test),
                         "chosen_stop_atr": grid[best] if use_stop else np.nan,
                         "test_with_stop_atr": test_result, "test_without_atr": realized[test].mean(),
                         "improvement_atr": test_result - realized[test].mean()})
        walk = pd.DataFrame(walk)

        best = int(np.argmax(curve["mean_result_atr"]))
        hit = self._stop_hit[:, best]
        slippage = -self._stop_out_atr[hit, best] / grid[best] if hit.any() else np.array([np.nan])
        return {"recovery": recovery, "curve": curve, "baseline_atr": baseline, "walk_forward": walk,
                "best_stop_atr": grid[best], "best_stop_gain_atr": curve["mean_result_atr"].iloc[best] - baseline,
                "stop_fill_mean_x_nominal": float(np.mean(slippage)), "stop_fill_worst_x_nominal": float(np.max(slippage))}

    # ------------------------------------------------------------------
    # Per-quarter stability and verdicts
    # ------------------------------------------------------------------

    def quarterly(self) -> pd.DataFrame:
        t = self.trades
        grouped = t.groupby(t["entry_time"].dt.to_period("Q"))
        table = grouped.agg(trades=("realized_atr", "size"), win_rate=("realized_atr", lambda s: (s > 0).mean()),
                            realized_atr=("realized_atr", "mean"), mfe_atr=("mfe_atr", "mean"),
                            mae_atr=("mae_atr", "mean"), giveback_atr=("giveback_atr", "mean"),
                            extension_atr=("extension_atr", "mean"), potential_atr=("potential_atr", "mean"),
                            trailing_atr=("trailing_atr", "mean"), median_capture=("capture", "median"))
        table.index = table.index.astype(str)
        return table

    def verdicts(self) -> list[str]:
        by_h, by_k = self.entry_quality()
        summary, drift = self.exit_quality()
        stops = self.stop_analysis()
        lines = []
        beat = by_h[by_h["mfe_mae_ratio"] > by_h["random_hi"]]
        worse = by_h[by_h["mfe_mae_ratio"] < by_h["random_lo"]]
        if len(beat):
            row = beat.iloc[0]
            lines.append(f"ENTRIES (MFE/MAE over fixed windows) beat random entries at "
                         f"{', '.join(map(str, beat['horizon_bars']))} bars ({row['mfe_mae_ratio']:.2f} vs random "
                         f"{row['random_lo']:.2f}-{row['random_hi']:.2f} at {int(row['horizon_bars'])} bars).")
        elif len(worse):
            row = worse.iloc[-1]
            lines.append(f"ENTRIES (MFE/MAE over fixed windows) are WORSE than random at "
                         f"{', '.join(map(str, worse['horizon_bars']))} bars ({row['mfe_mae_ratio']:.2f} vs random "
                         f"{row['random_lo']:.2f}-{row['random_hi']:.2f} at {int(row['horizon_bars'])} bars).")
        else:
            lines.append("ENTRIES (MFE/MAE over fixed windows) are indistinguishable from random entries.")
        k_beat = by_k[by_k["p_up_first"] > by_k["random_hi"]]
        k_worse = by_k[by_k["p_up_first"] < by_k["random_lo"]]
        if len(k_beat):
            row = k_beat.iloc[-1]
            lines.append(f"ENTRIES (first passage): +{row['barrier_atr']:g} ATR came before -{row['barrier_atr']:g} ATR "
                         f"{row['p_up_first']:.0%} of the time, above random entries ({row['random_median']:.0%}).")
        elif len(k_worse):
            row = k_worse.iloc[-1]
            lines.append(f"ENTRIES (first passage) point the WRONG way: +{row['barrier_atr']:g} ATR came before "
                         f"-{row['barrier_atr']:g} ATR only {row['p_up_first']:.0%} of the time, below random entries "
                         f"({row['random_median']:.0%}); worse than random at "
                         f"{', '.join(f'{k:g}' for k in k_worse['barrier_atr'])} ATR.")
        else:
            lines.append("ENTRIES (first passage): no different from random entries at any barrier.")
        lines.append(f"EXITS bank {summary['realized_atr']:+.2f} ATR/trade; the move offered "
                     f"{summary['potential_atr']:.2f}. Missed {summary['missed_atr']:.2f} = "
                     f"{summary['giveback_atr']:.2f} handed back in the trade + {summary['extension_atr']:.2f} after exit.")
        lines.append(f"A trailing stop at the move-end distance would have banked {summary['trailing_atr']:+.2f} "
                     f"ATR/trade; random exits with your holding times bank {summary['random_exit_median']:+.2f}.")
        pct = summary["exit_percentile_vs_random"]
        if summary["fixed_hold"]:
            lines.append(f"Exits are a fixed holding time ({int(summary['median_hold_bars'])} bars), so the random-exit "
                         "timing test does not apply.")
        else:
            timing = "ADD value" if pct >= 0.975 else "DESTROY value" if pct <= 0.025 else "are no better than random"
            lines.append(f"Exit timing {timing} vs random exits with your holding times (percentile {pct:.0%}).")
        late = drift.iloc[min(2, len(drift) - 1)]
        band = f"same direction from random bars {late['random_lo']:+.2f} to {late['random_hi']:+.2f}"
        if late["drift_atr"] > late["random_hi"]:
            lines.append(f"After your exits price kept going your way: {late['drift_atr']:+.2f} ATR over "
                         f"{int(late['bars_after_exit'])} bars ({band}). You exit EARLY.")
        elif late["drift_atr"] < late["random_lo"]:
            lines.append(f"After your exits price turned against the trade: {late['drift_atr']:+.2f} ATR over "
                         f"{int(late['bars_after_exit'])} bars ({band}). Exits are well timed.")
        else:
            lines.append(f"After your exits the direction had no edge left: {late['drift_atr']:+.2f} ATR over "
                         f"{int(late['bars_after_exit'])} bars ({band}).")
        walk = stops["walk_forward"]
        improved = int((walk["improvement_atr"] > 0).sum())
        grid = self.config.stop_grid_atr
        edge = (" It sits at the EDGE of the tested grid, so the true optimum may lie outside it."
                if stops["best_stop_atr"] in (min(grid), max(grid)) else "")
        lines.append(f"STOP: best in-sample stop {stops['best_stop_atr']:g} ATR ({stops['best_stop_gain_atr']:+.2f} "
                     f"ATR/trade vs your exits); chosen on earlier trades it improved {improved}/{len(walk)} later folds."
                     f"{edge} Stop fills averaged {stops['stop_fill_mean_x_nominal']:.2f}x the nominal stop "
                     f"(worst {stops['stop_fill_worst_x_nominal']:.2f}x: gaps and spread at the fill).")
        return lines

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _band_traces(self, x, real, lo, hi, name: str, colour: str, row: int, col: int) -> list[dict]:
        return [
            {"trace": go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
             "row": row, "col": col},
            {"trace": go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(139,148,158,0.25)", name="random 95% band", showlegend=False,
                                 hoverinfo="skip"), "row": row, "col": col},
            {"trace": go.Scatter(x=x, y=real, mode="lines+markers", line=dict(color=colour, width=2), name=name,
                                 showlegend=False), "row": row, "col": col},
        ]

    def _table(self, header: list[str], columns: list[list], colour: str) -> go.Table:
        return go.Table(header=dict(values=header, fill_color=self._PANEL, line_color=self._BORDER, align="left",
                                    font=dict(color=colour, size=11, family="'Courier New', monospace")),
                        cells=dict(values=columns, fill_color=self._BG, line_color=self._BORDER, align="left", height=20,
                                   font=dict(color=self._TEXT, size=10, family="'Courier New', monospace")))

    _MIN_POTENTIAL_ATR = 0.5  # panel 10 only: a trade whose move offered less has no meaningful share to bank

    def show_dashboard(self, output_html: str | None = None, height: int = 3000) -> go.Figure:
        t = self.trades
        by_h, by_k = self.entry_quality()
        summary, drift = self.exit_quality()
        stops = self.stop_analysis()
        win = t["realized_atr"] > 0
        colours = np.where(win, self._GREEN, self._RED)

        fig = make_subplots(
            rows=7, cols=2, vertical_spacing=0.04, horizontal_spacing=0.08, row_heights=[1, 1, 1, 1, 1, 1.1, 1.6],
            subplot_titles=[
                "① MAE vs MFE per trade (ATR) - green won, red lost", "② MAE vs final result per trade (ATR)",
                "③ Recovery: P(trade still wins | MAE reached x ATR)", "④ Stop candidates: mean result per trade (ATR)",
                "⑤ Entry: MFE/MAE over fixed windows vs random entries", "⑥ Entry: P(+k ATR before -k ATR) vs random",
                "⑦ Where the move went: mean per trade (ATR)", "⑧ After the exit: move in the trade's favour (ATR)",
                "⑨ Realized vs directional potential per trade (ATR)",
                "⑩ Share of the potential banked (trades whose move offered >= 0.5 ATR)",
                "⑪ Per-quarter stability", "⑫ Walk-forward stop choice", "⑬ Verdicts",
            ],
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}],
                   [{"type": "table", "colspan": 2}, None], [{"type": "table"}, {"type": "table"}]],
        )
        hover = [f"{a:%Y-%m-%d %H:%M} {'L' if s > 0 else 'S'}" for a, s in zip(t["entry_time"], t["side"])]
        fig.add_trace(go.Scattergl(x=t["mae_atr"], y=t["mfe_atr"], mode="markers", text=hover, showlegend=False,
                                   marker=dict(color=colours, size=5, opacity=0.6)), row=1, col=1)
        fig.add_trace(go.Scattergl(x=t["mae_atr"], y=t["realized_atr"], mode="markers", text=hover, showlegend=False,
                                   marker=dict(color=colours, size=5, opacity=0.6)), row=1, col=2)

        rec = stops["recovery"]
        fig.add_trace(go.Scatter(x=rec["mae_at_least_atr"], y=rec["p_still_wins"], mode="lines+markers",
                                 line=dict(color=self._BLUE, width=2), showlegend=False,
                                 customdata=rec["trades_reaching"], hovertemplate="MAE>=%{x} ATR: %{y:.0%} still win "
                                 "(%{customdata} trades)<extra></extra>"), row=2, col=1)
        curve = stops["curve"]
        fig.add_trace(go.Scatter(x=curve["stop_atr"], y=curve["mean_result_atr"], mode="lines+markers",
                                 line=dict(color=self._ORANGE, width=2), showlegend=False), row=2, col=2)
        fig.add_hline(y=stops["baseline_atr"], line=dict(color=self._SUB, dash="dash"), row=2, col=2)
        fig.add_vline(x=stops["best_stop_atr"], line=dict(color=self._GREEN, dash="dot"), row=2, col=2)

        for item in self._band_traces(by_h["horizon_bars"], by_h["mfe_mae_ratio"], by_h["random_lo"], by_h["random_hi"],
                                      "MFE/MAE", self._BLUE, 3, 1):
            fig.add_trace(item["trace"], row=item["row"], col=item["col"])
        for item in self._band_traces(by_k["barrier_atr"], by_k["p_up_first"], by_k["random_lo"], by_k["random_hi"],
                                      "P(+k first)", self._PURPLE, 3, 2):
            fig.add_trace(item["trace"], row=item["row"], col=item["col"])

        bars = ["Realized (your exit)", "Trailing stop", "Random exits", "MFE (in trade)", "Potential (hindsight)"]
        values = [summary["realized_atr"], summary["trailing_atr"], summary["random_exit_median"], summary["mfe_atr"],
                  summary["potential_atr"]]
        fig.add_trace(go.Bar(x=bars, y=values, showlegend=False, text=[f"{v:+.2f}" for v in values],
                             textposition="outside", marker_color=[self._BLUE, self._ORANGE, self._SUB, self._PURPLE,
                                                                   self._GREEN]), row=4, col=1)
        for item in self._band_traces(drift["bars_after_exit"], drift["drift_atr"], drift["random_lo"],
                                      drift["random_hi"], "after-exit drift", self._GREEN, 4, 2):
            fig.add_trace(item["trace"], row=item["row"], col=item["col"])
        fig.add_hline(y=0, line=dict(color=self._SUB, dash="dash"), row=4, col=2)

        top = float(np.nanpercentile(t["potential_atr"], 99)) if len(t) else 1.0
        fig.add_trace(go.Scattergl(x=t["potential_atr"], y=t["realized_atr"], mode="markers", text=hover,
                                   showlegend=False, marker=dict(color=colours, size=5, opacity=0.6)), row=5, col=1)
        fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=self._SUB, dash="dash")), row=5, col=1)
        offered = t["potential_atr"] >= self._MIN_POTENTIAL_ATR
        fig.add_trace(go.Histogram(x=t.loc[offered, "potential_capture"].clip(-1, 1), nbinsx=40, showlegend=False,
                                   marker_color=self._BLUE), row=5, col=2)

        q = self.quarterly().reset_index().rename(columns={"entry_time": "quarter"})
        fig.add_trace(self._table(list(q.columns), [q[c].round(3) if q[c].dtype.kind == "f" else q[c]
                                                    for c in q.columns], self._BLUE), row=6, col=1)
        walk = stops["walk_forward"].round(3).rename(columns={
            "train_trades": "train", "test_trades": "test", "chosen_stop_atr": "stop_atr", "test_with_stop_atr": "with",
            "test_without_atr": "without", "improvement_atr": "gain"})
        fig.add_trace(self._table(list(walk.columns), [walk[c] for c in walk.columns], self._ORANGE), row=7, col=1)
        fig.add_trace(self._table(["Verdicts (vs random baselines)"], [self.verdicts()], self._GREEN), row=7, col=2)

        axis_titles = {(1, 1): ("MAE (ATR)", "MFE (ATR)"), (1, 2): ("MAE (ATR)", "final result (ATR)"),
                       (2, 1): ("MAE reached (ATR)", "share still winning"), (2, 2): ("stop (ATR)", "mean result (ATR)"),
                       (3, 1): ("bars after entry", "MFE / MAE"), (3, 2): ("k (ATR)", "P(+k before -k)"),
                       (4, 2): ("bars after exit", "drift (ATR)"), (5, 1): ("potential (ATR)", "realized (ATR)"),
                       (5, 2): ("realized / potential", "trades")}
        for (row, col), (x_title, y_title) in axis_titles.items():
            fig.update_xaxes(title_text=x_title, row=row, col=col)
            fig.update_yaxes(title_text=y_title, row=row, col=col)

        fig.update_layout(
            height=height,
            title=dict(text=("<b>Excursion Lens</b>  <span style='color:#8b949e; font-size:13px'>| Were the entries "
                             "good, did the exits catch the move, and where should the stop go?</span>"),
                       font=dict(family="'Courier New', monospace", size=18, color=self._TEXT), x=0.5, xanchor="center",
                       y=0.995),
            paper_bgcolor=self._BG, plot_bgcolor=self._PANEL,
            font=dict(family="'Courier New', monospace", color=self._TEXT, size=11),
            margin=dict(l=60, r=40, t=90, b=40),
        )
        axis_style = dict(gridcolor=self._BORDER, zerolinecolor=self._BORDER, tickfont=dict(size=9, color=self._SUB),
                          showgrid=True, title_font=dict(size=10, color=self._SUB))
        fig.update_xaxes(**axis_style)
        fig.update_yaxes(**axis_style)
        for ann in fig.layout.annotations:
            ann.font.update(size=12, color=self._SUB, family="'Courier New', monospace")

        if output_html:
            fig.write_html(output_html)
            print(f"Dashboard saved -> {output_html}")
        else:
            fig.show()
        return fig
