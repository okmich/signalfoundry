"""
ExcursionLens - were the entries good, did the exits catch the move, and where should the stop go?

Price behaviour, not the clock, should decide exits, and an exit can only capture or waste what the entry offered.
So the lens scores entries and exits SEPARATELY and puts every number next to what random chance would have produced.

Per-trade definitions (``side`` = +1 long / -1 short; every excursion is >= 0):
    MFE        peak unrealized profit reached while the trade was open
    MAE        largest unrealized loss reached before the trade closed
    realized   what the trade banked (from its own entry/exit prices)
    giveback   MFE - realized: profit handed back before the exit
    potential  peak favourable move FROM THE ENTRY until the directional move ended. The move ends at the first bar,
               inside the trade or after it, whose adverse extreme sits ``k`` ATR below the best favourable price
               since entry (``k`` = the trade's ``move_end_col`` distance if given, else its stop distance, else
               ``leg_retrace_atr``), or at the cap: end of the stamped day of the trade's LAST HELD bar,
               ``max_leg_bars`` after entry, or a data break after the exit. The cap never falls inside the trade.
               If the move ended while the trade was open, a later rally is a NEW move: potential = MFE.
               potential >= MFE.
    extension  potential - MFE: how far the move kept going after the exit
    missed     potential - realized = giveback + extension ("you missed another 800 pts")
    trailing   what a ``k``-ATR trailing stop from the same entry would have banked: the live-achievable benchmark
               that splits ``missed`` into what a better exit rule could catch and what only hindsight could catch.
               It uses the same k-ATR retrace rule as potential and always covers at least the real trade's holding
               period, gaps included.

Three questions, each against a random baseline:
    entry_quality()  MFE/MAE ratio over FIXED windows after entry (so the exit cannot flatter the entry) and the share
                     of entries that reach +k ATR before -k ATR.
    exit_quality()   realized vs giveback vs extension, the trailing-stop benchmark, and the drift AFTER the exit.
    stop_analysis()  MAE vs final result, the recovery curve P(win | MAE >= x), and stop candidates re-simulated on
                     the trade paths. The suggested stop is fitted on earlier trades and judged on later ones
                     (walk-forward, with trades still open at the test fold's start purged from training).

Baselines:
    entries and after-exit drift  CIRCULAR SHIFT of the whole trade schedule by a random number of whole calendar
                     days, on the wall clock. Spacing, overlap, long/short sequence and hour of day are kept; only the
                     alignment with price changes. Independent random bars would ignore that real trades overlap and
                     repeat a side, which makes the bands far too narrow (a random-walk book of back-to-back trades
                     fell outside its "95%" band 25% of the time). Same idea as
                     ``timing_significance.circular_shift_null``.
    exit timing      random exits drawn from the strategy's own holding times, entries fixed.
    verdicts         each family of cells (horizons, barriers, after-exit offsets) is judged with ONE family-wise max-z
                     test (Westfall-Young) against the same null, so testing five cells does not buy five chances.
                     The per-cell bands in the tables and charts are point-wise. See ``significance()``.

Timing vs fills: exit timing and after-exit drift are measured from the exit BAR's fill price (its close or open,
plus spread for a short), for the real trades and the baselines alike. Slippage, an intrabar stop fill or a limit
fill therefore never reads as good or bad timing; ``exit_quality()`` reports it separately as ``fill_effect_atr``.

Causality: ATR is taken from the bar BEFORE the entry bar. ``potential`` and ``extension`` use hindsight by design.
They are evaluation measures only and must never feed a strategy. Excursions use bar highs/lows (the bid); with
``spread`` a long buys at bid + spread and a short covers at bid + spread (pass ``spread`` only when the trade prices
include it, so realized and path agree; trades without prices are priced that way).

Dual-mode, like its siblings:
    from okmich_quant_research.backtesting.excursion_lens import ExcursionLens

    el = ExcursionLens(trades_df, ohlc)                    # any trade list (vectorbt records_readable or simple names)
    el = ExcursionLens.from_portfolio(pf, ohlc)            # completed backtest (one portfolio column)
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
    cap_at_day_end: bool | None = None  # cap the potential at the last held bar's day end; None -> only intraday bars
    horizons: tuple[int, ...] | None = None  # entry windows; None -> median hold x (0.25, 0.5, 1, 2, 4)
    barriers_atr: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
    post_exit_bars: int | None = None  # after-exit window; None -> median holding bars
    stop_grid_atr: tuple[float, ...] = tuple(np.round(np.arange(0.25, 5.01, 0.25), 2))
    n_stop_folds: int = 5
    n_null: int = 200
    break_gap_multiple: float = 10.0  # a bar gap above this x the median spacing is a data break (weekend)
    seed: int = 0

    def __post_init__(self):
        object.__setattr__(self, "entry_fill", EntryFill(self.entry_fill))  # accepts "open" / "close" from YAML or JSON
        problems = []
        if self.atr_period < 1:
            problems.append(f"atr_period must be >= 1, got {self.atr_period}")
        if not self.leg_retrace_atr > 0:
            problems.append(f"leg_retrace_atr must be > 0, got {self.leg_retrace_atr}")
        if self.max_leg_bars < 1:
            problems.append(f"max_leg_bars must be >= 1, got {self.max_leg_bars}")
        if self.horizons is not None and any(int(h) != h or h < 1 for h in self.horizons):
            problems.append(f"horizons must be whole numbers of bars >= 1, got {self.horizons}")
        if not self.barriers_atr or any(not k > 0 for k in self.barriers_atr):
            problems.append(f"barriers_atr must be non-empty and > 0, got {self.barriers_atr}")
        if self.post_exit_bars is not None and self.post_exit_bars < 1:
            problems.append(f"post_exit_bars must be >= 1, got {self.post_exit_bars}")
        if not self.stop_grid_atr or any(not k > 0 for k in self.stop_grid_atr):
            problems.append(f"stop_grid_atr must be non-empty and > 0, got {self.stop_grid_atr}")
        if self.n_stop_folds < 2:
            problems.append(f"n_stop_folds must be >= 2 (one fold to fit, one to test), got {self.n_stop_folds}")
        if self.n_null < 1:
            problems.append(f"n_null must be >= 1, got {self.n_null}")
        if not self.break_gap_multiple > 1:
            problems.append(f"break_gap_multiple must be > 1, got {self.break_gap_multiple}")
        if problems:
            raise ValueError("ExcursionConfig: " + "; ".join(problems))


_SIGNIFICANCE = 0.025  # per side of a two-sided 5% family-wise test
_MIN_STOP_COVERAGE = 0.5  # a stop level is a candidate only if it can be simulated on at least this share of trades


def _midrank(real: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Share of FINITE null draws below ``real``, ties counted half (a null identical to ``real`` sits at 0.5).

    Missing draws are left out, exactly as ``np.nanpercentile`` leaves them out of the band.
    """
    real = np.atleast_1d(real)
    null = null.reshape(len(null), -1)
    finite = np.isfinite(null)
    count = finite.sum(axis=0)
    below = ((null < real[None, :]) & finite).sum(axis=0)
    ties = ((null == real[None, :]) & finite).sum(axis=0)
    return np.where(count > 0, (below + 0.5 * ties) / np.maximum(count, 1), np.nan)


def _nanmean(values: np.ndarray) -> np.ndarray:
    """Row-wise mean ignoring NaN; an all-NaN row gives NaN without numpy's empty-slice warning."""
    counts = np.sum(np.isfinite(values), axis=1)
    totals = np.nansum(np.where(np.isfinite(values), values, 0.0), axis=1)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def _column_nanmean(values: np.ndarray) -> np.ndarray:
    """Column-wise mean ignoring NaN, without the empty-slice warning."""
    return _nanmean(values.T)


def _familywise(real: np.ndarray, null: np.ndarray) -> tuple[float, float]:
    """Family-wise p-values that the real curve beats / trails the null across ALL its cells (max-z, Westfall-Young).

    Real and null draws are exchangeable under the null, so all of them get the same treatment: each cell is
    standardised by the mean and sd of the OTHER draws, a draw's score is its largest (smallest) z across the cells,
    and the p-value is the real score's rank among all scores. A cell missing in a draw is skipped for that draw.
    """
    pool = np.vstack([np.atleast_1d(real)[None, :], null.reshape(len(null), -1)])
    finite = np.isfinite(pool)
    values = np.where(finite, pool, 0.0)
    others = finite.sum(axis=0)[None, :] - finite
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (values.sum(axis=0)[None, :] - values) / others
        var = ((values ** 2).sum(axis=0)[None, :] - values ** 2 - others * mean ** 2) / (others - 1)
        z = np.where(finite & (var > 0), (pool - mean) / np.sqrt(var), np.nan)
    high = np.where(np.isfinite(z), z, -np.inf).max(axis=1)
    low = np.where(np.isfinite(z), z, np.inf).min(axis=1)
    n = pool.shape[0]
    p_beat = (1 + np.sum(high[1:] >= high[0])) / n if np.isfinite(high[0]) else 1.0
    p_worse = (1 + np.sum(low[1:] <= low[0])) / n if np.isfinite(low[0]) else 1.0
    return float(p_beat), float(p_worse)


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
def _trade_kernel(e_idx, x_idx, side, p0, p1, atr, k_leg, cap_end, offset, open_, high, low, close, spr, seg,
                  stop_grid):
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
        held_end = min(max(last_in, e), n - 1)
        best, worst = 0.0, 0.0
        ended_at = -1  # first held bar whose adverse extreme retraced k ATR from the best so far: the move ended there
        for q in range(nq):
            stop_out[i, q] = realized
        for j in range(first, min(last_in, n - 1) + 1):
            f, adv = _fav_adv(s, high[j], low[j], spr[j], ref)
            for q in range(nq):  # a bar meets the stop before it may extend MFE: the order inside a bar is unknown
                if not stop_hit[i, q] and adv >= stop_grid[q] * a:
                    stop_out[i, q] = min(_price_f(s, open_[j], spr[j], ref), -stop_grid[q] * a)
                    stop_hit[i, q] = True
            if ended_at < 0 and best + adv >= k_leg[i] * a:
                ended_at = j
            best = max(best, f)
            worst = max(worst, adv)
        mfe[i] = max(best, realized, 0.0)
        mae[i] = max(worst, -realized, 0.0)

        b = mfe[i]
        end = held_end if ended_at < 0 else ended_at
        exit_seg = seg[held_end]
        if ended_at < 0 and b - realized < k_leg[i] * a:  # the move had not ended in the trade or at the exit
            j = x + offset
            while j <= cap_end[i] and j < n and seg[j] == exit_seg:
                f, adv = _fav_adv(s, high[j], low[j], spr[j], ref)
                if b + adv >= k_leg[i] * a:
                    break
                b = max(b, f)
                end = j
                j += 1
        potential[i] = b
        leg_end[i] = end

        # the benchmark may cross breaks while the real trade is open (it was exposed too), not after it closes
        tb, j, done = 0.0, first, False
        while j < n and j <= cap_end[i] and (j <= held_end or seg[j] == exit_seg):
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
def _window_excursions(targets, ref, high, low, spr, atr, seg, offset, horizons):
    """For each target bar and horizon: long MFE, long MAE, short MFE, short MAE in units of that bar's ATR. NaN when
    a break cuts the window or the bar has no positive ATR.

    A long buys at ref + spread and sells at the bid; a short sells at ref and covers at the bid + spread.
    """
    n, nh, m = ref.shape[0], horizons.shape[0], targets.shape[0]
    out = np.full((4, nh, m), np.nan)
    for c in range(m):
        t = targets[c]
        if not (np.isfinite(ref[t]) and np.isfinite(atr[t]) and atr[t] > 0):
            continue
        ref_long = ref[t] + spr[t]
        hi, lo, his, los = -np.inf, np.inf, -np.inf, np.inf
        steps, k, j = 0, 0, t + offset
        while k < nh and j < n and seg[j] == seg[t]:
            hi, lo = max(hi, high[j]), min(lo, low[j])
            his, los = max(his, high[j] + spr[j]), min(los, low[j] + spr[j])
            steps += 1
            j += 1
            if steps == horizons[k]:
                out[0, k, c] = max(hi - ref_long, 0.0) / atr[t]
                out[1, k, c] = max(ref_long - lo, 0.0) / atr[t]
                out[2, k, c] = max(ref[t] - los, 0.0) / atr[t]
                out[3, k, c] = max(his - ref[t], 0.0) / atr[t]
                k += 1
    return out


@njit(cache=True)
def _first_passage(targets, ref, high, low, spr, atr, seg, offset, barriers, cap):
    """Per target bar and barrier k: 1 if +k ATR came first, 0 if -k ATR did, 0.5 if both in one bar, NaN if neither
    by the cap. One forward scan resolves every barrier. Same fills as ``_window_excursions``.
    """
    n, nk, m = ref.shape[0], barriers.shape[0], targets.shape[0]
    out = np.full((2, nk, m), np.nan)
    for c in range(m):
        t = targets[c]
        if not (np.isfinite(ref[t]) and np.isfinite(atr[t]) and atr[t] > 0):
            continue
        ref_long = ref[t] + spr[t]
        res_l, res_s = np.full(nk, np.nan), np.full(nk, np.nan)
        unresolved = 2 * nk
        j, steps = t + offset, 0
        while steps < cap and j < n and seg[j] == seg[t] and unresolved > 0:
            for q in range(nk):
                d = barriers[q] * atr[t]
                if np.isnan(res_l[q]):
                    up, dn = high[j] >= ref_long + d, low[j] <= ref_long - d
                    if up or dn:
                        res_l[q] = 0.5 if (up and dn) else (1.0 if up else 0.0)
                        unresolved -= 1
                if np.isnan(res_s[q]):
                    fav, adv = low[j] + spr[j] <= ref[t] - d, high[j] + spr[j] >= ref[t] + d
                    if fav or adv:
                        res_s[q] = 0.5 if (fav and adv) else (1.0 if fav else 0.0)
                        unresolved -= 1
            j += 1
            steps += 1
        out[0, :, c] = res_l
        out[1, :, c] = res_s
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

    def __init__(self, trades_df: pd.DataFrame, ohlc: pd.DataFrame,
                 spread: pd.Series | np.ndarray | float | None = None, stop_col: str | None = None,
                 move_end_col: str | None = None, config: ExcursionConfig | None = None, **config_overrides):
        """
        Parameters
        ----------
        trades_df    : vectorbt ``pf.trades.records_readable`` (ONE portfolio column) or a frame with entry_time,
                       exit_time, side (+1/-1 or Long/Short) and optionally entry_price / exit_price. Open trades
                       (``Status == "Open"`` or no exit time) are dropped. Without prices, trades are priced at the
                       bar's fill (``entry_fill``), a long paying the spread on entry and a short on exit.
        ohlc         : open/high/low/close bars; every trade timestamp must be one of its index stamps.
        spread       : optional bid-ask spread in PRICE units: a scalar, an array with one value per ``ohlc`` row, or a
                       Series indexed like ``ohlc``. A long pays it on entry, a short on exit.
        stop_col     : optional trades column with each trade's initial stop price, on the loss side of the entry. It
                       enables R units, lets the stop analysis recognise grid levels wider than a stop the trade hit,
                       and, unless ``move_end_col`` is given, sets the move-end retrace ``k`` to the stop distance. A
                       stop at or beyond breakeven is not a risk distance and is ignored for that trade (warning).
        move_end_col : optional trades column with each trade's move-end retrace distance in PRICE units. Use it when
                       the exit itself is a retrace of the stop distance (e.g. a CTL flip): with ``k`` equal to the
                       exit rule, the move ends exactly at the exit and ``extension`` is zero by construction.
        config       : ``ExcursionConfig``; keyword overrides (e.g. ``n_null=100``) are applied on top.
        """
        self.config = replace(config or ExcursionConfig(), **config_overrides)
        self._ohlc = self._prepare_ohlc(ohlc)
        self._spr = self._spread_array(spread, ohlc.index)
        self._seg, self._seg_end, self._day_end = self._segments(self._ohlc.index)
        spacing = self._ohlc.index.to_series().diff().median()
        self._cap_day = (self.config.cap_at_day_end if self.config.cap_at_day_end is not None
                         else bool(pd.notna(spacing) and spacing < pd.Timedelta(days=1)))
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
        """Backtest mode: build from a single-column vectorbt ``Portfolio`` (select one with ``pf[col]`` first).

        vectorbt's trade records carry no stop price. If the portfolio used stops, build the records yourself, add
        each trade's stop price and pass ``stop_col``; otherwise stop levels wider than a stop a trade hit cannot be
        recognised as unknowable. ``from_signal`` does this for a scalar, non-trailing ``sl_stop``.
        """
        return cls(portfolio.trades.records_readable, ohlc, **kwargs)

    @classmethod
    def from_signal(cls, ohlc: pd.DataFrame, signal_fn, close_col: str = "close", vbt_kwargs: dict | None = None,
                    **kwargs) -> "ExcursionLens":
        """Alpha-hunting mode: ``signal_fn(ohlc)`` returns signed positions ({-1, 0, +1}); trades come from vectorbt.

        Trades are priced on ``ohlc``'s own close (column names match case-insensitively), the same series the lens
        measures the path on. A scalar, non-trailing ``sl_stop`` in ``vbt_kwargs`` becomes each trade's stop price.
        """
        from .signal_adapter import signal_to_portfolio
        names = {str(c).strip().lower(): c for c in ohlc.columns}
        if str(close_col).strip().lower() != "close" or "close" not in names:
            raise ValueError(f"from_signal prices trades on ohlc's close, the series the lens measures; got close_col="
                             f"{close_col!r}. Pass a frame whose open/high/low/close describe the traded price.")
        vbt = dict(vbt_kwargs or {})
        pf = signal_to_portfolio(ohlc, signal_fn, close_col=names["close"], open_col=names.get("open", "open"), **vbt)
        records = pf.trades.records_readable
        stop = vbt.get("sl_stop")
        if stop is not None and "stop_col" not in kwargs:
            if np.ndim(stop) == 0 and not vbt.get("sl_trail", False):
                records = records.copy()
                is_long = records["Direction"].astype(str).str.strip().str.lower() == "long"
                entry = records["Avg Entry Price"].to_numpy(dtype=float)
                records["Stop Price"] = np.where(is_long, entry * (1 - float(stop)), entry * (1 + float(stop)))
                kwargs["stop_col"] = "Stop Price"
            else:
                warnings.warn("ExcursionLens: sl_stop is per-bar or trailing, so stop prices cannot be rebuilt; stop "
                              "levels wider than a stop a trade hit are not recognised as unknowable.")
        return cls(records, ohlc, **kwargs)

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

    def _spread_array(self, spread, raw_index: pd.Index) -> np.ndarray:
        """Spread per prepared bar. Scalars broadcast; arrays align by position to the ORIGINAL ``ohlc`` rows."""
        n = len(self._ohlc)
        if spread is None:
            return np.zeros(n)
        if np.ndim(spread) == 0:
            value = float(spread)
            if not (np.isfinite(value) and value >= 0):
                raise ValueError(f"spread must be a finite, non-negative price distance, got {spread!r}")
            return np.full(n, value)
        if isinstance(spread, pd.Series):
            series = spread.astype(float)
        else:
            values = np.asarray(spread, dtype=float)
            if values.ndim != 1 or values.size != len(raw_index):
                raise ValueError(f"spread as an array needs one value per ohlc row ({len(raw_index)}), got shape "
                                 f"{values.shape}; pass a Series to align by timestamp instead.")
            series = pd.Series(values, index=raw_index)
        aligned = series[~series.index.duplicated(keep="last")].reindex(self._ohlc.index)
        missing = int(aligned.isna().sum())
        if missing == n:
            raise ValueError("spread shares no timestamps with ohlc; check its index (and timezone).")
        if missing:
            warnings.warn(f"ExcursionLens: spread is missing on {missing} bar(s); they are treated as zero spread.")
        values = aligned.fillna(0.0).to_numpy(dtype=float)
        if (values < 0).any():
            raise ValueError("spread must be non-negative.")
        return values

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
        if "Column" in df.columns and df["Column"].nunique() > 1:
            raise ValueError(f"trades_df holds trades from {df['Column'].nunique()} portfolio columns, but one ohlc "
                             "path can only score one of them. Select a column first (pf[col], or filter on 'Column').")
        open_trades = pd.to_datetime(self._column(df, "exit_time")).isna().to_numpy()
        if "Status" in df.columns:
            open_trades |= (df["Status"].astype(str) == "Open").to_numpy()
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
            raise ValueError(f"{unmatched} trade timestamp(s) are not bars of `ohlc`. Pass the same bars the trades "
                             "were generated on (timestamps must match exactly).")
        side = np.asarray(side, dtype=float)
        out = pd.DataFrame({"entry_time": entry_time.to_numpy(), "exit_time": exit_time.to_numpy(),
                            "side": side, "e_idx": e_idx, "x_idx": x_idx})
        entry_price = self._column(df, "entry_price", required=False)
        exit_price = self._column(df, "exit_price", required=False)
        ref, spr = self._ref_price, self._spr
        out["entry_price"] = (entry_price.to_numpy(dtype=float) if entry_price is not None
                              else np.where(side > 0, ref[e_idx] + spr[e_idx], ref[e_idx]))
        out["exit_price"] = (exit_price.to_numpy(dtype=float) if exit_price is not None
                             else np.where(side > 0, ref[x_idx], ref[x_idx] + spr[x_idx]))
        out["stop_price"] = df[stop_col].to_numpy(dtype=float) if stop_col else np.nan
        out["move_end"] = df[move_end_col].to_numpy(dtype=float) if move_end_col else np.nan
        atr = self._atr_ref[e_idx]
        bad = ((out["side"].abs() != 1) | (out["x_idx"] < out["e_idx"]) | ~(np.isfinite(atr) & (atr > 0))
               | ~np.isfinite(out["entry_price"]) | ~np.isfinite(out["exit_price"]))
        if bad.any():
            warnings.warn(f"ExcursionLens: dropped {int(bad.sum())} trade(s) with no side, exit before entry, a "
                          "missing entry/exit price, or no positive ATR yet.")
        out = out[~bad].sort_values("entry_time").reset_index(drop=True)
        if out.empty:
            raise ValueError("ExcursionLens: no closed trades left to analyse (all were open or filtered out).")
        return out

    def _cap_end(self, e_idx: np.ndarray, x_idx: np.ndarray) -> np.ndarray:
        """Last bar the potential and trailing benchmark may reach, anchored on the trade's LAST HELD bar.

        Anchoring on the entry bar would end the benchmark before a trade that crosses midnight, a break or the
        ``max_leg_bars`` cap had even closed (a 23:55 close-fill entry would get no benchmark at all).
        """
        n = len(self._ohlc)
        held_end = np.clip(np.maximum(x_idx + self._offset - 1, e_idx), 0, n - 1)
        cap = np.minimum(e_idx + self._offset + self.config.max_leg_bars - 1, self._seg_end[held_end])
        if self._cap_day:
            cap = np.minimum(cap, self._day_end[held_end])
        return np.maximum(cap, held_end).astype(np.int64)

    def _compute_trade_metrics(self) -> None:
        t, o = self.trades, self._ohlc
        e_idx, x_idx = t["e_idx"].to_numpy(np.int64), t["x_idx"].to_numpy(np.int64)
        side, p0, p1 = t["side"].to_numpy(float), t["entry_price"].to_numpy(float), t["exit_price"].to_numpy(float)
        atr = self._atr_ref[e_idx]
        risk = side * (p0 - t["stop_price"].to_numpy(float))  # positive only when the stop is on the loss side
        wrong_side = np.isfinite(risk) & (risk <= 0)
        if wrong_side.any():
            warnings.warn(f"ExcursionLens: {int(wrong_side.sum())} trade(s) have a stop at or beyond breakeven; it is "
                          "not a risk distance and is ignored for those trades (no R units, default k).")
        stop_dist = np.where(np.isfinite(risk) & (risk > 0), risk, np.nan)
        move_end = t["move_end"].to_numpy(float)
        k_leg = np.where(np.isfinite(stop_dist), stop_dist / atr, self.config.leg_retrace_atr)
        k_leg = np.where(np.isfinite(move_end) & (move_end > 0), move_end / atr, k_leg)
        grid = np.asarray(self.config.stop_grid_atr, dtype=float)
        mfe, mae, potential, leg_end, trail, trail_end, stop_out, stop_hit = _trade_kernel(
            e_idx, x_idx, side, p0, p1, atr, k_leg, self._cap_end(e_idx, x_idx), self._offset, o["open"].to_numpy(),
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

        # A trade that hit its OWN stop has an unknown future: a wider grid stop would have kept it open. Those
        # cells cannot be simulated and are left out, never filled with the original stop-out loss.
        own_stop = stop_dist / atr
        stopped_by_own = np.isfinite(own_stop) & (mae >= stop_dist * (1 - 1e-9))
        unknowable = stopped_by_own[:, None] & (grid[None, :] > own_stop[:, None] * (1 + 1e-9))
        self._stop_out_atr = np.where(unknowable, np.nan, stop_out / atr[:, None])
        self._stop_hit = stop_hit & ~unknowable
        self._holds = np.maximum(x_idx - e_idx, 1)

    def _fill_at(self, bars: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Exit fill at ``bars`` (-1 = no bar -> NaN): the bar's close/open, plus the spread when a short covers."""
        safe = np.where(bars >= 0, bars, 0)
        fill = np.where(side > 0, self._ref_price[safe], self._ref_price[safe] + self._spr[safe])
        return np.where(bars >= 0, fill, np.nan)

    # ------------------------------------------------------------------
    # Circular-shift null (shared by entries and after-exit drift)
    # ------------------------------------------------------------------

    @cached_property
    def _wall(self) -> pd.DatetimeIndex:
        """Wall-clock stamps: a whole-day shift keeps the hour of day even across a DST change."""
        index = self._ohlc.index
        return index.tz_localize(None) if index.tz is not None else index

    @cached_property
    def _wall_lookup(self) -> pd.Series:
        """Wall time -> first bar position (a DST fall-back hour repeats wall times; the first one is kept)."""
        wall = self._wall
        return pd.Series(np.arange(len(wall)), index=wall)[~wall.duplicated(keep="first")]

    @cached_property
    def _span_days(self) -> int:
        return int((self._wall[-1].normalize() - self._wall[0].normalize()).days) + 1

    @cached_property
    def _shift_days(self) -> np.ndarray:
        """Whole-day shifts for the null: clear of every window, so a shifted trade never re-reads its own path."""
        spacing = self._ohlc.index.to_series().diff().median()
        longest = max(int(self.horizons.max()), int(self.post_exit_offsets.max()))
        min_days = int(np.ceil(longest * spacing / pd.Timedelta(days=1))) + 1
        candidates = np.arange(min_days, self._span_days - min_days + 1)
        if candidates.size == 0:
            raise ValueError(f"The sample spans {self._span_days} day(s), too short for a circular-shift baseline that "
                             f"clears {min_days} day(s) of windows on each side.")
        if candidates.size < 20:
            warnings.warn(f"ExcursionLens: only {candidates.size} distinct day shifts fit in the sample; the random "
                          "baselines are coarse.")
        rng = np.random.default_rng(self.config.seed)
        return rng.choice(candidates, self.config.n_null, replace=candidates.size < self.config.n_null)

    def _shifted(self, positions: np.ndarray, days: int) -> np.ndarray:
        """Bar positions of ``positions`` moved ``days`` days on the wall clock, wrapping around; -1 where no bar."""
        wall = self._wall
        origin = wall[0].normalize()
        moved = origin + (wall[positions] - origin + pd.Timedelta(days=int(days))) % pd.Timedelta(days=self._span_days)
        return self._wall_lookup.reindex(moved).fillna(-1).to_numpy(dtype=np.int64)

    # ------------------------------------------------------------------
    # Entry quality (exit-agnostic, fixed windows, circular-shift null)
    # ------------------------------------------------------------------

    @cached_property
    def horizons(self) -> np.ndarray:
        if self.config.horizons:
            return np.unique(np.asarray(self.config.horizons, dtype=np.int64))
        median = max(int(np.median(self._holds)), 1)
        return np.unique(np.maximum(np.round(median * np.array([0.25, 0.5, 1, 2, 4])), 1).astype(np.int64))

    @cached_property
    def _entry_targets(self) -> tuple[np.ndarray, np.ndarray]:
        """Every bar an entry statistic is ever read at (real entries and all their shifted copies), and the column
        each bar occupies; only those bars are scanned."""
        e_idx = self.trades["e_idx"].to_numpy()
        bars = np.concatenate([e_idx] + [self._shifted(e_idx, days) for days in self._shift_days])
        targets = np.unique(bars[bars >= 0])
        column = np.full(len(self._ohlc), -1, dtype=np.int64)
        column[targets] = np.arange(targets.size)
        return targets, column

    @cached_property
    def _bar_excursions(self) -> tuple[np.ndarray, np.ndarray]:
        o = self._ohlc
        targets, _ = self._entry_targets
        win = _window_excursions(targets, self._ref_price, o["high"].to_numpy(), o["low"].to_numpy(), self._spr,
                                 self._atr_ref, self._seg, self._offset, self.horizons)
        fp = _first_passage(targets, self._ref_price, o["high"].to_numpy(), o["low"].to_numpy(), self._spr,
                            self._atr_ref, self._seg, self._offset, np.asarray(self.config.barriers_atr, dtype=float),
                            self.config.max_leg_bars)
        return win, fp

    def _entry_stats(self, bars: np.ndarray, side: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean MFE, MAE (ATR) per horizon and P(+k first) per barrier for entries at ``bars`` (-1 = no bar)."""
        win, fp = self._bar_excursions
        _, column = self._entry_targets
        valid = bars >= 0
        cols = column[np.where(valid, bars, 0)]
        is_long = side > 0
        mfe = np.where(is_long, win[0][:, cols], win[2][:, cols])
        mae = np.where(is_long, win[1][:, cols], win[3][:, cols])
        hits = np.where(is_long, fp[0][:, cols], fp[1][:, cols])
        for block in (mfe, mae, hits):
            block[:, ~valid] = np.nan
        return _nanmean(mfe), _nanmean(mae), _nanmean(hits)

    @cached_property
    def _entry_null(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        side, e_idx = self.trades["side"].to_numpy(), self.trades["e_idx"].to_numpy()
        draws = [self._entry_stats(self._shifted(e_idx, days), side) for days in self._shift_days]
        return tuple(np.stack(part) for part in zip(*draws))

    @cached_property
    def _entry_real(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._entry_stats(self.trades["e_idx"].to_numpy(), self.trades["side"].to_numpy())

    def entry_quality(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(by_horizon, by_barrier): real statistic, shifted-schedule 2.5/50/97.5 percentiles (point-wise) and the real
        one's percentile. Verdicts use the family-wise test in ``significance()``."""
        side = self.trades["side"].to_numpy()
        mfe, mae, hits = self._entry_real
        null_mfe, null_mae, null_hits = self._entry_null
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio, null_ratio = mfe / mae, null_mfe / null_mae
        by_horizon = pd.DataFrame({"horizon_bars": self.horizons, "mfe_atr": mfe, "mae_atr": mae,
                                   "mfe_mae_ratio": ratio})
        by_horizon = by_horizon.join(self._band(ratio, null_ratio))
        by_barrier = pd.DataFrame({"barrier_atr": self.config.barriers_atr, "p_up_first": hits})
        by_barrier = by_barrier.join(self._band(hits, null_hits))
        fp = self._bar_excursions[1]
        cols = self._entry_targets[1][self.trades["e_idx"].to_numpy()]
        by_barrier["resolved_share"] = np.mean(np.isfinite(np.where(side > 0, fp[0][:, cols], fp[1][:, cols])), axis=1)
        return by_horizon, by_barrier

    @staticmethod
    def _band(real: np.ndarray, null: np.ndarray) -> pd.DataFrame:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN cell gives a NaN band, not a warning
            lo, med, hi = np.nanpercentile(null, [2.5, 50, 97.5], axis=0)
        return pd.DataFrame({"random_lo": lo, "random_median": med, "random_hi": hi,
                             "percentile_vs_random": np.where(np.isfinite(real), _midrank(real, null), np.nan)})

    # ------------------------------------------------------------------
    # Exit quality (entries fixed)
    #   timing: your exit bars vs random exit bars drawn from your own holding times (same entries)
    #   after-exit drift: did the trade's direction still have edge after the exit? vs the same exits on the
    #   circular-shifted schedule (a direction with no edge left)
    #   Both are measured from the exit BAR's fill price, so fills (slippage, intrabar stops) never read as timing.
    # ------------------------------------------------------------------

    @cached_property
    def post_exit_offsets(self) -> np.ndarray:
        w = self.config.post_exit_bars or max(int(np.median(self._holds)), 1)
        return np.unique(np.maximum(np.round(w * np.array([0.25, 0.5, 1, 1.5, 2])), 1).astype(np.int64))

    @property
    def fixed_hold(self) -> bool:
        """True when every trade held the same number of bars: random-exit timing then equals the real exits."""
        return np.unique(self._holds).size == 1

    def _drift_after(self, bars: np.ndarray, side: np.ndarray) -> np.ndarray:
        """Mean move in ``side``'s favour ``d`` bars after ``bars`` (-1 = no bar), from each bar's exit fill, in that
        bar's ATR. Real and shifted exits use the same fill, so they always cover the same bars."""
        n = len(self._ohlc)
        close = self._ohlc["close"].to_numpy()
        valid = bars >= 0
        safe = np.where(valid, bars, 0)
        ref_price = self._fill_at(bars, side)
        atr = self._atr_ref[safe]
        usable = valid & np.isfinite(atr) & (atr > 0)
        out = np.empty(len(self.post_exit_offsets))
        for k, d in enumerate(self.post_exit_offsets):
            later = safe + d
            ok = usable & (later < n) & (later <= self._seg_end[safe])
            later = np.minimum(later, n - 1)
            move = np.where(side > 0, close[later] - ref_price, ref_price - (close[later] + self._spr[later]))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[k] = _nanmean(np.where(ok, move / atr, np.nan)[None, :])[0]
        return out

    def _random_exit_bars(self, rng: np.random.Generator) -> np.ndarray:
        """Real entries, holding times drawn from the real ones. Like real trades they may run over breaks."""
        e_idx = self.trades["e_idx"].to_numpy()
        hold = rng.choice(self._holds, len(e_idx))
        return np.minimum(e_idx + hold, len(self._ohlc) - 1)

    def _realized_at_bars(self, bars: np.ndarray) -> float:
        """Mean result per trade (entry ATR) if every trade had exited at the fill price of ``bars``."""
        t = self.trades
        side, p0, atr = t["side"].to_numpy(), t["entry_price"].to_numpy(), t["atr_entry"].to_numpy()
        return float(_nanmean((side * (self._fill_at(bars, side) - p0) / atr)[None, :])[0])

    @cached_property
    def _exit_null(self) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(self.config.seed + 1)
        side, x_idx = self.trades["side"].to_numpy(), self.trades["x_idx"].to_numpy()
        realized, drift = [], []
        for days in self._shift_days:
            realized.append(self._realized_at_bars(self._random_exit_bars(rng)))
            drift.append(self._drift_after(self._shifted(x_idx, days), side))
        return np.asarray(realized), np.stack(drift)

    @cached_property
    def _drift_real(self) -> np.ndarray:
        return self._drift_after(self.trades["x_idx"].to_numpy(), self.trades["side"].to_numpy())

    def exit_quality(self) -> tuple[pd.Series, pd.DataFrame]:
        """(summary, after_exit_drift). Summary means are per trade in entry ATR; drift is in ATR at the exit bar.

        ``realized_atr`` is what the trades banked. The timing test compares ``realized_at_exit_bar_atr`` (the same
        exits at their bars' fill price) with random exits priced the same way; ``fill_effect_atr`` is the
        difference, i.e. what slippage or intrabar fills added or cost.
        """
        t = self.trades
        null_realized, null_drift = self._exit_null
        real_drift = self._drift_real
        realized_mean = t["realized_atr"].mean()
        at_bar = self._realized_at_bars(t["x_idx"].to_numpy())
        summary = pd.Series({
            "trades": len(t), "median_hold_bars": float(np.median(self._holds)), "fixed_hold": float(self.fixed_hold),
            "realized_atr": realized_mean, "realized_at_exit_bar_atr": at_bar,
            "fill_effect_atr": realized_mean - at_bar,
            "mfe_atr": t["mfe_atr"].mean(), "giveback_atr": t["giveback_atr"].mean(),
            "extension_atr": t["extension_atr"].mean(), "potential_atr": t["potential_atr"].mean(),
            "missed_atr": t["missed_atr"].mean(), "trailing_atr": t["trailing_atr"].mean(),
            "median_capture": t["capture"].median(), "median_potential_capture": t["potential_capture"].median(),
            "random_exit_lo": np.percentile(null_realized, 2.5), "random_exit_median": np.median(null_realized),
            "random_exit_hi": np.percentile(null_realized, 97.5),
            "exit_percentile_vs_random": float(_midrank(np.array([at_bar]), null_realized[:, None])[0]),
        })
        drift = pd.DataFrame({"bars_after_exit": self.post_exit_offsets, "drift_atr": real_drift})
        return summary, drift.join(self._band(real_drift, null_drift))

    # ------------------------------------------------------------------
    # Family-wise significance
    # ------------------------------------------------------------------

    def significance(self) -> pd.DataFrame:
        """One family-wise test per family of cells against the circular-shift null (max-z, Westfall-Young).

        p_beat: the real curve sits above the null somewhere in the family more than chance allows. p_worse: below.
        A verdict fires at p <= 0.025 per side (two-sided 5%).
        """
        mfe, mae, hits = self._entry_real
        null_mfe, null_mae, null_hits = self._entry_null
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio, null_ratio = mfe / mae, null_mfe / null_mae
        families = {"entry_mfe_mae": (ratio, null_ratio), "entry_first_passage": (hits, null_hits),
                    "after_exit_drift": (self._drift_real, self._exit_null[1])}
        rows = {name: dict(zip(("p_beat", "p_worse"), _familywise(real, null)), cells=int(np.size(real)))
                for name, (real, null) in families.items()}
        return pd.DataFrame(rows).T[["cells", "p_beat", "p_worse"]].astype(float)

    # ------------------------------------------------------------------
    # Stop analysis (MAE-based, walk-forward)
    # ------------------------------------------------------------------

    def _stop_gain(self, rows: np.ndarray, realized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Per stop level over ``rows``: mean result, gain vs the SAME trades' own exits, and simulable share."""
        out = self._stop_out_atr[rows]
        valid = np.isfinite(out)
        with_stop = _column_nanmean(out)
        own = _column_nanmean(np.where(valid, realized[rows][:, None], np.nan))
        coverage = valid.mean(axis=0) if len(rows) else np.zeros(out.shape[1])
        return with_stop, with_stop - own, coverage

    @staticmethod
    def _pick(gain: np.ndarray, coverage: np.ndarray) -> int | None:
        candidates = (coverage >= _MIN_STOP_COVERAGE) & np.isfinite(gain)
        return int(np.argmax(np.where(candidates, gain, -np.inf))) if candidates.any() else None

    def stop_analysis(self) -> dict:
        """Recovery curve, in-sample stop curve, walk-forward stop choice and stop-fill slippage.

        Stop levels wider than a trade's own stop cannot be simulated for trades that hit that stop. Those cells are
        left out: every level is compared with the same trades' own exits, and a level must cover at least half the
        trades to be chosen. ``best_stop_improves`` says whether the best level actually beats the trades' own exits.
        Walk-forward: trades still open when a test fold starts are purged from its training set, and a fold whose
        chosen level covers less than half its trades is reported as not testable (NaN improvement).
        """
        t = self.trades
        grid = np.asarray(self.config.stop_grid_atr, dtype=float)
        realized, mae = t["realized_atr"].to_numpy(), t["mae_atr"].to_numpy()
        recovery = pd.DataFrame({"mae_at_least_atr": grid})
        recovery["trades_reaching"] = [(mae >= x).sum() for x in grid]
        recovery["p_still_wins"] = [np.mean(realized[mae >= x] > 0) if (mae >= x).any() else np.nan for x in grid]
        recovery["mean_final_atr"] = [realized[mae >= x].mean() if (mae >= x).any() else np.nan for x in grid]

        everyone = np.arange(len(t))
        with_stop, gain, coverage = self._stop_gain(everyone, realized)
        valid = np.isfinite(self._stop_out_atr)
        stopped_share = self._stop_hit.sum(axis=0) / np.maximum(valid.sum(axis=0), 1)
        curve = pd.DataFrame({"stop_atr": grid, "mean_result_atr": with_stop,
                              "same_trades_own_exit_atr": with_stop - gain, "gain_atr": gain,
                              "share_stopped": stopped_share, "simulable_share": coverage})
        baseline = realized.mean()

        entry_time, exit_time = t["entry_time"].to_numpy(), t["exit_time"].to_numpy()
        n_folds = min(self.config.n_stop_folds, len(t))
        folds = np.array_split(everyone, n_folds) if n_folds >= 2 else []
        walk = []
        for f in range(1, len(folds)):
            earlier, test = np.concatenate(folds[:f]), folds[f]
            train = earlier[exit_time[earlier] < entry_time[test[0]]]  # purge trades still open at the test start
            row = {"fold": f, "train_trades": len(train), "purged": len(earlier) - len(train), "test_trades": len(test),
                   "chosen_stop_atr": np.nan, "test_coverage": np.nan}
            _, train_gain, train_coverage = self._stop_gain(train, realized) if len(train) else (None, None, None)
            best = self._pick(train_gain, train_coverage) if len(train) else None
            if best is not None and train_gain[best] > 0:
                test_with, test_gain, test_coverage = self._stop_gain(test, realized)
                row.update({"chosen_stop_atr": grid[best], "test_coverage": test_coverage[best]})
                if test_coverage[best] >= _MIN_STOP_COVERAGE:
                    row.update({"test_with_stop_atr": test_with[best], "improvement_atr": test_gain[best],
                                "test_without_atr": test_with[best] - test_gain[best]})
                else:
                    row.update({"test_with_stop_atr": np.nan, "test_without_atr": np.nan, "improvement_atr": np.nan})
            else:
                own = realized[test].mean()
                row.update({"test_with_stop_atr": own, "test_without_atr": own, "improvement_atr": 0.0})
            walk.append(row)
        walk = pd.DataFrame(walk, columns=["fold", "train_trades", "purged", "test_trades", "chosen_stop_atr",
                                           "test_coverage", "test_with_stop_atr", "test_without_atr",
                                           "improvement_atr"])

        best = self._pick(gain, coverage)
        result = {"recovery": recovery, "curve": curve, "baseline_atr": baseline, "walk_forward": walk,
                  "best_stop_atr": np.nan, "best_stop_gain_atr": np.nan, "best_stop_improves": False,
                  "stop_fill_mean_x_nominal": np.nan, "stop_fill_worst_x_nominal": np.nan}
        if best is None:
            return result
        hit = self._stop_hit[:, best]
        slippage = -self._stop_out_atr[hit, best] / grid[best] if hit.any() else np.array([np.nan])
        result.update({"best_stop_atr": grid[best], "best_stop_gain_atr": gain[best],
                       "best_stop_improves": bool(gain[best] > 0),
                       "stop_fill_mean_x_nominal": float(np.mean(slippage)),
                       "stop_fill_worst_x_nominal": float(np.max(slippage))})
        return result

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
        sig = self.significance()
        lines = []

        ratio = sig.loc["entry_mfe_mae"]
        cells = f"family-wise across {int(ratio['cells'])} horizons"
        if ratio["p_beat"] <= _SIGNIFICANCE:
            row = by_h.loc[by_h["percentile_vs_random"].idxmax()]
            lines.append(f"ENTRIES (MFE/MAE over fixed windows) beat the shifted schedule (p {ratio['p_beat']:.3f}, "
                         f"{cells}); strongest at {int(row['horizon_bars'])} bars: {row['mfe_mae_ratio']:.2f} vs "
                         f"random {row['random_lo']:.2f}-{row['random_hi']:.2f}.")
        elif ratio["p_worse"] <= _SIGNIFICANCE:
            row = by_h.loc[by_h["percentile_vs_random"].idxmin()]
            lines.append(f"ENTRIES (MFE/MAE over fixed windows) are WORSE than the shifted schedule (p "
                         f"{ratio['p_worse']:.3f}, {cells}); weakest at {int(row['horizon_bars'])} bars: "
                         f"{row['mfe_mae_ratio']:.2f} vs random {row['random_lo']:.2f}-{row['random_hi']:.2f}.")
        else:
            lines.append(f"ENTRIES (MFE/MAE over fixed windows) are indistinguishable from the shifted schedule "
                         f"(p beat {ratio['p_beat']:.2f}, p worse {ratio['p_worse']:.2f}, {cells}).")

        passage = sig.loc["entry_first_passage"]
        if passage["p_beat"] <= _SIGNIFICANCE:
            row = by_k.loc[by_k["percentile_vs_random"].idxmax()]
            lines.append(f"ENTRIES (first passage): +{row['barrier_atr']:g} ATR came before "
                         f"-{row['barrier_atr']:g} ATR {row['p_up_first']:.0%} of the time, above the shifted schedule "
                         f"({row['random_median']:.0%}); family-wise p {passage['p_beat']:.3f}.")
        elif passage["p_worse"] <= _SIGNIFICANCE:
            row = by_k.loc[by_k["percentile_vs_random"].idxmin()]
            lines.append(f"ENTRIES (first passage) point the WRONG way: +{row['barrier_atr']:g} ATR came before "
                         f"-{row['barrier_atr']:g} ATR only {row['p_up_first']:.0%} of the time, below the shifted "
                         f"schedule ({row['random_median']:.0%}); family-wise p {passage['p_worse']:.3f}.")
        else:
            lines.append(f"ENTRIES (first passage): no different from the shifted schedule at any barrier (p beat "
                         f"{passage['p_beat']:.2f}, p worse {passage['p_worse']:.2f}).")

        lines.append(f"EXITS bank {summary['realized_atr']:+.2f} ATR/trade; the move offered "
                     f"{summary['potential_atr']:.2f}. Missed {summary['missed_atr']:.2f} = "
                     f"{summary['giveback_atr']:.2f} handed back in the trade + {summary['extension_atr']:.2f} "
                     "after exit.")
        lines.append(f"A trailing stop at the move-end distance would have banked {summary['trailing_atr']:+.2f} "
                     f"ATR/trade; random exits with your holding times bank {summary['random_exit_median']:+.2f} at "
                     f"bar prices (yours {summary['realized_at_exit_bar_atr']:+.2f}; fills added "
                     f"{summary['fill_effect_atr']:+.2f}).")
        pct = summary["exit_percentile_vs_random"]
        if summary["fixed_hold"]:
            lines.append(f"Exits are a fixed holding time ({int(summary['median_hold_bars'])} bars), so the "
                         "random-exit timing test does not apply.")
        else:
            timing = "ADD value" if pct >= 0.975 else "DESTROY value" if pct <= 0.025 else "are no better than random"
            lines.append(f"Exit timing {timing} vs random exits with your holding times (percentile {pct:.0%}).")

        after = sig.loc["after_exit_drift"]
        late = drift.iloc[min(2, len(drift) - 1)]
        band = (f"{late['drift_atr']:+.2f} ATR over {int(late['bars_after_exit'])} bars; shifted schedule "
                f"{late['random_lo']:+.2f} to {late['random_hi']:+.2f}")
        if after["p_beat"] <= _SIGNIFICANCE:
            lines.append(f"After your exits price kept going your way ({band}; family-wise p {after['p_beat']:.3f}). "
                         "You exit EARLY.")
        elif after["p_worse"] <= _SIGNIFICANCE:
            lines.append(f"After your exits price turned against the trade ({band}; family-wise p "
                         f"{after['p_worse']:.3f}). Exits are well timed.")
        else:
            lines.append(f"After your exits the direction had no edge left ({band}).")

        walk = stops["walk_forward"]
        tested = walk["improvement_atr"].notna()
        untestable = f" ({int((~tested).sum())} not testable)" if (~tested).any() else ""
        if stops["best_stop_improves"]:
            improved = int((walk["improvement_atr"] > 0).sum())
            grid = self.config.stop_grid_atr
            edge = (" It sits at the EDGE of the tested grid, so the true optimum may lie outside it."
                    if stops["best_stop_atr"] in (min(grid), max(grid)) else "")
            lines.append(f"STOP: best in-sample stop {stops['best_stop_atr']:g} ATR "
                         f"({stops['best_stop_gain_atr']:+.2f} ATR/trade vs the same trades' own exits); chosen on "
                         f"earlier trades it improved {improved}/{int(tested.sum())} later folds{untestable}.{edge} "
                         f"Stop fills averaged {stops['stop_fill_mean_x_nominal']:.2f}x the nominal stop (worst "
                         f"{stops['stop_fill_worst_x_nominal']:.2f}x: gaps and spread at the fill).")
        elif np.isfinite(stops["best_stop_atr"]):
            lines.append(f"STOP: no tested stop beats your own exits (the least bad, {stops['best_stop_atr']:g} ATR, "
                         f"gives {stops['best_stop_gain_atr']:+.2f} ATR/trade vs the same trades' own exits).")
        else:
            lines.append("STOP: no grid level can be simulated on at least half the trades (they hit their own, "
                         "tighter stops), so no stop is suggested.")
        return lines

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _band_traces(self, x, real, lo, hi, name: str, colour: str, row: int, col: int) -> list[dict]:
        return [
            {"trace": go.Scatter(x=x, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
             "row": row, "col": col},
            {"trace": go.Scatter(x=x, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(139,148,158,0.25)", name="shifted-schedule 95% band (point-wise)",
                                 showlegend=False, hoverinfo="skip"), "row": row, "col": col},
            {"trace": go.Scatter(x=x, y=real, mode="lines+markers", line=dict(color=colour, width=2), name=name,
                                 showlegend=False), "row": row, "col": col},
        ]

    def _table(self, header: list[str], columns: list[list], colour: str) -> go.Table:
        return go.Table(header=dict(values=header, fill_color=self._PANEL, line_color=self._BORDER, align="left",
                                    font=dict(color=colour, size=11, family="'Courier New', monospace")),
                        cells=dict(values=columns, fill_color=self._BG, line_color=self._BORDER, align="left",
                                   height=20, font=dict(color=self._TEXT, size=10, family="'Courier New', monospace")))

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
                "③ Recovery: P(trade still wins | MAE reached x ATR)",
                "④ Stop candidates: mean result vs the same trades' own exits (ATR)",
                "⑤ Entry: MFE/MAE over fixed windows vs the shifted schedule",
                "⑥ Entry: P(+k ATR before -k ATR) vs the shifted schedule",
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
                                 line=dict(color=self._ORANGE, width=2), showlegend=False,
                                 customdata=curve["simulable_share"],
                                 hovertemplate="stop %{x} ATR: %{y:.2f} (simulable on %{customdata:.0%} of trades)"
                                               "<extra></extra>"), row=2, col=2)
        fig.add_trace(go.Scatter(x=curve["stop_atr"], y=curve["same_trades_own_exit_atr"], mode="lines",
                                 line=dict(color=self._SUB, dash="dash"), showlegend=False,
                                 hovertemplate="same trades, own exits: %{y:.2f}<extra></extra>"), row=2, col=2)
        if stops["best_stop_improves"]:
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
            "train_trades": "train", "test_trades": "test", "chosen_stop_atr": "stop_atr", "test_coverage": "cover",
            "test_with_stop_atr": "with", "test_without_atr": "without", "improvement_atr": "gain"})
        fig.add_trace(self._table(list(walk.columns), [walk[c] for c in walk.columns], self._ORANGE), row=7, col=1)
        fig.add_trace(self._table(["Verdicts (family-wise vs random baselines)"], [self.verdicts()], self._GREEN),
                      row=7, col=2)

        axis_titles = {(1, 1): ("MAE (ATR)", "MFE (ATR)"), (1, 2): ("MAE (ATR)", "final result (ATR)"),
                       (2, 1): ("MAE reached (ATR)", "share still winning"),
                       (2, 2): ("stop (ATR)", "mean result (ATR)"),
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
