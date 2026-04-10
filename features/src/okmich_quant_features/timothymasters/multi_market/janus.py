"""
JANUS multi-market indicators — Gary Anderson's relative-strength system.

Python port of JANUS.CPP (~1,169 lines) from:
    Timothy Masters, "Statistically Sound Indicators For Financial
    Market Prediction", Apress, 2013.

All indicators share a common computation pipeline and are computed as a single class.
Functional wrappers are provided for one-off usage; for batch use, create a single ``Janus`` object and extract multiple outputs.

Indicators
----------
market_index           Cumulative median-return market index
rs / rs_fractile       Relative strength (offensive/defensive) and its fractile
rss / rss_change       Relative-strength spread and its first difference
dom / doe              Dominance / Dominance of Equity accumulation
dom_index_equity       Cumulative DOM index (all-market aggregate)
rm / rm_fractile       Relative momentum (DOM-based RS) and its fractile
rs_leader/laggard      Equity curves of top/bottom RS performers
rs_ps                  Performance spread (leader − laggard)
rm_leader/laggard      Same pattern for RM-based leader/laggard
oos_avg                Average out-of-sample return
cma_oos                Adaptive CMA out-of-sample equity
leader_cma_oos         Leader adaptive CMA out-of-sample equity
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Janus class
# ---------------------------------------------------------------------------


class Janus:
    """Compute all JANUS multi-market indicators in a single pass.

    Parameters
    ----------
    closes : list[np.ndarray]
        Close price arrays for N >= 2 markets, all same length.
    lookback : int
        Rolling window for RS/RM computation (default 252).
    spread_tail : float
        Fraction of markets used for RSS top-k / bottom-k (default 0.1).
    min_cma : int
        Minimum CMA lookback to search (default 20).
    max_cma : int
        Maximum CMA lookback to search (default 60).
    """

    def __init__(self, closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                 max_cma: int = 60):
        # --- validation ---
        if len(closes) < 2:
            raise ValueError(f"At least 2 markets required; got {len(closes)}.")
        n = len(closes[0])
        for i, c in enumerate(closes):
            if len(c) != n:
                raise ValueError(
                    f"All arrays must have the same length; "
                    f"closes[{i}] has {len(c)}, expected {n}."
                )
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2; got {lookback}.")

        self._n_bars = n
        self._n_markets = len(closes)
        self._lookback = lookback
        self._spread_tail = spread_tail
        self._min_cma = min_cma
        self._max_cma = max_cma

        # --- Phase 1: log returns + market index ---
        self._log_returns = np.empty((self._n_markets, n - 1), dtype=np.float64)
        for m in range(self._n_markets):
            c = np.asarray(closes[m], dtype=np.float64)
            self._log_returns[m] = np.log(c[1:] / c[:-1])

        # n_returns = n_bars - 1; output arrays are length n_bars with first
        # lookback bars as NaN.
        self._n_returns = n - 1

        # Market index: cumulative sum of median returns across markets
        median_returns = np.median(self._log_returns, axis=0)  # (n_returns,)
        self._market_index_arr = np.full(n, np.nan, dtype=np.float64)
        self._market_index_arr[1:] = np.cumsum(median_returns)
        self._market_index_arr[0] = 0.0

        # --- Phase 2–10: compute all derived indicators ---
        self._compute_all()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute_all(self):
        n = self._n_bars
        nm = self._n_markets
        lb = self._lookback

        # ---- Phase 2: RS (lag=0) ----
        rs_raw, rs_frac = self._compute_rs(lag=0)
        self._rs_arr = rs_raw          # (n_bars, n_markets)
        self._rs_frac_arr = rs_frac    # (n_bars, n_markets)

        # ---- Phase 3: RS (lag=1) — for RSS change ----
        rs_lagged, _ = self._compute_rs(lag=1)
        self._rs_lagged_arr = rs_lagged

        # ---- Phase 4: RSS ----
        self._compute_rss()

        # ---- Phase 5: DOM / DOE ----
        self._compute_dom_doe()

        # ---- Phase 6: RM (lag=0) ----
        rm_raw, rm_frac = self._compute_rm(lag=0)
        self._rm_arr = rm_raw
        self._rm_frac_arr = rm_frac

        # ---- Phase 7: RM (lag=1) — not strictly needed but for completeness
        rm_lagged, _ = self._compute_rm(lag=1)
        self._rm_lagged_arr = rm_lagged

        # ---- Phase 8: RS performance spread ----
        self._compute_rs_ps()

        # ---- Phase 9: RM performance spread ----
        self._compute_rm_ps()

        # ---- Phase 10: CMA ----
        self._compute_cma()

    def _compute_rs(self, lag: int = 0):
        """Compute RS and RS-fractile for all markets.

        Returns
        -------
        rs_raw : (n_bars, n_markets) — clipped to [-200, 200]
        rs_frac : (n_bars, n_markets) — fractile in [0, 1]
        """
        n = self._n_bars
        nm = self._n_markets
        lb = self._lookback
        lr = self._log_returns  # (n_markets, n_returns)

        rs_raw = np.full((n, nm), np.nan, dtype=np.float64)
        rs_frac = np.full((n, nm), np.nan, dtype=np.float64)

        # We need at least lookback+lag returns to start computing
        start_ret = lb + lag  # first return index where we can compute

        for iret in range(start_ret, self._n_returns + 1):
            # iret corresponds to bar index iret (return iret-1 is the latest)
            # Window of lookback returns ending at iret-1-lag
            end = iret - lag
            beg = end - lb

            # Market-index returns in the window (median of all markets per bar)
            index_window = np.median(lr[:, beg:end], axis=0)  # (lb,)

            # Median of the index window
            med = np.median(index_window)

            # Offensive / defensive split of the index
            off_mask = index_window >= med
            def_mask = ~off_mask

            index_off = np.sum(index_window[off_mask] - med)
            index_def = np.sum(index_window[def_mask] - med)

            # Guard against zero
            if index_off < 1e-30:
                index_off = 1e-30
            if index_def > -1e-30:
                index_def = -1e-30

            # Vectorized across all markets
            all_windows = lr[:, beg:end]  # (nm, lb)
            m_off = np.sum(all_windows[:, off_mask] - med, axis=1)
            m_def = np.sum(all_windows[:, def_mask] - med, axis=1)
            rs_vals = 70.710678 * (m_off / index_off - m_def / index_def)
            np.clip(rs_vals, -200.0, 200.0, out=rs_vals)
            rs_raw[iret] = rs_vals

            # Vectorized fractile
            sorted_vals = np.sort(rs_vals)
            counts = np.searchsorted(sorted_vals, rs_vals, side='right')
            rs_frac[iret] = counts / nm

        return rs_raw, rs_frac

    def _compute_rss(self):
        """Compute RSS (relative-strength spread) and its change."""
        n = self._n_bars
        nm = self._n_markets
        lb = self._lookback

        self._rss_arr = np.full(n, np.nan, dtype=np.float64)
        self._rss_change_arr = np.full(n, np.nan, dtype=np.float64)

        # Number of markets in each tail
        k = max(1, int(nm * self._spread_tail + 0.5))

        for i in range(n):
            if np.isnan(self._rs_arr[i, 0]):
                continue

            row = self._rs_arr[i]
            sorted_rs = np.sort(row)

            # Top-k average minus bottom-k average
            top_k = sorted_rs[-k:]
            bot_k = sorted_rs[:k]
            self._rss_arr[i] = np.mean(top_k) - np.mean(bot_k)

        # RSS change: current RS spread vs lagged RS spread
        for i in range(n):
            if np.isnan(self._rs_arr[i, 0]) or np.isnan(self._rs_lagged_arr[i, 0]):
                continue

            row_cur = self._rs_arr[i]
            row_lag = self._rs_lagged_arr[i]

            sorted_cur = np.sort(row_cur)
            sorted_lag = np.sort(row_lag)

            top_cur = np.mean(sorted_cur[-k:]) - np.mean(sorted_cur[:k])
            top_lag = np.mean(sorted_lag[-k:]) - np.mean(sorted_lag[:k])

            self._rss_change_arr[i] = top_cur - top_lag

    def _compute_dom_doe(self):
        """Compute DOM (dominance) and DOE (dominance of equity)."""
        n = self._n_bars
        nm = self._n_markets
        lr = self._log_returns

        # DOM and DOE per market + a global index version
        # Market index at position -1 conceptually, we store it separately
        self._dom_arr = np.full((n, nm), np.nan, dtype=np.float64)
        self._doe_arr = np.full((n, nm), np.nan, dtype=np.float64)
        self._dom_index_arr = np.full(n, np.nan, dtype=np.float64)
        self._doe_index_arr = np.full(n, np.nan, dtype=np.float64)

        # Running sums — one per market + index
        dom_sum = np.zeros(nm, dtype=np.float64)
        doe_sum = np.zeros(nm, dtype=np.float64)
        dom_index_sum = 0.0
        doe_index_sum = 0.0

        # We can compute DOM/DOE once rss_change is valid
        for i in range(n):
            if np.isnan(self._rss_change_arr[i]):
                continue

            # Returns at bar i (return index i-1)
            if i == 0:
                continue
            ret_idx = i - 1

            rss_chg = self._rss_change_arr[i]
            median_ret = np.median(lr[:, ret_idx])

            if rss_chg > 0:
                # Expanding: accumulate into DOM
                for m in range(nm):
                    dom_sum[m] += lr[m, ret_idx]
                dom_index_sum += median_ret
            elif rss_chg < 0:
                # Contracting: accumulate into DOE
                for m in range(nm):
                    doe_sum[m] += lr[m, ret_idx]
                doe_index_sum += median_ret

            for m in range(nm):
                self._dom_arr[i, m] = dom_sum[m]
                self._doe_arr[i, m] = doe_sum[m]
            self._dom_index_arr[i] = dom_index_sum
            self._doe_index_arr[i] = doe_index_sum

    def _compute_rm(self, lag: int = 0):
        """Compute RM (relative momentum) using DOM changes instead of raw returns.

        Before DOM is valid, falls back to raw returns.
        """
        n = self._n_bars
        nm = self._n_markets
        lb = self._lookback

        rm_raw = np.full((n, nm), np.nan, dtype=np.float64)
        rm_frac = np.full((n, nm), np.nan, dtype=np.float64)

        # Build "momentum returns" — diff of DOM where available, else raw log returns
        mom_returns = self._log_returns.copy()  # (nm, n_returns)
        # DOM changes: dom[i] - dom[i-1] where both are valid
        for i in range(1, n):
            ret_idx = i - 1
            if ret_idx < self._n_returns and not np.isnan(self._dom_arr[i, 0]) and not np.isnan(self._dom_arr[i - 1, 0]):
                mom_returns[:, ret_idx] = self._dom_arr[i, :] - self._dom_arr[i - 1, :]

        start_ret = lb + lag

        for iret in range(start_ret, self._n_returns + 1):
            end = iret - lag
            beg = end - lb

            window = mom_returns[:, beg:end]
            if np.any(np.isnan(window)):
                continue

            index_window = np.median(window, axis=0)
            med = np.median(index_window)

            off_mask = index_window >= med
            def_mask = ~off_mask

            index_off = np.sum(index_window[off_mask] - med)
            index_def = np.sum(index_window[def_mask] - med)

            if index_off < 1e-30:
                index_off = 1e-30
            if index_def > -1e-30:
                index_def = -1e-30

            # Vectorized across all markets
            m_off = np.sum(window[:, off_mask] - med, axis=1)
            m_def = np.sum(window[:, def_mask] - med, axis=1)
            rm_vals = 70.710678 * (m_off / index_off - m_def / index_def)
            np.clip(rm_vals, -300.0, 300.0, out=rm_vals)
            rm_raw[iret] = rm_vals

            sorted_vals = np.sort(rm_vals)
            counts = np.searchsorted(sorted_vals, rm_vals, side='right')
            rm_frac[iret] = counts / nm

        return rm_raw, rm_frac

    def _compute_rs_ps(self):
        """Compute RS-based performance spread indicators."""
        n = self._n_bars
        nm = self._n_markets
        lr = self._log_returns

        self._oos_avg_arr = np.full(n, np.nan, dtype=np.float64)
        self._rs_leader_arr = np.full(n, np.nan, dtype=np.float64)
        self._rs_laggard_arr = np.full(n, np.nan, dtype=np.float64)

        leader_sum = 0.0
        laggard_sum = 0.0
        avg_sum = 0.0
        started = False

        for i in range(1, n):
            ret_idx = i - 1
            # Need RS fractile from previous bar to select leader/laggard
            prev = i - 1
            if np.isnan(self._rs_frac_arr[prev, 0]):
                continue

            if not started:
                started = True

            frac = self._rs_frac_arr[prev]
            returns = lr[:, ret_idx]

            # Leader = highest fractile market, laggard = lowest
            leader_idx = np.argmax(frac)
            laggard_idx = np.argmin(frac)

            leader_sum += returns[leader_idx]
            laggard_sum += returns[laggard_idx]
            avg_sum += np.mean(returns)

            self._rs_leader_arr[i] = leader_sum
            self._rs_laggard_arr[i] = laggard_sum
            self._oos_avg_arr[i] = avg_sum

    def _compute_rm_ps(self):
        """Compute RM-based performance spread indicators."""
        n = self._n_bars
        nm = self._n_markets

        self._rm_leader_arr = np.full(n, np.nan, dtype=np.float64)
        self._rm_laggard_arr = np.full(n, np.nan, dtype=np.float64)

        leader_sum = 0.0
        laggard_sum = 0.0
        started = False

        for i in range(1, n):
            ret_idx = i - 1
            prev = i - 1
            if np.isnan(self._rm_frac_arr[prev, 0]):
                continue

            if not started:
                started = True

            frac = self._rm_frac_arr[prev]
            returns = self._log_returns[:, ret_idx]

            leader_idx = np.argmax(frac)
            laggard_idx = np.argmin(frac)

            leader_sum += returns[leader_idx]
            laggard_sum += returns[laggard_idx]

            self._rm_leader_arr[i] = leader_sum
            self._rm_laggard_arr[i] = laggard_sum

    def _compute_cma(self):
        """Compute adaptive CMA out-of-sample equity curves.

        Finds the best EMA lookback (min_cma to max_cma) by maximizing
        in-sample equity on dom_index, then uses that lookback for OOS equity.

        Uses incremental EMA tracking for O(n × k) instead of O(n² × k).
        """
        n = self._n_bars
        min_c = self._min_cma
        max_c = self._max_cma
        n_lookbacks = max_c - min_c + 1

        self._cma_oos_arr = np.full(n, np.nan, dtype=np.float64)
        self._leader_cma_oos_arr = np.full(n, np.nan, dtype=np.float64)

        dom_idx = self._dom_index_arr

        # Find first valid dom_index bar
        first_valid = -1
        for i in range(n):
            if not np.isnan(dom_idx[i]):
                first_valid = i
                break
        if first_valid < 0:
            return

        cma_start = first_valid + max_c + 1
        if cma_start >= n:
            return

        # Precompute median returns for each bar
        median_rets = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            if i - 1 < self._n_returns:
                median_rets[i] = np.median(self._log_returns[:, i - 1])

        # Incremental EMA and equity tracking for each lookback
        alphas = np.array([2.0 / (lb + 1.0) for lb in range(min_c, max_c + 1)])
        emas = np.full(n_lookbacks, dom_idx[first_valid], dtype=np.float64)
        equities = np.zeros(n_lookbacks, dtype=np.float64)

        # Walk forward from first_valid+1 to cma_start-1 to build up EMA/equity state
        for i in range(first_valid + 1, cma_start):
            if not np.isnan(dom_idx[i]):
                # Check signal before updating EMA
                for k in range(n_lookbacks):
                    if dom_idx[i - 1] > emas[k] and not np.isnan(median_rets[i]):
                        equities[k] += median_rets[i]
                    emas[k] = alphas[k] * dom_idx[i] + (1.0 - alphas[k]) * emas[k]

        oos_sum = 0.0
        leader_oos_sum = 0.0

        for i in range(cma_start, n):
            # Find best lookback by in-sample equity
            best_k = np.argmax(equities)
            best_ema = emas[best_k]

            # OOS return at bar i
            if not np.isnan(median_rets[i]) and dom_idx[i - 1] > best_ema:
                oos_sum += median_rets[i]
                # Leader OOS
                if not np.isnan(self._rs_frac_arr[i - 1, 0]):
                    leader_idx = np.argmax(self._rs_frac_arr[i - 1])
                    leader_oos_sum += self._log_returns[leader_idx, i - 1]
                else:
                    leader_oos_sum += median_rets[i]

            self._cma_oos_arr[i] = oos_sum
            self._leader_cma_oos_arr[i] = leader_oos_sum

            # Update incremental EMAs and equities for the next bar
            if not np.isnan(dom_idx[i]):
                for k in range(n_lookbacks):
                    if dom_idx[i - 1] > emas[k] and not np.isnan(median_rets[i]):
                        equities[k] += median_rets[i]
                    emas[k] = alphas[k] * dom_idx[i] + (1.0 - alphas[k]) * emas[k]

    # ------------------------------------------------------------------
    # Public output properties / methods
    # ------------------------------------------------------------------

    @property
    def n_bars(self) -> int:
        return self._n_bars

    @property
    def n_markets(self) -> int:
        return self._n_markets

    @property
    def market_index(self) -> np.ndarray:
        """Cumulative sum of median returns (length n_bars)."""
        return self._market_index_arr.copy()

    @property
    def dom_index_equity(self) -> np.ndarray:
        """Cumulative DOM index equity (length n_bars)."""
        return self._dom_index_arr.copy()

    def rs(self, market: int) -> np.ndarray:
        """Raw RS for a market, clipped to [-200, 200]."""
        return self._rs_arr[:, market].copy()

    def rs_fractile(self, market: int) -> np.ndarray:
        """RS fractile for a market, in [0, 1]."""
        return self._rs_frac_arr[:, market].copy()

    @property
    def rss(self) -> np.ndarray:
        """Relative-strength spread (width)."""
        return self._rss_arr.copy()

    @property
    def rss_change(self) -> np.ndarray:
        """First difference of RSS."""
        return self._rss_change_arr.copy()

    def dom(self, market: int | None = None) -> np.ndarray:
        """Cumulative dominance. market=None returns the index version."""
        if market is None:
            return self._dom_index_arr.copy()
        return self._dom_arr[:, market].copy()

    def doe(self, market: int | None = None) -> np.ndarray:
        """Cumulative dominance of equity. market=None returns the index version."""
        if market is None:
            return self._doe_index_arr.copy()
        return self._doe_arr[:, market].copy()

    def rm(self, market: int) -> np.ndarray:
        """Raw RM for a market, clipped to [-300, 300]."""
        return self._rm_arr[:, market].copy()

    def rm_fractile(self, market: int) -> np.ndarray:
        """RM fractile for a market, in [0, 1]."""
        return self._rm_frac_arr[:, market].copy()

    @property
    def oos_avg(self) -> np.ndarray:
        """Cumulative average OOS return."""
        return self._oos_avg_arr.copy()

    @property
    def rs_leader_equity(self) -> np.ndarray:
        """Cumulative equity of RS leader."""
        return self._rs_leader_arr.copy()

    @property
    def rs_laggard_equity(self) -> np.ndarray:
        """Cumulative equity of RS laggard."""
        return self._rs_laggard_arr.copy()

    @property
    def rs_ps(self) -> np.ndarray:
        """RS performance spread: leader − laggard equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rs_leader_arr) & ~np.isnan(self._rs_laggard_arr)
        out[valid] = self._rs_leader_arr[valid] - self._rs_laggard_arr[valid]
        return out

    @property
    def rs_leader_advantage(self) -> np.ndarray:
        """RS leader equity minus average equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rs_leader_arr) & ~np.isnan(self._oos_avg_arr)
        out[valid] = self._rs_leader_arr[valid] - self._oos_avg_arr[valid]
        return out

    @property
    def rs_laggard_advantage(self) -> np.ndarray:
        """RS laggard equity minus average equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rs_laggard_arr) & ~np.isnan(self._oos_avg_arr)
        out[valid] = self._rs_laggard_arr[valid] - self._oos_avg_arr[valid]
        return out

    @property
    def rm_leader_equity(self) -> np.ndarray:
        """Cumulative equity of RM leader."""
        return self._rm_leader_arr.copy()

    @property
    def rm_laggard_equity(self) -> np.ndarray:
        """Cumulative equity of RM laggard."""
        return self._rm_laggard_arr.copy()

    @property
    def rm_ps(self) -> np.ndarray:
        """RM performance spread: leader − laggard equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rm_leader_arr) & ~np.isnan(self._rm_laggard_arr)
        out[valid] = self._rm_leader_arr[valid] - self._rm_laggard_arr[valid]
        return out

    @property
    def rm_leader_advantage(self) -> np.ndarray:
        """RM leader equity minus average equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rm_leader_arr) & ~np.isnan(self._oos_avg_arr)
        out[valid] = self._rm_leader_arr[valid] - self._oos_avg_arr[valid]
        return out

    @property
    def rm_laggard_advantage(self) -> np.ndarray:
        """RM laggard equity minus average equity."""
        out = np.full(self._n_bars, np.nan, dtype=np.float64)
        valid = ~np.isnan(self._rm_laggard_arr) & ~np.isnan(self._oos_avg_arr)
        out[valid] = self._rm_laggard_arr[valid] - self._oos_avg_arr[valid]
        return out

    @property
    def cma_oos(self) -> np.ndarray:
        """Adaptive CMA out-of-sample equity."""
        return self._cma_oos_arr.copy()

    @property
    def leader_cma_oos(self) -> np.ndarray:
        """Leader adaptive CMA out-of-sample equity."""
        return self._leader_cma_oos_arr.copy()


# ---------------------------------------------------------------------------
# Functional wrappers
# ---------------------------------------------------------------------------

def _make_janus(closes, lookback=252, spread_tail=0.1, min_cma=20, max_cma=60):
    return Janus(closes, lookback=lookback, spread_tail=spread_tail,
                 min_cma=min_cma, max_cma=max_cma)


def janus_market_index(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                       max_cma: int = 60) -> np.ndarray:
    """Cumulative median-return market index."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).market_index


def janus_rs(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
             min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Raw RS for a single market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs(market)


def janus_rs_fractile(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
                      min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """RS fractile for a single market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_fractile(market)


def janus_delta_rs_fractile(closes: list[np.ndarray], market: int = 0, lookback: int = 252, delta_length: int = 20,
                            spread_tail: float = 0.1, min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Change in RS fractile over delta_length bars."""
    if delta_length < 1:
        raise ValueError(f"delta_length must be >= 1, got {delta_length}")
    frac = _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_fractile(market)
    out = np.full_like(frac, np.nan)
    out[delta_length:] = frac[delta_length:] - frac[:-delta_length]
    return out


def janus_rss(closes: list[np.ndarray], lookback: int = 252, smoothing: int = 0, spread_tail: float = 0.1,
              min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Relative-strength spread."""
    result = _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rss
    if smoothing > 0:
        result = _ema_smooth(result, smoothing)
    return result


def janus_delta_rss(closes: list[np.ndarray], lookback: int = 252, smoothing: int = 0, spread_tail: float = 0.1,
                    min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Change in RSS (rss_change)."""
    result = _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rss_change
    if smoothing > 0:
        result = _ema_smooth(result, smoothing)
    return result


def janus_dom(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
              min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Cumulative dominance for a market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).dom(market)


def janus_doe(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
              min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Cumulative dominance of equity for a market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).doe(market)


def janus_dom_index(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                    max_cma: int = 60) -> np.ndarray:
    """DOM index equity curve."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).dom_index_equity


def janus_rm(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
             min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Raw RM for a single market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm(market)


def janus_rm_fractile(closes: list[np.ndarray], market: int = 0, lookback: int = 252, spread_tail: float = 0.1,
                      min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """RM fractile for a single market."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_fractile(market)


def janus_delta_rm_fractile(closes: list[np.ndarray], market: int = 0, lookback: int = 252, delta_length: int = 20,
                            spread_tail: float = 0.1, min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Change in RM fractile over delta_length bars."""
    if delta_length < 1:
        raise ValueError(f"delta_length must be >= 1, got {delta_length}")
    frac = _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_fractile(market)
    out = np.full_like(frac, np.nan)
    out[delta_length:] = frac[delta_length:] - frac[:-delta_length]
    return out


def janus_rs_leader_equity(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1,
                           min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """Cumulative equity of RS leader."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_leader_equity


def janus_rs_laggard_equity(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                            max_cma: int = 60) -> np.ndarray:
    """Cumulative equity of RS laggard."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_laggard_equity


def janus_rs_ps(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                max_cma: int = 60) -> np.ndarray:
    """RS performance spread: leader − laggard."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_ps


def janus_rs_leader_advantage(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1,
                              min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """RS leader equity minus average equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_leader_advantage


def janus_rs_laggard_advantage(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1,
                               min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """RS laggard equity minus average equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rs_laggard_advantage


def janus_rm_leader_equity(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                           max_cma: int = 60) -> np.ndarray:
    """Cumulative equity of RM leader."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_leader_equity


def janus_rm_laggard_equity(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                            max_cma: int = 60) -> np.ndarray:
    """Cumulative equity of RM laggard."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_laggard_equity


def janus_rm_ps(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                max_cma: int = 60) -> np.ndarray:
    """RM performance spread: leader − laggard."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_ps


def janus_rm_leader_advantage(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                              max_cma: int = 60) -> np.ndarray:
    """RM leader equity minus average equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_leader_advantage


def janus_rm_laggard_advantage(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1,
                               min_cma: int = 20, max_cma: int = 60) -> np.ndarray:
    """RM laggard equity minus average equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).rm_laggard_advantage


def janus_oos_avg(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                  max_cma: int = 60) -> np.ndarray:
    """Cumulative average OOS return."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).oos_avg


def janus_cma_oos(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                  max_cma: int = 60) -> np.ndarray:
    """Adaptive CMA out-of-sample equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).cma_oos


def janus_leader_cma_oos(closes: list[np.ndarray], lookback: int = 252, spread_tail: float = 0.1, min_cma: int = 20,
                         max_cma: int = 60) -> np.ndarray:
    """Leader adaptive CMA out-of-sample equity."""
    return _make_janus(closes, lookback, spread_tail, min_cma, max_cma).leader_cma_oos


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ema_smooth(arr: np.ndarray, span: int) -> np.ndarray:
    """Apply EMA smoothing to a 1-D array, preserving NaN positions."""
    if span <= 1:
        return arr
    alpha = 2.0 / (span + 1.0)
    out = arr.copy()
    # Find first non-NaN
    first = -1
    for i in range(len(out)):
        if not np.isnan(out[i]):
            first = i
            break
    if first < 0:
        return out
    ema = out[first]
    for i in range(first + 1, len(out)):
        if np.isnan(out[i]):
            continue
        ema = alpha * out[i] + (1.0 - alpha) * ema
        out[i] = ema
    return out
