"""
ITA simulation — Rules 1–3 (long) and 1S–3S (short) without HMM regime filter.

Used as the objective function inside Bayesian optimisation of (θ, α):
evaluates raw DC profitability on a training window without fitting an HMM,
keeping each candidate evaluation fast.

Long rules (Wu & Han 2023 Section 3.3.2 / Algorithm 1):
  Rule 1L — BUY   : upward DC confirmed → open long at current price.
  Rule 2L — SELL  : new high AND ph >= (1 + 2θ) * pl → AOL profit target.
  Rule 3L — SELL  : downward DC confirmed → asymmetric stop-loss.

Short rules (mirror):
  Rule 1S — SELL  : downward DC confirmed → open short at current price.
  Rule 2S — BUY   : new low AND pl <= (1 - 2θ) * ph → AOL profit target.
  Rule 3S — BUY   : upward DC confirmed → asymmetric stop-loss.

Rule 4 (regime filter) is intentionally omitted — the HMM is not available
during Bayesian optimisation. All bars are treated as regime S1 (normal).

References
----------
Wu, Y. & Han, J. (2023). Intelligent Trading Strategy Based on Improved
    Directional Change and Regime Change Detection. arXiv:2309.15383v1.
    Section 3.3.2 (trading rules), Algorithm 1 (pseudocode), Eq. 6–7 (metrics).

Hu, Z., Li, Y. & Wu, Y. (2022). Incorporating Improved Directional Change and
    Regime Change Detection to Formulate Trading Strategies in Foreign Exchange
    Markets. SSRN:4048864. Section 2.4.2 (rule definitions).
"""
import numpy as np
import pandas as pd


def run_ita_simulation(idc: pd.DataFrame, prices: pd.Series, theta: float, initial_capital: float = 10_000.0) -> dict:
    """
    Simulate ITA trading rules 1–3 (long and short) on idc_parse() output without HMM regime filter.

    Intended as the objective function for Bayesian optimisation of (θ, α).
    All DC confirmation bars are treated as regime S1 (normal) — no HMM is
    queried. Rule 4 (regime suspension) is therefore not applied.

    Both long and short sides are traded symmetrically:
      - Upward DC → enter long; Downward DC → exit long OR enter short.
      - Downward DC → enter short; Upward DC → exit short OR enter long.
    An exit and entry can occur on the same bar (e.g. Rule 3L fires on
    downturn_dc, then Rule 1S immediately opens a short at the same price).

    Parameters
    ----------
    idc : pd.DataFrame
        Output of idc_parse(prices, theta, alpha). Must contain columns:
        upturn_dc, downturn_dc, new_high, new_low, ph, pl.
    prices : pd.Series
        Close price series aligned to idc.index.
    theta : float
        DC threshold — used for the Rule 2 AOL profit target checks.
    initial_capital : float
        Starting capital (default 10,000 — Wu & Han 2023 paper default).

    Returns
    -------
    dict
        cumulative_return : float — CRR % = (final - initial) / initial * 100.
        n_trades          : int   — number of completed round-trip trades.
        n_winners         : int   — trades with pnl > 0.
        max_drawdown      : float — MDD % peak-to-trough on capital curve.

    Notes
    -----
    - Full capital deployed per trade (paper default).
    - Entry and exit prices are the close of the signal bar (no slippage).
    - Open position is closed at the last bar if no exit rule fires first.
    - Short PnL: (entry_price - exit_price) / entry_price.
    """
    price_arr = prices.values.astype(np.float64)
    upturn_dc_arr = idc["upturn_dc"].values
    downturn_dc_arr = idc["downturn_dc"].values
    new_high_arr = idc["new_high"].values
    new_low_arr = idc["new_low"].values
    ph_arr = idc["ph"].values
    pl_arr = idc["pl"].values

    n = len(price_arr)
    long_aol_factor = 1.0 + 2.0 * theta   # Rule 2L: ph >= factor * pl
    short_aol_factor = 1.0 - 2.0 * theta  # Rule 2S: pl <= factor * ph

    capital = initial_capital
    position = 0   # 0=flat, 1=long, -1=short
    entry_price = 0.0
    capital_curve = [initial_capital]
    n_trades = 0
    n_winners = 0

    for i in range(n):
        p = price_arr[i]

        # ── Exit logic ─────────────────────────────────────────────────────
        if position == 1:
            # Rule 2L: AOL profit target for long
            if new_high_arr[i] and ph_arr[i] >= long_aol_factor * pl_arr[i]:
                pnl = (p - entry_price) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                position = 0
            # Rule 3L: downturn DC stop-loss for long
            elif downturn_dc_arr[i]:
                pnl = (p - entry_price) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                position = 0

        elif position == -1:
            # Rule 2S: AOL profit target for short
            if new_low_arr[i] and pl_arr[i] <= short_aol_factor * ph_arr[i]:
                pnl = (entry_price - p) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                position = 0
            # Rule 3S: upturn DC stop-loss for short
            elif upturn_dc_arr[i]:
                pnl = (entry_price - p) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                position = 0

        # ── Entry logic (runs after exit, same bar OK) ─────────────────────
        if position == 0:
            if upturn_dc_arr[i]:
                # Rule 1L: enter long on upward DC
                entry_price = p
                position = 1
            elif downturn_dc_arr[i]:
                # Rule 1S: enter short on downward DC
                entry_price = p
                position = -1

    # Close any open position at end of period
    if position != 0:
        p = price_arr[-1]
        pnl = (p - entry_price) / entry_price if position == 1 else (entry_price - p) / entry_price
        capital *= 1.0 + pnl
        capital_curve.append(capital)
        n_trades += 1
        if pnl > 0:
            n_winners += 1

    return {
        "cumulative_return": (capital - initial_capital) / initial_capital * 100.0,
        "n_trades": n_trades,
        "n_winners": n_winners,
        "max_drawdown": _compute_max_drawdown(capital_curve),
    }


def _compute_max_drawdown(capital_curve: list) -> float:
    """
    Maximum drawdown as a percentage of peak capital.

    Wu & Han (2023) Equation 7: MDD = max(Px - Py) / Px where y > x.
    """
    peak = capital_curve[0]
    max_dd = 0.0
    for c in capital_curve:
        if c > peak:
            peak = c
        dd = (peak - c) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100.0
