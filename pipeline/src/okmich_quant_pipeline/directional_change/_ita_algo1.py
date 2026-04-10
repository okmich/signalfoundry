"""
ITA Algorithm 1 — Intelligent Trading Algorithm with HMM regime filter.

Full implementation of Wu & Han (2023) Algorithm 1, extended to both directions:

Long rules:
  Rule 1L — BUY  : upward DC confirmed AND HMM regime == S1 (normal).
  Rule 2L — SELL : new high reached AND ph >= (1 + 2θ) * pl (AOL profit target).
  Rule 3L — SELL : downward DC confirmed (asymmetric stop-loss).
  Rule 4L — SELL : upward DC confirmed AND HMM regime == S2 (abnormal) — close
                   any open long; do not open new ones.

Short rules (mirror):
  Rule 1S — SHORT : downward DC confirmed AND HMM regime == S1.
  Rule 2S — COVER : new low reached AND pl <= (1 - 2θ) * ph (AOL profit target).
  Rule 3S — COVER : upward DC confirmed (asymmetric stop-loss).
  Rule 4S — COVER : downward DC confirmed AND HMM regime == S2 — close any
                    open short; do not open new ones.

Per-bar signals are sourced from idc_parse(), which provides upturn_dc,
downturn_dc, new_high, new_low, ph, pl, and rdc as a pre-computed DataFrame.
The HMM is queried at upward DC bars (for long rules) and downward DC bars
(for short rules) using the RDC of the just-completed trend.

An exit and new entry can occur on the same DC confirmation bar:
  - Downward DC: Rule 3L may close a long, then Rule 1S opens a short (if S1).
  - Upward   DC: Rule 3S may close a short, then Rule 1L opens a long (if S1).

For the no-HMM variant used in Bayesian optimisation, see run_ita_simulation()
in _ita_sim.py.

References
----------
Wu, Y. & Han, J. (2023). Intelligent Trading Strategy Based on Improved
    Directional Change and Regime Change Detection. arXiv:2309.15383v1.
    Algorithm 1 (lines 1–35), Section 3.3.2 (trading rules), Eq. 6–7 (metrics).

Hu, Z., Li, Y. & Wu, Y. (2022). Incorporating Improved Directional Change and
    Regime Change Detection to Formulate Trading Strategies in Foreign Exchange
    Markets. SSRN:4048864. Section 2.4.2 (rule definitions).
"""
import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse
from okmich_quant_ml.hmm import PomegranateHMM

from ._rcd_hmm import get_current_regime
from ._ita_sim import _compute_max_drawdown


def run_ita_algorithm1(prices: pd.Series, theta: float, alpha: float, hmm: PomegranateHMM, s1_idx: int, initial_capital: float = 10_000.0) -> dict:
    """
    Execute ITA Algorithm 1 (long + short) on a price series using a fitted RCD-HMM.

    Implements Wu & Han (2023) Algorithm 1 extended to both sides. The HMM is
    queried at upward DC bars (long entry/exit gate) and downward DC bars
    (short entry/exit gate). The initial DC (rdc=NaN) is skipped for both sides.

    Parameters
    ----------
    prices : pd.Series
        Close price series for the test window, in chronological order.
    theta : float
        DC threshold (Bayesian optimised on training window).
    alpha : float
        Asymmetric attenuation coefficient for the opposite-direction DC threshold.
    hmm : PomegranateHMM
        Fitted 2-state HMM from fit_rcd_hmm(), trained on training window RDC.
    s1_idx : int
        HMM state index corresponding to S1 (normal regime), from s1_state_index().
    initial_capital : float
        Starting capital. Wu & Han (2023) paper default: 10,000 EUR.

    Returns
    -------
    dict
        final_capital     : float — capital after all trades.
        cumulative_return : float — CRR % = (final - initial) / initial * 100.
        max_drawdown      : float — MDD % peak-to-trough on capital curve.
        n_trades          : int   — completed round-trip trades (long + short).
        n_winners         : int   — trades with pnl > 0.
        win_ratio         : float — n_winners / n_trades, 0 if no trades.
        profit_factor     : float — gross_profit / gross_loss, inf if no losses.
        sharpe            : float — mean(pnl) / std(pnl) across trades, 0 if < 2.
        trade_log         : list  — one dict per completed trade with 'side' field.

    Notes
    -----
    - Full capital deployed per trade (paper default).
    - Entry and exit at close of signal bar (no slippage modelled).
    - Open position is closed at the last bar if no exit rule fires first.
    - Short PnL: (entry_price - exit_price) / entry_price.
    - Rules 4L/4S (close on S2) are structurally unreachable via idc_parse batch
      signals, as the DC state machine guarantees an intervening opposite-direction
      DC between any two same-direction DCs. They are retained for completeness.
    """
    idc = idc_parse(prices, theta, alpha)

    price_arr = prices.values.astype(np.float64)
    upturn_dc_arr = idc["upturn_dc"].values
    downturn_dc_arr = idc["downturn_dc"].values
    new_high_arr = idc["new_high"].values
    new_low_arr = idc["new_low"].values
    ph_arr = idc["ph"].values
    pl_arr = idc["pl"].values
    rdc_arr = idc["rdc"].values

    n = len(price_arr)
    long_aol_factor = 1.0 + 2.0 * theta
    short_aol_factor = 1.0 - 2.0 * theta

    capital = initial_capital
    position = 0   # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    capital_curve = [initial_capital]
    trade_log = []
    pnl_list = []
    n_trades = 0
    n_winners = 0

    def _close(exit_price, exit_bar, exit_rule, side):
        nonlocal capital, position, n_trades, n_winners
        if side == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        capital *= 1.0 + pnl
        capital_curve.append(capital)
        trade_log.append({
            'entry_bar': entry_bar, 'entry_price': entry_price,
            'exit_bar': exit_bar, 'exit_price': exit_price,
            'exit_rule': exit_rule, 'pnl_pct': pnl * 100.0,
            'capital': capital, 'side': side,
        })
        pnl_list.append(pnl)
        n_trades += 1
        if pnl > 0:
            n_winners += 1
        position = 0

    for i in range(n):
        p = price_arr[i]

        # ── Exit logic ────────────────────────────────────────────────────
        if position == 1:
            if new_high_arr[i] and ph_arr[i] >= long_aol_factor * pl_arr[i]:
                _close(p, i, 'Rule2L_AOLTarget', 'long')
            elif downturn_dc_arr[i]:
                _close(p, i, 'Rule3L_StopLoss', 'long')

        elif position == -1:
            if new_low_arr[i] and pl_arr[i] <= short_aol_factor * ph_arr[i]:
                _close(p, i, 'Rule2S_AOLTarget', 'short')
            elif upturn_dc_arr[i]:
                _close(p, i, 'Rule3S_StopLoss', 'short')

        # ── Upward DC: Rule 4L (close long on S2) + Rule 1L (enter long on S1) ──
        if upturn_dc_arr[i]:
            rdc_val = rdc_arr[i]
            if not np.isfinite(rdc_val):
                continue
            state = get_current_regime(hmm, float(rdc_val), s1_idx)

            if state == 2 and position == 1:
                # Rule 4L: abnormal regime — close open long
                _close(p, i, 'Rule4L_RegimeFilter', 'long')
            elif state == 1 and position == 0:
                # Rule 1L: normal regime — enter long
                position = 1
                entry_price = p
                entry_bar = i

        # ── Downward DC: Rule 4S (close short on S2) + Rule 1S (enter short on S1) ─
        elif downturn_dc_arr[i]:
            rdc_val = rdc_arr[i]
            if not np.isfinite(rdc_val):
                continue
            state = get_current_regime(hmm, float(rdc_val), s1_idx)

            if state == 2 and position == -1:
                # Rule 4S: abnormal regime — close open short
                _close(p, i, 'Rule4S_RegimeFilter', 'short')
            elif state == 1 and position == 0:
                # Rule 1S: normal regime — enter short
                position = -1
                entry_price = p
                entry_bar = i

    # Close any open position at end of test period
    if position != 0:
        p = price_arr[-1]
        side = 'long' if position == 1 else 'short'
        _close(p, n - 1, 'EndOfPeriod', side)

    gross_profit = sum(x for x in pnl_list if x > 0)
    gross_loss = abs(sum(x for x in pnl_list if x < 0))
    std = np.std(pnl_list) if len(pnl_list) >= 2 else 0.0
    sharpe = float(np.mean(pnl_list) / std) if std > 1e-10 else 0.0

    return {
        'final_capital': capital,
        'cumulative_return': (capital - initial_capital) / initial_capital * 100.0,
        'max_drawdown': _compute_max_drawdown(capital_curve),
        'n_trades': n_trades,
        'n_winners': n_winners,
        'win_ratio': n_winners / n_trades if n_trades > 0 else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'sharpe': sharpe,
        'trade_log': trade_log,
    }
