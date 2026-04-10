"""
C+GP+TS trading algorithm — replaces ITA Rule 2 (fixed AOL exit) with a
learned exit target from the C+GP model.

run_cgpts_algorithm   — full strategy execution on a test price series.

Entry logic is identical to ITA (Rule 1: upward DC + S1 regime).
Exit logic replaces the AOL heuristic:
  - αDC prediction → hold until estimated DCE (DCC + predicted OS bars).
  - βDC prediction → no position opened (trend ends at DCC; no OS to ride).
  - Early exit    → downward DC fires before estimated DCE (Rule 3 equivalent).
  - Regime exit   → S2 signal while in position (Rule 4 equivalent).

Reference: Adegboye & Kampouridis (2020) Algorithms 2–3; Section 3.3.
Integration: ITA regime gate + C+GP exit timing.
"""
import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse
from okmich_quant_ml.hmm import PomegranateHMM

from ._rcd_hmm import get_current_regime
from ._ita_sim import _compute_max_drawdown
from ._cgpts_model import predict_trend_end


def run_cgpts_algorithm(prices: pd.Series, theta: float, alpha: float, hmm: PomegranateHMM, s1_idx: int, cgpts_model: dict, initial_capital: float = 10_000.0) -> dict:
    """
    Execute the C+GP+TS strategy on a test price series.

    Uses the ITA regime filter (HMM S1/S2) for entry gating and the C+GP
    model for exit timing.  Long-only, full capital per trade.

    Entry rule  : upward DC confirmed AND HMM regime == S1 AND αDC predicted.
                  (βDC predicted → skip entry; no overshoot worth riding.)
    Exit rules  :
      C+GP exit : current_bar >= entry_bar + estimated_dce_offset.
      Early exit: downward DC confirmed before estimated DCE.
      Regime exit: HMM regime == S2 while in position.
    End-of-period: open position closed at final bar.

    Parameters
    ----------
    prices : pd.Series
        Close price series for the test window.
    theta : float
        DC threshold (Bayesian-optimised on training window, same as in cgpts_model).
    alpha : float
        Asymmetric attenuation coefficient.
    hmm : PomegranateHMM
        Fitted 2-state HMM from fit_rcd_hmm(), trained on training window.
    s1_idx : int
        HMM state index for S1 (normal regime) from s1_state_index().
    cgpts_model : dict
        Output of train_cgpts_model().
    initial_capital : float
        Starting capital.  Default 10,000.

    Returns
    -------
    dict
        final_capital     : float — capital after all trades.
        cumulative_return : float — CRR % = (final - initial) / initial * 100.
        max_drawdown      : float — MDD % peak-to-trough on capital curve.
        n_trades          : int   — completed round-trip trades.
        n_winners         : int   — trades with pnl_pct > 0.
        trade_log         : list  — one dict per completed trade.
    """
    idc = idc_parse(prices, theta, alpha)

    price_arr = prices.values.astype(np.float64)
    upturn_dc_arr = idc['upturn_dc'].values
    downturn_dc_arr = idc['downturn_dc'].values
    rdc_arr = idc['rdc'].values
    n = len(price_arr)

    # Build a bar-indexed lookup from parse_dc_events for DC event attributes.
    # At each upturn_dc bar we need: dc_length, dcc_price, ext_end_price,
    # previous dcc_price, and previous has_os.
    # We use idc ph/pl for live ext_end price reconstruction, and trends for
    # the confirmed per-event data.
    #
    # Strategy: for each upturn DC bar (idc), find the corresponding trend row
    # in parse_dc_events where dcc_pos matches the current bar position.
    # The dc_length and label data come from label_alpha_beta_dc(), already
    # embedded in cgpts_model via training — but for TEST inference we only
    # need the raw DC attributes available at DCC bar, which idc provides.

    capital = initial_capital
    in_position = False
    entry_price = 0.0
    entry_bar = 0
    target_exit_bar = 0
    prev_dcc_price = None
    prev_has_os = False
    capital_curve = [initial_capital]
    trade_log = []
    n_trades = 0
    n_winners = 0

    # t_dc0 from idc gives the bar of the last confirmed extreme (trough for
    # upturn mode, peak for downturn mode).
    t_dc0_arr = idc['t_dc0'].values
    ph_arr = idc['ph'].values
    pl_arr = idc['pl'].values

    for i in range(n):
        p = price_arr[i]

        if in_position:
            # ── C+GP exit: estimated DCE reached ─────────────────────────────
            if i >= target_exit_bar:
                pnl = (p - entry_price) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                trade_log.append({
                    'entry_bar': entry_bar, 'entry_price': entry_price,
                    'exit_bar': i, 'exit_price': p,
                    'exit_rule': 'CGP_EstimatedDCE',
                    'pnl_pct': pnl * 100.0, 'capital': capital,
                })
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                in_position = False

            # ── Early exit: downward DC before target ─────────────────────────
            elif downturn_dc_arr[i]:
                pnl = (p - entry_price) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                trade_log.append({
                    'entry_bar': entry_bar, 'entry_price': entry_price,
                    'exit_bar': i, 'exit_price': p,
                    'exit_rule': 'EarlyExit_DownturnDC',
                    'pnl_pct': pnl * 100.0, 'capital': capital,
                })
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                in_position = False

        # ── Upward DC confirmed ───────────────────────────────────────────────
        if upturn_dc_arr[i]:
            rdc_val = rdc_arr[i]
            if not np.isfinite(rdc_val):
                # Initial DC — no completed prior trend, skip HMM query
                continue

            state = get_current_regime(hmm, float(rdc_val), s1_idx)

            # Regime exit: S2 while in position
            if state == 2 and in_position:
                pnl = (p - entry_price) / entry_price
                capital *= 1.0 + pnl
                capital_curve.append(capital)
                trade_log.append({
                    'entry_bar': entry_bar, 'entry_price': entry_price,
                    'exit_bar': i, 'exit_price': p,
                    'exit_rule': 'RegimeExit_S2',
                    'pnl_pct': pnl * 100.0, 'capital': capital,
                })
                n_trades += 1
                if pnl > 0:
                    n_winners += 1
                in_position = False

            elif state == 1 and not in_position:
                # Compute DC event attributes at this upturn_dc bar.
                # ext_end (trough) is at t_dc0 (last extreme bar before upturn DCC).
                trough_bar = int(t_dc0_arr[i])
                trough_price = float(pl_arr[i])
                dc_len = i - trough_bar
                dcc_p = float(p)

                prediction = predict_trend_end(
                    dc_length=dc_len,
                    dcc_price=dcc_p,
                    ext_end_price=trough_price,
                    prev_dcc_price=prev_dcc_price,
                    prev_has_os=prev_has_os,
                    cgpts_model=cgpts_model,
                )

                # Update memory for next event
                prev_dcc_price = dcc_p
                prev_has_os = prediction['trend_type'] == 'alpha_dc'

                if prediction['trend_type'] == 'alpha_dc':
                    in_position = True
                    entry_price = p
                    entry_bar = i
                    target_exit_bar = i + prediction['estimated_dce_offset']
                # βDC → skip entry (no overshoot predicted)

    # Close any open position at end of period
    if in_position:
        p = price_arr[-1]
        pnl = (p - entry_price) / entry_price
        capital *= 1.0 + pnl
        capital_curve.append(capital)
        trade_log.append({
            'entry_bar': entry_bar, 'entry_price': entry_price,
            'exit_bar': n - 1, 'exit_price': p,
            'exit_rule': 'EndOfPeriod',
            'pnl_pct': pnl * 100.0, 'capital': capital,
        })
        n_trades += 1
        if pnl > 0:
            n_winners += 1

    return {
        'final_capital': capital,
        'cumulative_return': (capital - initial_capital) / initial_capital * 100.0,
        'max_drawdown': _compute_max_drawdown(capital_curve),
        'n_trades': n_trades,
        'n_winners': n_winners,
        'trade_log': trade_log,
    }
