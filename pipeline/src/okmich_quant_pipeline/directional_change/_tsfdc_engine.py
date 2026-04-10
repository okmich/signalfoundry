"""
TSFDC trading engine — executes Rules 1, 2, 3 for both TSFDC-down and TSFDC-up.

run_tsfdc_algorithm   — runs the full dual-direction TSFDC strategy on a price series.

Entry rules (contrarian):
  Rule .1 (immediate): STheta DCC confirmed + FBBTheta=False → open position immediately.
  Rule .2 (delayed):   BTheta DCC confirmed + stored FBBTheta=True → open position at BTheta DCC.

Exit rule:
  Rule .3: Opposite-direction STheta DCC confirmed while in position → close position.

Priority at each bar: exits > Rule .2 delayed entry > Rule .1 immediate entry.

Reference: Bakhach, Tsang & Chinthalapati (ISAFM, 2018) Sections 4.1, 4.2, 11.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from okmich_quant_features.directional_change import idc_parse

from ._ita_sim import _compute_max_drawdown
from ._tsfdc_classifier import predict_bbtheta


def run_tsfdc_algorithm(prices: pd.Series, stheta: float, btheta: float, classifier: Pipeline, initial_capital_down: float = 10_000.0, initial_capital_up: float = 10_000.0) -> dict:
    """
    Execute TSFDC-down and TSFDC-up strategies on a test price series.

    Runs both strategies simultaneously on the same bar stream.  Each strategy
    maintains an independent position and capital curve.  Only one position per
    direction can be open at a time.

    The dual DC parsers are implemented using two idc_parse() calls, which give
    per-bar STheta and BTheta DCC confirmation flags along with running ph/pl
    extremes and t_dc0 (bar of last confirmed extreme).

    Live feature computation at each STheta DCC:
      TMV = abs(curr_ext - prev_ext) / (prev_ext * stheta)
      T   = curr_ext_bar - prev_ext_bar
      OSV = (curr_dcc_price - prev_dcc_price) / (prev_dcc_price * stheta)
      COP = (prev_ext_price - last_btheta_dcc_price) / (last_btheta_dcc_price * btheta)
            where prev_ext_price is the start of the current trend (previous extreme)

    End-of-period forced closes are excluded from performance metrics per
    Bakhach et al. (2018) Section 6.1.

    Parameters
    ----------
    prices : pd.Series
        Test price series.
    stheta : float
        Small DC threshold.
    btheta : float
        Big DC threshold (> stheta).
    classifier : sklearn.pipeline.Pipeline
        Fitted BBTheta classifier from train_bbtheta_classifier().
    initial_capital_down : float
        Starting capital for the TSFDC-down (long) strategy.
    initial_capital_up : float
        Starting capital for the TSFDC-up (short) strategy.

    Returns
    -------
    dict with keys:
        down : dict — TSFDC-down results (long trades on downtrend reversals).
        up   : dict — TSFDC-up results (short trades on uptrend reversals).

    Each direction dict contains:
        final_capital     : float — capital after all counted trades.
        cumulative_return : float — (final - initial) / initial * 100.
        max_drawdown      : float — MDD % peak-to-trough.
        n_trades          : int   — completed trades (end-of-period excluded).
        n_winners         : int   — trades with pnl_pct > 0.
        win_ratio         : float — n_winners / n_trades (0 if n_trades == 0).
        profit_factor     : float — gross profit / gross loss (inf if no losses).
        trade_log         : list  — one dict per trade entry+exit pair.
    """
    if btheta <= stheta:
        raise ValueError(f"btheta ({btheta}) must be greater than stheta ({stheta})")

    idc_s = idc_parse(prices, stheta)
    idc_b = idc_parse(prices, btheta)

    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)

    upturn_dc_s = idc_s['upturn_dc'].values
    downturn_dc_s = idc_s['downturn_dc'].values
    ph_s = idc_s['ph'].values
    pl_s = idc_s['pl'].values
    t_dc0_s = idc_s['t_dc0'].values

    upturn_dc_b = idc_b['upturn_dc'].values
    downturn_dc_b = idc_b['downturn_dc'].values

    # ── Shared live feature state ─────────────────────────────────────────────
    prev_s_ext_price = np.nan
    prev_s_ext_bar = -1
    prev_s_dcc_price = np.nan
    last_b_dcc_price = np.nan

    # ── TSFDC-down state (generates LONG trades) ──────────────────────────────
    pos_d = 0           # 0=flat, 1=long
    entry_price_d = 0.0
    entry_bar_d = 0
    pending_d = False   # True → waiting for BTheta down DCC to enter
    capital_d = float(initial_capital_down)
    capital_curve_d = [capital_d]
    trades_d: list = []
    open_trade_d: dict | None = None

    # ── TSFDC-up state (generates SHORT trades) ───────────────────────────────
    pos_u = 0           # 0=flat, -1=short
    entry_price_u = 0.0
    entry_bar_u = 0
    pending_u = False   # True → waiting for BTheta up DCC to enter
    capital_u = float(initial_capital_up)
    capital_curve_u = [capital_u]
    trades_u: list = []
    open_trade_u: dict | None = None

    for i in range(n):
        p = price_arr[i]

        # ── Update last BTheta DCC price FIRST (used in COP at same bar) ──────
        b_up = bool(upturn_dc_b[i])
        b_dn = bool(downturn_dc_b[i])
        if b_up or b_dn:
            last_b_dcc_price = p

        # ── Handle STheta DCC events ─────────────────────────────────────────
        s_up = bool(upturn_dc_s[i])
        s_dn = bool(downturn_dc_s[i])

        if s_up or s_dn:
            # Determine current trend's ext_price and ext_bar from idc state
            if s_up:
                curr_ext_price = float(pl_s[i])   # trough confirmed
            else:
                curr_ext_price = float(ph_s[i])   # peak confirmed
            curr_ext_bar = int(t_dc0_s[i])
            curr_dcc_price = p

            # Compute live features
            if not np.isnan(prev_s_ext_price):
                tmv = abs(curr_ext_price - prev_s_ext_price) / (prev_s_ext_price * stheta)
                t_feat = float(curr_ext_bar - prev_s_ext_bar)
                osv = (curr_dcc_price - prev_s_dcc_price) / (prev_s_dcc_price * stheta)
                cop = (prev_s_ext_price - last_b_dcc_price) / (last_b_dcc_price * btheta) if not np.isnan(last_b_dcc_price) else np.nan
            else:
                tmv = t_feat = osv = np.nan
                cop = np.nan

            # Predict FBBTheta (False if any feature is NaN)
            if not (np.isnan(tmv) or np.isnan(t_feat) or np.isnan(osv)):
                fbtheta = predict_bbtheta(classifier, tmv, t_feat, osv, cop)
            else:
                fbtheta = False

            # ── TSFDC-down: process downturn DCC ─────────────────────────────
            if s_dn:
                # Rule down.3: upward STheta DCC closes long → not applicable here (s_dn)
                # Cancel stale pending if opposite direction
                if pending_d:
                    # pending_d was set on a downturn DCC; another downturn cancels stale
                    # Actually: stale if we were waiting for BTheta down but something odd
                    pass  # same direction, don't cancel

                # Rule down.1 or store for Rule down.2
                if pos_d == 0:
                    if not fbtheta:
                        # Rule down.1: immediate LONG
                        pos_d = 1
                        entry_price_d = p
                        entry_bar_d = i
                        open_trade_d = {'entry_bar': i, 'entry_price': p, 'entry_rule': 'down.1'}
                        pending_d = False
                    else:
                        # Store for Rule down.2 (wait for BTheta down DCC)
                        pending_d = True

            # ── TSFDC-down: upward STheta DCC fires Rule down.3 (exit) ───────
            if s_up:
                # Cancel pending if opposite direction arrived
                if pending_d:
                    pending_d = False
                # Rule down.3: close long position
                if pos_d == 1:
                    pnl_d = (p - entry_price_d) / entry_price_d
                    capital_d *= 1.0 + pnl_d
                    capital_curve_d.append(capital_d)
                    trades_d.append({
                        **open_trade_d,
                        'exit_bar': i, 'exit_price': p, 'exit_rule': 'down.3',
                        'pnl_pct': pnl_d * 100.0, 'capital': capital_d,
                    })
                    pos_d = 0
                    open_trade_d = None

            # ── TSFDC-up: process upturn DCC ─────────────────────────────────
            if s_up:
                if pending_u:
                    pass  # same direction, keep waiting

                if pos_u == 0:
                    if not fbtheta:
                        # Rule up.1: immediate SHORT
                        pos_u = -1
                        entry_price_u = p
                        entry_bar_u = i
                        open_trade_u = {'entry_bar': i, 'entry_price': p, 'entry_rule': 'up.1'}
                        pending_u = False
                    else:
                        pending_u = True

            # ── TSFDC-up: downward STheta DCC fires Rule up.3 (exit) ─────────
            if s_dn:
                if pending_u:
                    pending_u = False
                if pos_u == -1:
                    pnl_u = (entry_price_u - p) / entry_price_u
                    capital_u *= 1.0 + pnl_u
                    capital_curve_u.append(capital_u)
                    trades_u.append({
                        **open_trade_u,
                        'exit_bar': i, 'exit_price': p, 'exit_rule': 'up.3',
                        'pnl_pct': pnl_u * 100.0, 'capital': capital_u,
                    })
                    pos_u = 0
                    open_trade_u = None

            # ── Update shared state after processing ──────────────────────────
            prev_s_ext_price = curr_ext_price
            prev_s_ext_bar = curr_ext_bar
            prev_s_dcc_price = curr_dcc_price

        # ── Handle BTheta DCC events — Rule .2 delayed entries ───────────────
        if b_dn and pending_d and pos_d == 0:
            # Rule down.2: BTheta downward DCC fires, open delayed LONG
            pos_d = 1
            entry_price_d = p
            entry_bar_d = i
            open_trade_d = {'entry_bar': i, 'entry_price': p, 'entry_rule': 'down.2'}
            pending_d = False

        if b_up and pending_u and pos_u == 0:
            # Rule up.2: BTheta upward DCC fires, open delayed SHORT
            pos_u = -1
            entry_price_u = p
            entry_bar_u = i
            open_trade_u = {'entry_bar': i, 'entry_price': p, 'entry_rule': 'up.2'}
            pending_u = False

    # ── End-of-period: close open positions but exclude from metrics ──────────
    final_price = price_arr[-1]

    def _close_eop(pos, entry_price, open_trade, capital, capital_curve):
        """Close an open position at end of period; return updated capital only."""
        if pos != 0 and open_trade is not None:
            if pos == 1:
                pnl = (final_price - entry_price) / entry_price
            else:
                pnl = (entry_price - final_price) / entry_price
            return capital * (1.0 + pnl)
        return capital

    capital_d = _close_eop(pos_d, entry_price_d, open_trade_d, capital_d, capital_curve_d)
    capital_u = _close_eop(pos_u, entry_price_u, open_trade_u, capital_u, capital_curve_u)

    return {
        'down': _summarise(trades_d, capital_d, float(initial_capital_down), capital_curve_d),
        'up': _summarise(trades_u, capital_u, float(initial_capital_up), capital_curve_u),
    }


def _summarise(trades: list, final_capital: float, initial_capital: float, capital_curve: list) -> dict:
    n_trades = len(trades)
    n_winners = sum(1 for t in trades if t['pnl_pct'] > 0)
    mdd = _compute_max_drawdown(capital_curve)
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    return {
        'final_capital': final_capital,
        'cumulative_return': (final_capital - initial_capital) / initial_capital * 100.0,
        'max_drawdown': mdd,
        'n_trades': n_trades,
        'n_winners': n_winners,
        'win_ratio': n_winners / n_trades if n_trades > 0 else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'trade_log': trades,
    }
