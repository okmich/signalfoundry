"""
ATDC trading engine — Adaptive Threshold Directional Change.

run_atdc_algorithm — execute ITA-style DC strategy where theta adapts periodically
                     based on a chosen market metric (RDC volatility, price volatility,
                     TMV, or a user-supplied callable).

Both long and short sides are traded (same rules as ITA Algorithm 1):
  Long  entry : upward DC confirmed AND (S1 regime OR no HMM).
  Short entry : downward DC confirmed AND (S1 regime OR no HMM).
  Long  exits : AOL profit target (Rule 2L), downturn DC stop-loss (Rule 3L),
                S2 regime gate (Rule 4L), end-of-period.
  Short exits : AOL profit target (Rule 2S), upturn DC stop-loss (Rule 3S),
                S2 regime gate (Rule 4S), end-of-period.

Theta adaptation
----------------
The price series is processed in blocks of `adaptation_step` bars.  Before each
block the metric is computed on the previous `lookback_window` bars, and theta is
updated via:
    signal   = (metric - baseline) / (|baseline| + ε)
    theta_new = clip(theta_old × (1 + adaptation_rate × signal), theta_min, theta_max)

idc_parse is re-run on the full series up to the end of each block with the updated
theta so that DC events reflect the new threshold.  Position state is carried across
blocks.

If `hmm=None` the regime gate is disabled and all DC events are treated as S1.
If `use_gp=False` the C+GP trend-end model is not used; instead long exits happen
at the next opposite DC event or via the AOL target only.
"""
import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse

from ._atdc_adapter import AdaptationMode, compute_adapted_theta
from ._ita_sim import _compute_max_drawdown
from ._rcd_hmm import get_current_regime


def run_atdc_algorithm(prices: pd.Series, theta_init: float, theta_min: float, theta_max: float, adaptation_rate: float, lookback_window: int, adaptation_step: int, adaptation_mode: AdaptationMode | str = 'volatility', alpha: float = 1.0, hmm=None, s1_idx: int = 0, use_gp: bool = False, cgpts_model=None, initial_capital: float = 10_000.0, custom_fn=None) -> dict:
    """
    Execute the ATDC strategy on a test price series.

    Parameters
    ----------
    prices : pd.Series
        Price series to trade on.
    theta_init : float
        Initial DC threshold.
    theta_min : float
        Minimum allowed theta (lower adaptation bound).
    theta_max : float
        Maximum allowed theta (upper adaptation bound).
    adaptation_rate : float
        Sensitivity of theta to metric deviation. Typical range: 0.1 – 2.0.
    lookback_window : int
        Number of bars to look back when computing the adaptation metric.
    adaptation_step : int
        Number of bars between theta updates.
    adaptation_mode : AdaptationMode or str
        'volatility' (default), 'rdc', 'tmv', or 'custom'.
    alpha : float
        Asymmetric DC attenuation coefficient.
    hmm : PomegranateHMM or None
        Fitted HMM for regime gating. None → no regime gate.
    s1_idx : int
        HMM state index for S1 (normal regime). Ignored when hmm=None.
    use_gp : bool
        If True, use cgpts_model for long exit timing (C+GP target bar).
        If False, long positions exit on opposite DC or AOL target only.
    cgpts_model : dict or None
        Trained C+GP model from train_cgpts_model(). Required when use_gp=True.
    initial_capital : float
        Starting capital.
    custom_fn : callable or None
        Custom metric function when adaptation_mode='custom'.
        Signature: fn(prices_window, current_theta) -> float.

    Returns
    -------
    dict
        final_capital, cumulative_return, max_drawdown, n_trades, n_winners,
        win_ratio, profit_factor, sharpe, theta_history, trade_log.

    Notes
    -----
    - Short PnL: (entry_price - exit_price) / entry_price.
    - C+GP exit is only used for long positions (cgpts_model trained on upturn data).
    - If use_gp=True but cgpts_model is None, GP exit is silently skipped.
    """
    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)
    mode = AdaptationMode(adaptation_mode)

    if use_gp and cgpts_model is not None:
        from ._cgpts_model import predict_trend_end
        _predict = predict_trend_end
    else:
        _predict = None

    # State
    theta = float(theta_init)
    long_aol_factor = 1.0 + 2.0 * theta
    short_aol_factor = 1.0 - 2.0 * theta
    capital = float(initial_capital)
    position = 0       # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    target_exit_bar = 0
    prev_dcc_price = None
    prev_has_os = False
    capital_curve = [capital]
    trade_log = []
    pnl_list = []
    n_trades = 0
    n_winners = 0

    # Adaptation state
    baseline_metric = 0.0
    metric_history = []
    theta_history = [(0, theta)]

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

    for block_start in range(0, n, adaptation_step):
        block_end = min(block_start + adaptation_step, n)

        # ── Adapt theta before this block (except the very first) ─────────
        if block_start > 0 and block_start >= lookback_window:
            lb_start = max(0, block_start - lookback_window)
            prices_lb = prices.iloc[lb_start:block_start]
            theta, metric = compute_adapted_theta(
                prices_lb, theta, baseline_metric, mode,
                adaptation_rate, theta_min, theta_max, alpha, custom_fn,
            )
            metric_history.append(metric)
            baseline_metric = float(np.mean(metric_history))
            long_aol_factor = 1.0 + 2.0 * theta
            short_aol_factor = 1.0 - 2.0 * theta
            theta_history.append((block_start, theta))

        # ── Parse idc for the full series up to block_end with current theta ─
        prices_to_end = prices.iloc[:block_end]
        try:
            idc_full = idc_parse(prices_to_end, theta, alpha)
        except Exception:
            continue

        idc_block = idc_full.iloc[block_start:block_end]
        upturn_dc_arr = idc_block['upturn_dc'].values
        downturn_dc_arr = idc_block['downturn_dc'].values
        new_high_arr = idc_block['new_high'].values
        new_low_arr = idc_block['new_low'].values
        ph_arr = idc_block['ph'].values
        pl_arr = idc_block['pl'].values
        rdc_arr = idc_block['rdc'].values
        t_dc0_arr = idc_block['t_dc0'].values if 't_dc0' in idc_block.columns else None
        abs_block_bars = range(block_start, block_end)

        for j, abs_i in enumerate(abs_block_bars):
            p = price_arr[abs_i]

            # ── Exit logic ────────────────────────────────────────────────
            if position == 1:
                if new_high_arr[j] and ph_arr[j] >= long_aol_factor * pl_arr[j]:
                    _close(p, abs_i, 'Rule2L_AOLTarget', 'long')
                elif downturn_dc_arr[j]:
                    _close(p, abs_i, 'Rule3L_StopLoss', 'long')
                elif _predict is not None and abs_i >= target_exit_bar:
                    _close(p, abs_i, 'ATDC_CGP_Target', 'long')

            elif position == -1:
                if new_low_arr[j] and pl_arr[j] <= short_aol_factor * ph_arr[j]:
                    _close(p, abs_i, 'Rule2S_AOLTarget', 'short')
                elif upturn_dc_arr[j]:
                    _close(p, abs_i, 'Rule3S_StopLoss', 'short')

            # ── Regime query ──────────────────────────────────────────────
            rdc_val = rdc_arr[j]
            if hmm is not None and np.isfinite(rdc_val):
                regime = get_current_regime(hmm, float(rdc_val), s1_idx)
            elif hmm is None:
                regime = 1   # always S1 when no HMM
            else:
                regime = None  # NaN rdc — skip entry/Rule4

            # ── Upward DC: Rule 4L / Rule 1L ─────────────────────────────
            if upturn_dc_arr[j]:
                if regime is None:
                    continue
                if regime == 2 and position == 1:
                    _close(p, abs_i, 'Rule4L_RegimeFilter', 'long')
                elif regime == 1 and position == 0:
                    position = 1
                    entry_price = p
                    entry_bar = abs_i
                    # C+GP target exit bar
                    if _predict is not None and t_dc0_arr is not None:
                        trough_bar = int(t_dc0_arr[j])
                        dc_len = abs_i - trough_bar
                        try:
                            pred = _predict(
                                dc_length=dc_len, dcc_price=p,
                                ext_end_price=float(pl_arr[j]),
                                prev_dcc_price=prev_dcc_price,
                                prev_has_os=prev_has_os,
                                cgpts_model=cgpts_model,
                            )
                            prev_dcc_price = p
                            prev_has_os = pred['trend_type'] == 'alpha_dc'
                            target_exit_bar = abs_i + pred['estimated_dce_offset']
                        except Exception:
                            target_exit_bar = abs_i + adaptation_step

            # ── Downward DC: Rule 4S / Rule 1S ───────────────────────────
            elif downturn_dc_arr[j]:
                if regime is None:
                    continue
                if regime == 2 and position == -1:
                    _close(p, abs_i, 'Rule4S_RegimeFilter', 'short')
                elif regime == 1 and position == 0:
                    position = -1
                    entry_price = p
                    entry_bar = abs_i

    # ── Close open position at end of period ──────────────────────────────
    if position != 0:
        p = price_arr[-1]
        side = 'long' if position == 1 else 'short'
        _close(p, n - 1, 'EndOfPeriod', side)

    gross_profit = sum(x for x in pnl_list if x > 0)
    gross_loss = abs(sum(x for x in pnl_list if x < 0))
    arr = np.array(pnl_list) if pnl_list else np.array([])
    std = float(arr.std()) if len(arr) >= 2 else 0.0
    sharpe = float(arr.mean() / std) if std > 1e-10 else 0.0

    return {
        'final_capital': capital,
        'cumulative_return': (capital - initial_capital) / initial_capital * 100.0,
        'max_drawdown': _compute_max_drawdown(capital_curve),
        'n_trades': n_trades,
        'n_winners': n_winners,
        'win_ratio': n_winners / n_trades if n_trades > 0 else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'sharpe': sharpe,
        'theta_history': theta_history,
        'trade_log': trade_log,
    }
