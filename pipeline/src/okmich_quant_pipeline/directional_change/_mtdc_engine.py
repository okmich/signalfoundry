"""
MTDC trading engine — K-threshold weighted consensus with ITA regime gate.

run_mtdc_algorithm  — bar-by-bar MTDC execution: K idc_parse streams →
                      per-threshold C+GP signals → consensus action → ITA gate → trade.

Both long and short sides are traded:
  Long  entry : consensus == 'buy'  AND ITA regime == S1 AND flat.
  Short entry : consensus == 'sell' AND ITA regime == S1 AND flat.
  Long  exits : consensus 'sell', C+GP target bar, S2 regime, end-of-period.
  Short exits : consensus 'buy',  S2 regime, end-of-period.
    (Short exits do not use C+GP target bar — STDC models are trained on upturn
     events only; a full symmetric implementation would require separate downturn
     C+GP models.)

Reference: Adegboye, Kampouridis & Otero (AI Review, 2023) Sections 3–4;
           Algorithms 4, 5, 6; Equation 3.
"""
import numpy as np
import pandas as pd

from okmich_quant_features.directional_change import idc_parse
from okmich_quant_ml.hmm import PomegranateHMM

from ._cgpts_model import predict_trend_end
from ._ita_sim import _compute_max_drawdown
from ._mtdc_thresholds import ConsensusMode
from ._rcd_hmm import get_current_regime


def run_mtdc_algorithm(prices: pd.Series, hmm: PomegranateHMM, s1_idx: int, stdc_models: list, thetas: list, weights: np.ndarray, alpha: float = 1.0, initial_capital: float = 10_000.0, consensus_mode: ConsensusMode | str = ConsensusMode.WEIGHT, majority_weight: float = 0.5) -> dict:
    """
    Execute the MTDC strategy (long + short) on a test price series.

    Runs K DC parsers simultaneously. At each bar, collects signals from all K
    STDC models, resolves them via the weighted consensus rule, applies the ITA
    S1/S2 regime gate, and executes trades on both sides.

    Long entry:  consensus == 'buy'  AND ITA regime == S1 AND flat.
    Short entry: consensus == 'sell' AND ITA regime == S1 AND flat.

    Long exit rules:
      C+GP exit       — current bar >= weighted target_exit_bar.
      Consensus exit  — consensus resolves to 'sell' while long.
      Regime exit     — ITA regime == S2 while long.
      End-of-period   — open position closed at final bar.

    Short exit rules:
      Consensus exit  — consensus resolves to 'buy' while short.
      Regime exit     — ITA regime == S2 while short.
      End-of-period   — open position closed at final bar.
      (No C+GP target bar for shorts — STDC models trained on upturn data only.)

    Consensus modes
    ---------------
    'weight' (default):
        buy  if weight_buy > weight_sell and weight_buy > 0
        sell if weight_sell > weight_buy
        hold if weight_buy == weight_sell == 0
    'majority':
        buy  if weight_buy / (weight_buy + weight_sell) > majority_weight
        sell if weight_sell / (weight_buy + weight_sell) > majority_weight
        hold otherwise (no side achieved supermajority)

    Parameters
    ----------
    prices : pd.Series
        Test price series.
    hmm : PomegranateHMM
        Fitted 2-state HMM (trained on the training window).
    s1_idx : int
        HMM state index for S1 (normal regime) from s1_state_index().
    stdc_models : list of dict
        K trained C+GP+TS models from train_cgpts_model().
    thetas : list of float
        DC thresholds corresponding to stdc_models.
    weights : np.ndarray
        Shape (k,), GA-evolved weights from train_ga_weights().
    alpha : float
        Asymmetric attenuation coefficient.
    initial_capital : float
        Starting capital.
    consensus_mode : ConsensusMode or str
        ConsensusMode.WEIGHT or ConsensusMode.MAJORITY.
    majority_weight : float
        Supermajority threshold for consensus_mode='majority' (default 0.5).

    Returns
    -------
    dict
        final_capital, cumulative_return, max_drawdown, n_trades, n_winners,
        win_ratio, profit_factor, sharpe, trade_log.
    """
    if len(stdc_models) != len(thetas):
        raise ValueError("stdc_models and thetas must have the same length.")
    consensus_mode = ConsensusMode(consensus_mode)

    k = len(stdc_models)
    weights = np.asarray(weights, dtype=float)

    # Pre-parse all K idc streams
    idc_list = [idc_parse(prices, thetas[i], alpha) for i in range(k)]
    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)

    upturn_dc = [idc_list[i]['upturn_dc'].values for i in range(k)]
    downturn_dc = [idc_list[i]['downturn_dc'].values for i in range(k)]
    pl_arr = [idc_list[i]['pl'].values for i in range(k)]
    t_dc0_arr = [idc_list[i]['t_dc0'].values for i in range(k)]
    rdc_arr_0 = idc_list[0]['rdc'].values   # best-theta stream for HMM queries

    # Per-threshold live C+GP state
    prev_dcc_price = [None] * k
    prev_has_os = [False] * k

    # Position state
    capital = float(initial_capital)
    position = 0   # 0=flat, 1=long, -1=short
    entry_price = 0.0
    entry_bar = 0
    target_exit_bar = 0
    capital_curve = [capital]
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

        # ── Collect per-threshold signals at this bar ─────────────────────
        buy_w = 0.0
        sell_w = 0.0
        buy_rev_weighted = 0.0   # numerator for weighted avg reversal bar

        for t in range(k):
            w = weights[t]
            if upturn_dc[t][i]:
                trough_bar = int(t_dc0_arr[t][i])
                dc_len = i - trough_bar
                pred = predict_trend_end(
                    dc_length=dc_len, dcc_price=p,
                    ext_end_price=float(pl_arr[t][i]),
                    prev_dcc_price=prev_dcc_price[t],
                    prev_has_os=prev_has_os[t],
                    cgpts_model=stdc_models[t],
                )
                prev_dcc_price[t] = p
                prev_has_os[t] = pred['trend_type'] == 'alpha_dc'
                if pred['trend_type'] == 'alpha_dc':
                    buy_w += w
                    buy_rev_weighted += w * (i + pred['estimated_dce_offset'])

            elif downturn_dc[t][i]:
                sell_w += w
                prev_dcc_price[t] = p
                prev_has_os[t] = False

        # ── Resolve consensus action ──────────────────────────────────────
        total_voted = buy_w + sell_w
        if consensus_mode == ConsensusMode.WEIGHT:
            if buy_w > sell_w and buy_w > 0:
                consensus = 'buy'
            elif sell_w > buy_w:
                consensus = 'sell'
            else:
                consensus = 'hold'
        else:  # MAJORITY
            if total_voted > 0:
                if buy_w / total_voted > majority_weight:
                    consensus = 'buy'
                elif sell_w / total_voted > majority_weight:
                    consensus = 'sell'
                else:
                    consensus = 'hold'
            else:
                consensus = 'hold'

        # ── Compute weighted target exit bar (for buy consensus) ──────────
        if buy_w > 0:
            weighted_target = int(round(buy_rev_weighted / buy_w))
        else:
            weighted_target = i

        # ── ITA regime query (uses best-theta RDC stream) ─────────────────
        rdc_val = rdc_arr_0[i]
        regime = get_current_regime(hmm, float(rdc_val), s1_idx) if np.isfinite(rdc_val) else None

        # ── Exit logic ────────────────────────────────────────────────────
        if position == 1:
            exit_rule = None
            if i >= target_exit_bar:
                exit_rule = 'MTDC_WeightedDCE'
            elif consensus == 'sell':
                exit_rule = 'ConsensusExit_Sell'
            elif regime == 2:
                exit_rule = 'RegimeExit_S2'
            if exit_rule is not None:
                _close(p, i, exit_rule, 'long')

        elif position == -1:
            exit_rule = None
            if consensus == 'buy':
                exit_rule = 'ConsensusExit_Buy'
            elif regime == 2:
                exit_rule = 'RegimeExit_S2'
            if exit_rule is not None:
                _close(p, i, exit_rule, 'short')

        # ── Entry logic ───────────────────────────────────────────────────
        if position == 0 and regime == 1:
            if consensus == 'buy':
                position = 1
                entry_price = p
                entry_bar = i
                target_exit_bar = weighted_target
            elif consensus == 'sell':
                position = -1
                entry_price = p
                entry_bar = i

    # ── Close open position at end of period ──────────────────────────────
    if position != 0:
        p = price_arr[-1]
        side = 'long' if position == 1 else 'short'
        _close(p, n - 1, 'EndOfPeriod', side)

    gross_profit = sum(x for x in pnl_list if x > 0)
    gross_loss = abs(sum(x for x in pnl_list if x < 0))

    return {
        'final_capital': capital,
        'cumulative_return': (capital - initial_capital) / initial_capital * 100.0,
        'max_drawdown': _compute_max_drawdown(capital_curve),
        'n_trades': n_trades,
        'n_winners': n_winners,
        'win_ratio': n_winners / n_trades if n_trades > 0 else 0.0,
        'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        'sharpe': _compute_sharpe_from_trades(pnl_list),
        'trade_log': trade_log,
    }


def _compute_sharpe_from_trades(pnl_list: list) -> float:
    if len(pnl_list) < 2:
        return 0.0
    arr = np.array(pnl_list)
    std = arr.std()
    return float(arr.mean() / std) if std > 1e-10 else 0.0
