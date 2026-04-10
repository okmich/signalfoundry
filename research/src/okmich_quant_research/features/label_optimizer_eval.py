"""
Unified evaluation framework for supervised label optimizer scripts.

Combines two tiers of metrics:
- Economic quality (70%): How profitable, persistent, and well-structured are the causal labels?
- Agreement quality (30%): How well do causal labels agree with a forward-looking guide label?

Used by all supervised labeller optimizers to ensure consistent, comparable scoring across CTL, CTL-MA, Z-Score, and Trend Persistence.
"""

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import cohen_kappa_score, f1_score

from okmich_quant_labelling.utils.label_eval_util import evaluate_regime_returns_potentials
from okmich_quant_labelling.utils.label_util import map_label_to_trend_direction


# =========================================================================
# Agreement metrics (causal vs forward-looking guide label)
# =========================================================================

def compute_agreement_metrics(causal_labels: pd.Series, forward_labels: pd.Series) -> Optional[Dict[str, float]]:
    """
    Compute agreement between causal and forward-looking labels.

    Trimmed to two non-redundant metrics:
    - Cohen's Kappa: chance-corrected classification agreement
    - Macro F1: harmonic mean of precision/recall across classes

    Returns None if insufficient data.
    """
    mask = ~(pd.isna(causal_labels) | pd.isna(forward_labels))
    causal = causal_labels[mask].values
    forward = forward_labels[mask].values

    if len(causal) == 0 or len(forward) == 0:
        return None

    try:
        return {
            "cohen_kappa": cohen_kappa_score(causal, forward),
            "macro_f1": f1_score(forward, causal, average="macro", zero_division=0),
        }
    except Exception:
        return None


# =========================================================================
# Economic quality metrics (standalone quality of the causal label)
# =========================================================================

MIN_REGIME_COUNT = 200
MAX_REGIME_COUNT = 10_000  # floor — actual cap scales with dataset length
MAX_REGIME_FRACTION = 0.02  # max 2% of bars can be regime switches
MAX_AVG_DWELL = 500


def compute_economic_metrics(df: pd.DataFrame, label_col: str = "label", progressive_skip: int = 2,
                             whipsaw_cost: float = 0.0002, min_regime_count: int = MIN_REGIME_COUNT,
                             max_regime_count: int = MAX_REGIME_COUNT, max_regime_fraction: float = MAX_REGIME_FRACTION,
                             max_avg_dwell: int = MAX_AVG_DWELL) -> Optional[Dict[str, float]]:
    """
    Compute economic quality of causal labels using regime returns analysis.

    Evaluates how profitable, persistent, and well-structured the labelled
    regimes are when used as trading signals with realistic execution costs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'open', 'close' columns and the label column.
    label_col : str
        Column containing the causal labels.
    progressive_skip : int
        Number of bars to skip at regime entry (execution lag).
    whipsaw_cost : float
        Transaction cost per regime change.
    min_regime_count : int
        Minimum number of regime changes required. Rejects degenerate solutions
        that barely trade (e.g., 3 switches over 400k bars).
    max_regime_fraction : float
        Maximum fraction of bars that can be regime switches. The effective
        upper cap is max(max_regime_count, len(df) * max_regime_fraction),
        adapting to datasets of any length.
    max_avg_dwell : int
        Cap on avg_dwell when computing persistence_score. Prevents inflated
        scores from labels that almost never switch. persistence_score is
        recomputed as min(avg_dwell, max_avg_dwell) * win_rate.

    Returns None if evaluation fails or regime count is below minimum.
    """
    required = [label_col, "close", "open"]
    if any(c not in df.columns for c in required):
        return None

    if len(df[label_col].dropna().value_counts()) < 2:
        return None

    try:
        df_eval = df.copy()
        df_eval["log_return"] = np.log(df_eval["close"] / df_eval["close"].shift(1))

        # Early exit: count regime changes cheaply before expensive evaluation
        regime_switches = (df_eval[label_col].diff().fillna(0) != 0).sum()
        effective_max = max(max_regime_count, int(len(df_eval) * max_regime_fraction))
        if regime_switches < min_regime_count or regime_switches > effective_max:
            return None

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module=r"okmich_quant_labelling\.utils\.label_util")
            label_sign_map = map_label_to_trend_direction(df_eval, state_col=label_col, return_col="log_return",
                                                          method="simple")

        metrics_df, _ = evaluate_regime_returns_potentials(df_eval, label_col=label_col, progressive_skip=progressive_skip,
                                                           whipsaw_cost=whipsaw_cost, label_sign_map=label_sign_map,
                                                           include_overall=True)

        if metrics_df is None:
            return None

        overall = metrics_df[metrics_df["label"] == "overall"]
        if len(overall) == 0:
            return None

        row = overall.iloc[0]

        # Reject degenerate solutions with too few regime changes
        regime_count = int(row.get("count", 0))
        if regime_count < min_regime_count:
            return None

        # Cap avg_dwell to prevent inflated persistence_score
        avg_dwell = float(row.get("avg_dwell", 0))
        win_rate = float(row.get("win_rate", 0))
        capped_persistence = min(avg_dwell, max_avg_dwell) * win_rate

        # Cap profit_factor — inf means zero losses which is an artifact of few trades
        profit_factor = float(row.get("profit_factor", 0))
        if not np.isfinite(profit_factor):
            profit_factor = 0.0

        return {
            "profit_factor": profit_factor,
            "win_rate": win_rate,
            "persistence_score": capped_persistence,
            "regime_purity": float(row.get("regime_purity", 0)),
            "sharpe_ratio": float(row.get("sharpe_ratio", 0)),
            "cumulative_consistency_return": float(
                row.get("cumulative_consistency_return", 0)
            ),
        }
    except Exception:
        return None


# =========================================================================
# Composite scoring
# =========================================================================

ECONOMIC_METRICS = ["profit_factor", "win_rate", "persistence_score", "regime_purity", "sharpe_ratio"]
AGREEMENT_METRICS = ["cohen_kappa", "macro_f1"]

DEFAULT_SCORING_WEIGHTS = {"economic_weight": 0.70, "agreement_weight": 0.30}


def compute_composite_score(results_df: pd.DataFrame, economic_metrics: List[str] = None,
                            agreement_metrics: List[str] = None, scoring_weight: dict[str, float] = None) -> pd.Series:
    """
    Compute unified composite score from economic and agreement metrics.

    Each metric within a tier is min-max normalized, then averaged within its tier.
    The two tier averages are combined using the specified weights.

    Parameters
    ----------
    results_df : pd.DataFrame
        DataFrame with one row per parameter combination, containing both
        economic and agreement metric columns.
    economic_metrics : list of str
        Economic metric column names (default: ECONOMIC_METRICS).
    agreement_metrics : list of str
        Agreement metric column names (default: AGREEMENT_METRICS).
    economic_weight : float
        Weight for economic tier (default: 0.70).
    agreement_weight : float
        Weight for agreement tier (default: 0.30).

    Returns
    -------
    pd.Series : composite score for each row
    """
    if economic_metrics is None:
        economic_metrics = ECONOMIC_METRICS
    if agreement_metrics is None:
        agreement_metrics = AGREEMENT_METRICS

    sw = scoring_weight or DEFAULT_SCORING_WEIGHTS

    df = results_df.copy()

    # Handle infinite profit_factor — cap at a reasonable maximum
    if "profit_factor" in df.columns:
        df["profit_factor"] = df["profit_factor"].replace([np.inf, -np.inf], np.nan)
        max_finite = df["profit_factor"].max()
        df["profit_factor"] = df["profit_factor"].fillna(max_finite if pd.notna(max_finite) else 0)

    # Min-max normalize each metric
    def normalize_column(col: pd.Series) -> pd.Series:
        min_val = col.min()
        max_val = col.max()
        if max_val > min_val:
            return (col - min_val) / (max_val - min_val)
        return pd.Series(1.0, index=col.index)

    # Compute tier scores
    econ_cols = [m for m in economic_metrics if m in df.columns]
    agree_cols = [m for m in agreement_metrics if m in df.columns]

    if not econ_cols and not agree_cols:
        return pd.Series(0.0, index=df.index)

    econ_score = pd.Series(0.0, index=df.index)
    if econ_cols:
        for col in econ_cols:
            econ_score += normalize_column(df[col])
        econ_score /= len(econ_cols)

    agree_score = pd.Series(0.0, index=df.index)
    if agree_cols:
        for col in agree_cols:
            agree_score += normalize_column(df[col])
        agree_score /= len(agree_cols)

    # Weighted combination
    total_weight = 0.0
    composite = pd.Series(0.0, index=df.index)

    if econ_cols:
        composite += sw["economic_weight"] * econ_score
        total_weight += sw["economic_weight"]
    if agree_cols:
        composite += sw["agreement_weight"] * agree_score
        total_weight += sw["agreement_weight"]

    if total_weight > 0:
        composite /= total_weight

    return composite


def evaluate_label_params(df: pd.DataFrame, causal_labels: pd.Series, forward_labels: pd.Series,
                          label_col: str = "label", progressive_skip: int = 2,
                          whipsaw_cost: float = 0.0002) -> Optional[Dict[str, float]]:
    """
    Single-call evaluation combining economic and agreement metrics.

    This is the main entry point for supervised optimizers. It computes all metrics for one parameter combination.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'open', 'close' columns.
    causal_labels : pd.Series
        Labels generated by the causal labelling function.
    forward_labels : pd.Series
        Forward-looking guide labels (e.g., tri_amp_label, bi_ctl_label).
    label_col : str
        Temporary column name for causal labels in df.
    progressive_skip : int
        Execution lag in bars.
    whipsaw_cost : float
        Transaction cost per regime change.

    Returns None if evaluation fails for either tier.
    """
    # Agreement metrics
    agreement = compute_agreement_metrics(causal_labels, forward_labels)
    if agreement is None:
        return None

    # Economic metrics — add causal labels to df temporarily
    df_eval = df.copy()
    df_eval[label_col] = causal_labels.values if isinstance(causal_labels, pd.Series) else causal_labels

    economic = compute_economic_metrics(df_eval, label_col=label_col, progressive_skip=progressive_skip, whipsaw_cost=whipsaw_cost)
    if economic is None:
        return None

    return {**economic, **agreement}
