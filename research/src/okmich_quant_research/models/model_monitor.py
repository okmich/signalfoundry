"""
ModelMonitor — rolling health monitoring for deployed models.

Two monitoring modes:
- Regression (return prediction): rolling IC, IC-IR, Ljung-Box.
- Regime classification: rolling persistence, regime-conditioned Sharpe.

Returns a ModelHealthStatus enum and a metrics dict on each call.
"""

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.diagnostic import acorr_ljungbox

class ModelHealthStatus(str, Enum):
    """Operational state of a deployed model."""
    HEALTHY   = "HEALTHY"    # All metrics within bounds — trade normally.
    WARNING   = "WARNING"    # One metric degraded — monitor closely.
    DEGRADED  = "DEGRADED"   # Primary metric below threshold — schedule retrain.
    STALE     = "STALE"      # No recent predictions — model not receiving data.
    INVERTED  = "INVERTED"   # Primary metric has reversed sign — stop; retrain.


class ModelMonitor:
    """
    Rolling health monitor for regression and regime classification models.

    Parameters
    ----------
    ic_window : int, default=60
        Rolling window for IC computation (regression).
    persistence_window : int, default=20
        Rolling window for regime persistence computation.
    sharpe_window : int, default=60
        Rolling window for regime-conditioned Sharpe.
    ic_warning_threshold : float, default=0.02
        IC below this triggers WARNING.
    ic_degraded_threshold : float, default=0.0
        IC below this triggers DEGRADED.
    ic_ir_degraded_threshold : float, default=0.0
        IC-IR below this triggers DEGRADED.
    persistence_min : float, default=2.0
        Mean persistence below this triggers WARNING (regime flipping).
    sharpe_degraded_threshold : float, default=0.2
        Regime-conditioned Sharpe below this triggers DEGRADED.
    bars_per_year : int, default=72576
        Used to annualise regime-conditioned Sharpe (252 * 288 for 5-min bars).
    """

    def __init__(self, ic_window: int = 60, persistence_window: int = 20, sharpe_window: int = 60,
                 ic_warning_threshold: float = 0.02, ic_degraded_threshold: float = 0.0,
                 ic_ir_degraded_threshold: float = 0.0, persistence_min: float = 2.0,
                 sharpe_degraded_threshold: float = 0.2, bars_per_year: int = 252 * 288):
        self.ic_window = ic_window
        self.persistence_window = persistence_window
        self.sharpe_window = sharpe_window
        self.ic_warning_threshold = ic_warning_threshold
        self.ic_degraded_threshold = ic_degraded_threshold
        self.ic_ir_degraded_threshold = ic_ir_degraded_threshold
        self.persistence_min = persistence_min
        self.sharpe_degraded_threshold = sharpe_degraded_threshold
        self.bars_per_year = bars_per_year

    # ------------------------------------------------------------------
    # Regression health
    # ------------------------------------------------------------------

    def check_regression_health(self, predictions: pd.Series, actuals: pd.Series) -> Tuple[ModelHealthStatus, Dict[str, Any]]:
        """
        Assess health of a return-prediction model.

        Parameters
        ----------
        predictions : pd.Series
            Rolling predictions (most recent = last element).
        actuals : pd.Series
            Realised returns aligned to predictions.

        Returns
        -------
        status : ModelHealthStatus
        metrics : dict
            ic, ic_ir, ljung_box_pvalue, n_obs.
        """
        aligned = pd.concat(
            [predictions.rename("pred"), actuals.rename("actual")], axis=1
        ).dropna()

        if len(aligned) < 10:
            return ModelHealthStatus.STALE, {"n_obs": len(aligned)}

        # Rolling IC over last ic_window bars
        window = aligned.tail(self.ic_window)
        ic, ic_pvalue = scipy_stats.spearmanr(window["pred"], window["actual"])

        # IC-IR: per-bar rolling Spearman over sub-windows
        sub_ics = []
        step = max(1, self.ic_window // 10)
        for start in range(0, len(aligned) - step, step):
            chunk = aligned.iloc[start: start + step]
            if len(chunk) >= 5:
                c, _ = scipy_stats.spearmanr(chunk["pred"], chunk["actual"])
                if not np.isnan(c):
                    sub_ics.append(c)
        ic_ir = (np.mean(sub_ics) / np.std(sub_ics, ddof=1)) if len(sub_ics) >= 2 and np.std(sub_ics, ddof=1) > 0 else np.nan

        # Ljung-Box on recent errors — adaptive lag to avoid crash on short series
        errors = (aligned["actual"] - aligned["pred"]).tail(self.ic_window)
        max_safe_lag = max(1, len(errors) // 2 - 1)
        lb_lag = min(10, max_safe_lag)
        if len(errors) < 4 or lb_lag < 1:
            lb_pvalue = np.nan
        else:
            lb_result = acorr_ljungbox(errors, lags=lb_lag, return_df=True)
            lb_pvalue = float(lb_result["lb_pvalue"].iloc[-1])

        metrics = {
            "ic": float(ic) if not np.isnan(ic) else np.nan,
            "ic_pvalue": float(ic_pvalue) if not np.isnan(ic_pvalue) else np.nan,
            "ic_ir": float(ic_ir) if not np.isnan(ic_ir) else np.nan,
            "ljung_box_pvalue": lb_pvalue,
            "n_obs": len(aligned),
        }

        # Determine status — IC-IR degradation takes precedence over IC warning
        if not np.isnan(ic):
            if ic < 0 and abs(ic) > self.ic_warning_threshold:
                return ModelHealthStatus.INVERTED, metrics
            if ic < self.ic_degraded_threshold:
                return ModelHealthStatus.DEGRADED, metrics
            if not np.isnan(ic_ir) and ic_ir < self.ic_ir_degraded_threshold:
                return ModelHealthStatus.DEGRADED, metrics
            if ic < self.ic_warning_threshold:
                return ModelHealthStatus.WARNING, metrics
        return ModelHealthStatus.HEALTHY, metrics

    # ------------------------------------------------------------------
    # Regime classification health
    # ------------------------------------------------------------------

    def check_regime_health(self, labels: pd.Series,
                            returns: Optional[pd.Series] = None) -> Tuple[ModelHealthStatus, Dict[str, Any]]:
        """
        Assess health of a regime classification model.

        Parameters
        ----------
        labels : pd.Series
            Recent predicted regime labels.
        returns : pd.Series, optional
            Realised returns aligned to labels.  Required for Sharpe check.

        Returns
        -------
        status : ModelHealthStatus
        metrics : dict
            regime_persistence, regime_conditioned_sharpe, n_obs.
        """
        valid_labels = labels.dropna()
        if len(valid_labels) < 10:
            return ModelHealthStatus.STALE, {"n_obs": len(valid_labels)}

        # Persistence — use last persistence_window labels
        window_labels = valid_labels.tail(self.persistence_window)
        vals = window_labels.values
        run_lengths = []
        run = 1
        for i in range(1, len(vals)):
            if vals[i] == vals[i - 1]:
                run += 1
            else:
                run_lengths.append(run)
                run = 1
        run_lengths.append(run)
        persistence = float(np.mean(run_lengths))

        metrics: Dict[str, Any] = {
            "regime_persistence": persistence,
            "n_obs": len(valid_labels),
        }

        # Regime-conditioned Sharpe
        worst_sharpe = np.nan
        if returns is not None:
            window_returns = returns.tail(self.sharpe_window)
            aligned = pd.concat(
                [window_labels.rename("label"), window_returns.rename("ret")], axis=1
            ).dropna()
            sharpe_by_regime: Dict[Any, float] = {}
            for regime, group in aligned.groupby("label"):
                rets = group["ret"]
                if len(rets) >= 10 and rets.std() > 0:
                    sharpe_by_regime[regime] = float(
                        rets.mean() / rets.std() * np.sqrt(self.bars_per_year)
                    )
            metrics["regime_conditioned_sharpe"] = sharpe_by_regime
            if sharpe_by_regime:
                worst_sharpe = min(sharpe_by_regime.values())

        # Determine status
        if persistence < self.persistence_min:
            return ModelHealthStatus.WARNING, metrics
        if not np.isnan(worst_sharpe):
            if worst_sharpe < -self.sharpe_degraded_threshold:
                return ModelHealthStatus.INVERTED, metrics
            if worst_sharpe < self.sharpe_degraded_threshold:
                return ModelHealthStatus.DEGRADED, metrics
        return ModelHealthStatus.HEALTHY, metrics