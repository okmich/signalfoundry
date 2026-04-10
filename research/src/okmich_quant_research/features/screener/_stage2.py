"""
Stage 2 — Temporal Stability (IC-IR)
=====================================
Filters out features that were historically predictive but are unstable over time.
A feature with high mean IC but large variance of IC is unreliable in live trading.

Two complementary checks:
  1. Full-sample IC-IR >= min_icir        (average predictiveness)
  2. IC hit rate    >= walk_forward_pct   (consistency across time)

IC-IR = mean(IC) / std(IC) over the rolling IC series.
IC hit rate = fraction of rolling windows where IC > 0 (pointed in the right direction).

A feature must pass BOTH checks to survive Stage 2.

For return prediction, IC is the rolling Spearman correlation between the feature
and forward returns. For regime classification, IC is the rolling point-biserial
or Cramer's V between the feature and regime labels.

This stage is unique to this framework — most standard feature selection pipelines
skip it, but for financial time series it is critical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pointbiserialr

from ._result import StageReport


def _rolling_spearman(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Spearman correlation between x and y."""
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]

    ic_vals = []
    idx = []
    for i in range(window, len(x) + 1):
        xi = x.iloc[i - window:i].dropna()
        yi = y.iloc[i - window:i].loc[xi.index].dropna()
        xi = xi.loc[yi.index]
        if len(xi) < window // 2:
            continue
        r, _ = spearmanr(xi.values, yi.values)
        ic_vals.append(r if not np.isnan(r) else 0.0)
        idx.append(x.index[i - 1])

    if not ic_vals:
        return pd.Series(dtype=float)
    return pd.Series(ic_vals, index=idx)


def _rolling_pointbiserial(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Rolling point-biserial correlation for regime tasks.
    Works with binary labels; for multi-class uses one-vs-rest average.
    """
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]
    classes = sorted(y.unique())

    ic_vals = []
    idx = []

    for i in range(window, len(x) + 1):
        xi = x.iloc[i - window:i].dropna()
        yi = y.iloc[i - window:i].loc[xi.index].dropna()
        xi = xi.loc[yi.index]
        if len(xi) < window // 2:
            continue

        # One-vs-rest average
        corrs = []
        for cls in classes:
            binary = (yi == cls).astype(float)
            if binary.sum() < 2 or (len(binary) - binary.sum()) < 2:
                continue
            r, _ = pointbiserialr(binary.values, xi.values)
            if not np.isnan(r):
                corrs.append(abs(r))

        ic_vals.append(np.mean(corrs) if corrs else 0.0)
        idx.append(x.index[i - 1])

    if not ic_vals:
        return pd.Series(dtype=float)
    return pd.Series(ic_vals, index=idx)


def _block_spearman(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Non-overlapping block Spearman IC.

    Divides the series into non-overlapping blocks of ``window`` bars and
    computes one Spearman correlation per block — O(n) instead of O(n × window).
    Consecutive sliding-window IC values share 251/252 data points and are
    not independent, so block IC gives equivalent IC-IR estimates with a
    fraction of the computation.
    """
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]

    n = len(x)
    n_blocks = n // window
    if n_blocks == 0:
        return pd.Series(dtype=float)

    ic_vals = []
    for i in range(n_blocks):
        xi = x.iloc[i * window:(i + 1) * window]
        yi = y.iloc[i * window:(i + 1) * window]
        if len(xi) < window // 2:
            continue
        r, _ = spearmanr(xi.values, yi.values)
        ic_vals.append(r if not np.isnan(r) else 0.0)

    return pd.Series(ic_vals, dtype=float)


def _block_pointbiserial(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """
    Non-overlapping block point-biserial IC for regime tasks.
    Same block strategy as ``_block_spearman`` — O(n) instead of O(n × window).
    """
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    classes = sorted(y.unique())

    n = len(x)
    n_blocks = n // window
    if n_blocks == 0:
        return pd.Series(dtype=float)

    ic_vals = []
    for i in range(n_blocks):
        xi = x.iloc[i * window:(i + 1) * window]
        yi = y.iloc[i * window:(i + 1) * window]
        if len(xi) < window // 2:
            continue

        corrs = []
        for cls in classes:
            binary = (yi == cls).astype(float)
            if binary.sum() < 2 or (len(binary) - binary.sum()) < 2:
                continue
            r, _ = pointbiserialr(binary.values, xi.values)
            if not np.isnan(r):
                corrs.append(abs(r))

        ic_vals.append(np.mean(corrs) if corrs else 0.0)

    return pd.Series(ic_vals, dtype=float)


def _icir(ic_series: pd.Series) -> float:
    """IC-IR = mean(IC) / std(IC). Returns 0 if std is zero."""
    if ic_series.empty or ic_series.std() < 1e-12:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def stage2_temporal_stability(X: pd.DataFrame, y: pd.Series, task: str = "return", window: int = 252,
                              min_icir: float | None = None, walk_forward_pct: float = 0.60,
                              icir_pct: float | None = None, wf_threshold: float | None = None,
                              use_block_ic: bool = False,
                              verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict[str, float]]:
    """
    Temporal stability filter via IC-IR and walk-forward IC hit rate.

    A feature must pass TWO checks to survive:

    1. Full-sample IC-IR >= min_icir
       Ensures the feature has a high enough signal-to-noise ratio on average.

    2. IC hit rate >= walk_forward_pct
       Ensures the feature points in the right direction consistently across time.
       IC hit rate = fraction of rolling windows where IC > 0.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Forward returns (task='return') or regime labels (task='regime').
    task : str
        ``"return"`` or ``"regime"``.
    window : int
        Rolling window length for IC computation.
    min_icir : float or None
        Minimum IC-IR to keep a feature. Defaults to 0.30 for returns, 0.20 for regime.
        Ignored when ``icir_pct`` is set.
    icir_pct : float or None
        If set (0–1), use this percentile of the observed IC-IR distribution as the
        threshold instead of ``min_icir``. E.g. 0.30 keeps features above the 30th
        percentile of IC-IR scores.
    wf_threshold : float or None
        IC value a block must exceed to count as a "positive" window in the hit-rate
        check. When None, defaults to 0.0 for return (signed Spearman) and 0.05 for
        regime (abs point-biserial). Set explicitly to 0.0 for binary direction labels
        screened via screen_for_regimes().
    use_block_ic : bool
        If True, use non-overlapping block IC (O(n), fast).
        If False, use bar-by-bar sliding-window IC (O(n × window), original). Default False.
    walk_forward_pct : float
        Minimum fraction of rolling windows where IC must exceed the per-task threshold.
        For ``task="return"`` (signed Spearman): IC must be > 0 (pointing right direction).
        For ``task="regime"`` (abs point-biserial, always >= 0): IC must be > 0.05.
        Default 0.60 — feature must be consistently predictive in at least 60% of windows.
        Set to 0.0 to disable and fall back to full-sample IC-IR only.
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    icir_scores : dict[str, float]
        IC-IR value for every feature that entered this stage.
    """
    if min_icir is None:
        min_icir = 0.30 if task == "return" else 0.20

    # Resolve wf_threshold: explicit override > task default
    _wf_threshold = wf_threshold if wf_threshold is not None else (0.0 if task == "return" else 0.05)

    n_before = X.shape[1]
    icir_scores: dict[str, float] = {}
    ic_series_map: dict[str, pd.Series] = {}

    for col in X.columns:
        if task == "return":
            ic = _block_spearman(X[col], y, window) if use_block_ic else _rolling_spearman(X[col], y, window)
        else:
            ic = _block_pointbiserial(X[col], y, window) if use_block_ic else _rolling_pointbiserial(X[col], y, window)
        icir_scores[col] = _icir(ic)
        ic_series_map[col] = ic

    # Adaptive IC-IR threshold
    if icir_pct is not None:
        min_icir = float(np.percentile(list(icir_scores.values()), icir_pct * 100))

    kept, removed = [], []
    for col in X.columns:
        score = icir_scores[col]
        ic = ic_series_map[col]

        if score < min_icir:
            removed.append(col)
            continue

        hit_rate = float((ic > _wf_threshold).mean()) if not ic.empty else 0.0
        if hit_rate >= walk_forward_pct:
            kept.append(col)
        else:
            removed.append(col)

    icir_label = f"IC-IR>{min_icir:.3f}({'pct' if icir_pct is not None else 'abs'})"
    ic_method = "block" if use_block_ic else "rolling"
    if verbose:
        print(f"  Stage 2 {icir_label}, hit-rate>={walk_forward_pct}, wf_thr={_wf_threshold} "
              f"(window={window}, ic={ic_method}): "
              f"{n_before} -> {len(kept)} features ({len(removed)} removed)")
        if kept:
            top = sorted(icir_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
            print(f"    Top IC-IR: {[(k, round(v, 3)) for k, v in top]}")

    report = StageReport(stage="Stage2_TemporalStability", n_before=n_before, n_after=len(kept), removed=removed)
    return X[kept], report, icir_scores