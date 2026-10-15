"""
Stage 2 — Temporal Stability (IC-IR)
=====================================
Filters out features that were historically predictive but are unstable over time.
A feature with high mean IC but large variance of IC is unreliable in live trading.

Two complementary checks:
  1. Full-sample |IC-IR| >= min_icir        (average predictiveness, either sign)
  2. IC hit rate    >= walk_forward_pct     (consistency across time)

IC-IR = mean(IC) / std(IC) over the rolling IC series.
IC hit rate = fraction of rolling windows whose IC points the SAME way as the feature's mean IC.

Sign-agnostic by default: a feature whose IC is reliably NEGATIVE (a reversal / contrarian feature) is exactly as
informative as one whose IC is reliably positive — the model only has to flip it. Scoring the signed IC-IR and a
hit rate of IC > 0, as this stage once did, removed every reliable reversal feature however stable it was. The
signed IC-IR is still returned, so the direction is reported. ``sign_agnostic=False`` restores the old rule.

A feature must pass BOTH checks to survive Stage 2.

For return prediction, IC is the rolling Spearman correlation between the feature
and forward returns. For regime classification, IC is the SIGNED one-vs-rest point-biserial correlation between
the feature and each class indicator 1{y == c}. Each class gets its own IC series and IC-IR; the feature is scored
on the class c* whose |IC-IR| is largest, and its hit rate is taken on that class's oriented series. The signed
per-class IC has mean 0 when the feature is unrelated to the labels, so the IC-IR gate means the same thing on both
tasks. (The regime IC used to be the mean of |point-biserial| over classes. That is >= 0 by construction, so pure
noise scored IC-IR ~0.7 against persistent labels and the regime IC-IR gate never removed anything.)

This stage is unique to this framework — most standard feature selection pipelines
skip it, but for financial time series it is critical.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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


def _class_pointbiserial(xi: pd.Series, yi: pd.Series, classes: list) -> list[float]:
    """Signed one-vs-rest point-biserial r of ``xi`` against 1{yi == c}, per class.

    NaN for a class with fewer than 2 members on either side of the split in this window (or a constant ``xi``).
    Closed form r = (mean(x | c) - mean(x | not c)) * sqrt(p (1 - p)) / std(x), identical to
    ``scipy.stats.pointbiserialr`` and far cheaper per call.
    """
    x = np.asarray(xi, dtype=float)
    labels = np.asarray(yi)
    n = x.size
    sx = x.std()
    corrs = []
    for cls in classes:
        member = labels == cls
        k = int(member.sum())
        if k < 2 or n - k < 2 or sx == 0:
            corrs.append(np.nan)
            continue
        p = k / n
        corrs.append(float((x[member].mean() - x[~member].mean()) * np.sqrt(p * (1 - p)) / sx))
    return corrs


def _regime_classes(y: pd.Series) -> list:
    return sorted(y.dropna().unique())


def _rolling_pointbiserial(x: pd.Series, y: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling signed point-biserial correlation for regime tasks: one column per class (one-vs-rest), NaN where a
    class is missing from the window.
    """
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]
    classes = _regime_classes(y)

    ic_rows = []
    idx = []

    for i in range(window, len(x) + 1):
        xi = x.iloc[i - window:i].dropna()
        yi = y.iloc[i - window:i].loc[xi.index].dropna()
        xi = xi.loc[yi.index]
        if len(xi) < window // 2:
            continue
        ic_rows.append(_class_pointbiserial(xi, yi, classes))
        idx.append(x.index[i - 1])

    return pd.DataFrame(ic_rows, index=idx, columns=classes, dtype=float)


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


def _block_pointbiserial(x: pd.Series, y: pd.Series, window: int) -> pd.DataFrame:
    """
    Non-overlapping block signed point-biserial IC for regime tasks: one row per block, one column per class
    (one-vs-rest), NaN where a class is missing from the block.
    Same block strategy as ``_block_spearman`` — O(n) instead of O(n × window).
    """
    common = x.index.intersection(y.index)
    x = x.loc[common]
    y = y.loc[common]

    mask = x.notna() & y.notna()
    x = x[mask]
    y = y[mask]
    classes = _regime_classes(y)

    n = len(x)
    n_blocks = n // window
    if n_blocks == 0:
        return pd.DataFrame(columns=classes, dtype=float)

    ic_rows = []
    for i in range(n_blocks):
        xi = x.iloc[i * window:(i + 1) * window]
        yi = y.iloc[i * window:(i + 1) * window]
        if len(xi) < window // 2:
            continue
        ic_rows.append(_class_pointbiserial(xi, yi, classes))

    return pd.DataFrame(ic_rows, columns=classes, dtype=float)


def _best_class_ic(ic_by_class: pd.DataFrame, sign_agnostic: bool) -> tuple[pd.Series, float, object]:
    """Pick the class whose IC series scores best; return (its valid IC series, its signed IC-IR, the class).

    A class has an IC only in the windows where it has >= 2 members on each side. With persistent labels that is a
    minority of the windows, and the null spread of an IC-IR is 1/sqrt(n_valid), so a raw per-class IC-IR would
    let noise through a fixed gate far more often than on the return task. Each class's IC-IR is therefore put on
    the full-window scale: IC-IR_c * sqrt(n_valid_c / n_windows) = t_c / sqrt(n_windows), exactly what the return
    task's IC-IR is (every window valid). Measured with block windows of 252 bars on 20k-40k bars: 40 iid-noise
    features pass the 0.20 gate 14/40 unscaled (3 states, mean dwell 200) and 1-5/40 scaled (2-3 states, mean
    dwell 50-400). On ~120 windows 0.20 is a ~2-sigma gate per class: real, but not tight.

    Sign-agnostic: largest |IC-IR|. Signed rule: largest signed IC-IR. Ties go to the first class in sorted order,
    so the choice is deterministic (with two classes the indicators are complements and the IC-IRs mirror).
    """
    n_windows = len(ic_by_class)
    best_cls, best_icir, best_key = None, 0.0, -np.inf
    for cls in ic_by_class.columns:
        valid = ic_by_class[cls].dropna()
        icir = _icir(valid) * np.sqrt(len(valid) / n_windows) if n_windows else 0.0
        key = abs(icir) if sign_agnostic else icir
        if key > best_key:
            best_cls, best_icir, best_key = cls, icir, key
    if best_cls is None:
        return pd.Series(dtype=float), 0.0, None
    return ic_by_class[best_cls].dropna(), best_icir, best_cls


def _icir(ic_series: pd.Series) -> float:
    """IC-IR = mean(IC) / std(IC). Returns 0 if std is zero."""
    if ic_series.empty or ic_series.std() < 1e-12:
        return 0.0
    return float(ic_series.mean() / ic_series.std())


def stage2_temporal_stability(X: pd.DataFrame, y: pd.Series, task: str = "return", window: int = 252,
                              min_icir: float | None = None, walk_forward_pct: float = 0.60,
                              icir_pct: float | None = None, wf_threshold: float | None = None,
                              use_block_ic: bool = False, sign_agnostic: bool = True,
                              verbose: bool = True) -> tuple[pd.DataFrame, StageReport, dict[str, float]]:
    """
    Temporal stability filter via IC-IR and walk-forward IC hit rate.

    A feature must pass TWO checks to survive:

    1. Full-sample |IC-IR| >= min_icir
       Ensures the feature has a high enough signal-to-noise ratio on average, in either direction.

    2. IC hit rate >= walk_forward_pct
       Ensures the feature points the same way consistently across time.
       IC hit rate = fraction of rolling windows where sign(mean IC) · IC > wf_threshold.

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
        IC value a block must exceed (after orientation) to count as a hit in the hit-rate check. When None,
        defaults to 0.0 for both tasks: the return IC (Spearman) and the regime IC (signed per-class
        point-biserial) are both centred on 0 under no relationship.
    use_block_ic : bool
        If True, use non-overlapping block IC (O(n), fast).
        If False, use bar-by-bar sliding-window IC (O(n × window), original). Default False here;
        FeatureScreener passes True.
    sign_agnostic : bool
        If True (default), both checks are applied to the feature's IC oriented by the sign of its mean IC, so a
        stable negative-IC feature passes like a stable positive-IC one. The returned IC-IR keeps its sign.
        If False, the original signed rule: IC-IR >= min_icir and IC > threshold. On ``task="regime"`` the class
        c* is then the one with the largest SIGNED IC-IR (the feature must be reliably HIGH in some class) instead
        of the largest |IC-IR|.
    walk_forward_pct : float
        Minimum fraction of IC windows (blocks) where the oriented IC must exceed ``wf_threshold``.
        On ``task="regime"`` the windows are those where class c* has at least 2 members on each side.
        Default 0.60 — feature must be consistently predictive in at least 60% of windows.
        Set to 0.0 to disable and fall back to full-sample IC-IR only.
    verbose : bool

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
        ``detail["regime_class"]`` (regime task) maps each feature to the class c* it was scored on.
    icir_scores : dict[str, float]
        IC-IR value for every feature that entered this stage (regime task: the signed IC-IR of class c*).
    """
    if min_icir is None:
        min_icir = 0.30 if task == "return" else 0.20

    # Resolve wf_threshold: explicit override > task default (0.0 for both tasks; every IC here is signed)
    _wf_threshold = wf_threshold if wf_threshold is not None else 0.0

    n_before = X.shape[1]
    icir_scores: dict[str, float] = {}
    ic_series_map: dict[str, pd.Series] = {}
    regime_class: dict[str, object] = {}

    for col in X.columns:
        if task == "return":
            ic = _block_spearman(X[col], y, window) if use_block_ic else _rolling_spearman(X[col], y, window)
            icir_scores[col] = _icir(ic)
        else:
            ic_by_class = (_block_pointbiserial(X[col], y, window) if use_block_ic
                           else _rolling_pointbiserial(X[col], y, window))
            ic, icir_scores[col], regime_class[col] = _best_class_ic(ic_by_class, sign_agnostic)
        ic_series_map[col] = ic

    # Orientation: +1 for every feature under the signed rule, else the sign of its IC-IR (0 IC-IR counts as +1).
    orientation = {col: (1.0 if (not sign_agnostic or icir_scores[col] >= 0) else -1.0) for col in X.columns}
    oriented_icir = {col: orientation[col] * icir_scores[col] for col in X.columns}

    # Adaptive IC-IR threshold (on the oriented scores, so the percentile ranks strength, not sign)
    if icir_pct is not None:
        min_icir = float(np.percentile(list(oriented_icir.values()), icir_pct * 100))

    kept, removed = [], []
    for col in X.columns:
        score = oriented_icir[col]
        ic = orientation[col] * ic_series_map[col]

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
            top = sorted(icir_scores.items(), key=lambda kv: oriented_icir[kv[0]], reverse=True)[:5]
            print(f"    Top IC-IR (signed): {[(k, round(v, 3)) for k, v in top]}")

    detail = {"sign_agnostic": sign_agnostic, "negative_ic_kept": [c for c in kept if orientation[c] < 0]}
    if task == "regime":
        detail["regime_class"] = regime_class
    report = StageReport(stage="Stage2_TemporalStability", n_before=n_before, n_after=len(kept), removed=removed,
                         detail=detail)
    return X[kept], report, icir_scores