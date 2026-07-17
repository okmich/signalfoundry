"""
Stage 0b — Persistence Diagnostic / Filter (HMM-specific, OPT-IN)
================================================================
Scores how much persistent structure a candidate feature carries in its own marginals:

    score(f) = max(|acf1(x)|, |acf1(|x|)|, |acf1(x^2)|)

A feature scoring ~0 is indistinguishable from noise in all three marginals. Such a feature cannot by
itself describe a persistent regime, and in a joint emission it can drag the MAP into a coin flip.

Default is DIAGNOSTIC (``min_persistence=0.0``): the score is computed and reported, nothing is removed.
Removal is opt-in, and deliberately so — see the limits below.

WHAT THIS TEST CANNOT SEE  (why it must not be a global reject by default)
-------------------------------------------------------------------------
The score is *per-feature* and *marginal*; the emission is *joint* and *shape-aware*. It is therefore
blind to at least two regime structures the model genuinely uses, both verified empirically:

  * **Covariance regimes.** Two features whose marginals are white but whose *correlation* switches
    persistently. Constructed case: both marginals score ~0.016 (rejected) while their rolling
    correlation has acf 0.999. ``hmm_pmgnt`` uses full covariance by default, so this is a live case.
  * **Tail-shape / mixture regimes.** A feature that is normal in regime A and heavy-tailed in regime B
    with EQUAL variance. Constructed case (dwell 200 bars, kurtosis -0.04 vs 13.11): score 0.016 —
    rejected — even though a Lambda emission fits a per-state tail parameter and can separate them.
    Adding moments does not rescue it: acf(x^4) on that series is -0.003.

A model-aware gate (judge the *subset's* resulting regime, or validate incrementally / out-of-sample)
is the correct design. Until such a gate exists and has been validated cross-instrument and
walk-forward, prefer this as a diagnostic and enable removal only where the calibration is known to hold.

SAMPLE-SIZE LIMIT
-----------------
The score is an autocorrelation estimate, so its noise floor scales ~1/sqrt(n) and the max over three
statistics inflates it further. Monte-Carlo on IID normal noise, % surviving a 0.15 floor:
    n=60 -> 47%,  n=120 -> 21%,  n=200 -> 8.6%,  n=600 -> 0.1%,  n=4000 -> 0%.
So a fixed threshold is meaningless on short samples. ``min_obs`` (default 600) refuses to reject a
feature whose sample cannot support the call; those features are KEPT and named in the report.
Conversely, on very large n almost any non-zero acf is "statistically significant", which is why the
criterion is an effect-size floor rather than a significance test.

CALIBRATION (do not treat as universal)
---------------------------------------
``min_persistence=0.15`` was calibrated on ONE dataset: FXPIG M5, 25 symbols, 322 feature-instances
(2026-07-17). There, scores were cleanly bimodal — coins 0.009-0.069 vs the lowest legitimate feature
(``momentum.roc_velocity``) at 0.210 — so 0.10/0.15/0.20 behaved identically and 0.25 began
false-positiving. That gap is a property of that dataset/timeframe, NOT a law. It has not been validated
on other instruments, timeframes, or out-of-sample.

Supporting evidence there (controlled refit, USDCAD liquidity, hmm_lambda, reproduced the shipped fit
exactly): dropping ``order_flow.vir_zscore`` (score 0.016) moved median dwell 2 -> 16 bars, flip
0.308 -> 0.060, mean top_prob 0.813 -> 0.926, all states still occupied, no lag paid.

Deliberately NOT in the shared ``features/screener/_stage0``: persistence only matters for regime models.
A tree/GBM screener may legitimately use a white-noise feature as a per-bar predictor.

This is a FLOOR, never an objective. Do not score *toward* persistence: the persistence of window-based
features is largely manufactured by their lookback, so maximising it just buys dwell with lag.
"""
from __future__ import annotations

import pandas as pd

from ..screener._result import StageReport

DEFAULT_MIN_OBS = 600


def adjacent_pair_count(series: pd.Series) -> int:
    """Number of (t-1, t) pairs where BOTH observations are present.

    Autocorrelation is only meaningful over genuinely adjacent observations. Dropping NaNs first would
    splice non-adjacent points together and manufacture persistence: ``[0, NaN, 1, NaN, 2, NaN, 3]``
    has no adjacent valid pair at all, yet compresses to a perfectly trending [0,1,2,3] (score 1.0)."""
    valid = series.notna()
    return int((valid & valid.shift(1)).sum())


def persistence_score(series: pd.Series) -> float:
    """``max(|acf1(x)|, |acf1(|x|)|, |acf1(x^2)|)`` computed over ADJACENT pairs only.

    Returns NaN when the series is degenerate or has too few adjacent pairs to correlate. Note the
    blind spots documented in the module docstring: this sees only marginal structure, not covariance
    or tail-shape regimes.
    """
    if adjacent_pair_count(series) < 3:
        return float("nan")
    x = pd.to_numeric(series, errors="coerce")
    if x.std(skipna=True) in (0, None) or not (x.std(skipna=True) > 0):
        return float("nan")
    # NOTE: no dropna() — pandas' corr aligns on index and drops only the non-finite PAIRS, so a gap
    # correctly contributes nothing instead of splicing its neighbours together.
    stats = [x.autocorr(1), x.abs().autocorr(1), (x ** 2).autocorr(1)]
    finite = [abs(s) for s in stats if s == s]
    return max(finite) if finite else float("nan")


def stage0b_persistence_filter(X: pd.DataFrame, min_persistence: float = 0.0,
                               min_obs: int = DEFAULT_MIN_OBS,
                               verbose: bool = True) -> tuple[pd.DataFrame, StageReport]:
    """Score marginal persistence; optionally remove features that are noise in every marginal.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (rows = bars, cols = features). May contain NaN.
    min_persistence : float
        Effect-size floor in [0, 1]. ``0.0`` (default) = DIAGNOSTIC ONLY: score and report, remove
        nothing. Removal is opt-in because this marginal test is blind to covariance and tail-shape
        regimes the emission can use (see module docstring). ``0.15`` is the FXPIG-M5 calibration;
        values >= 0.25 removed legitimate features there.
    min_obs : int
        Minimum adjacent valid pairs required to reject a feature. Below this the score is too noisy to
        act on (IID noise clears 0.15 ~47% of the time at n=60), so the feature is KEPT and flagged.
    verbose : bool
        Print removals, and the weakest scores when running as a diagnostic.

    Returns
    -------
    X_filtered : pd.DataFrame
    report : StageReport
    """
    if min_persistence != min_persistence:  # NaN would silently disable every comparison
        raise ValueError("min_persistence must be a number, got NaN")
    if not 0.0 <= min_persistence <= 1.0:
        raise ValueError(f"min_persistence must be in [0, 1], got {min_persistence}")
    if min_obs < 3:
        raise ValueError(f"min_obs must be >= 3, got {min_obs}")

    n_before = X.shape[1]
    scores = {c: persistence_score(X[c]) for c in X.columns}
    pairs = {c: adjacent_pair_count(X[c]) for c in X.columns}

    kept, removed, undersampled = [], [], []
    for col in X.columns:
        s, n = scores[col], pairs[col]
        if min_persistence <= 0.0:            # diagnostic mode — never destructive
            kept.append(col)
        elif s != s:                          # degenerate: the variance filter's remit, not ours
            kept.append(col)
        elif n < min_obs:                     # cannot support the call at this sample size
            undersampled.append(col)
            kept.append(col)
        elif s < min_persistence:
            removed.append(col)
        else:
            kept.append(col)

    if verbose:
        if removed:
            detail = ", ".join(f"{c} ({scores[c]:.3f})" for c in removed)
            print(f"  Stage 0b removed {len(removed)} memoryless features (score < {min_persistence}): {detail}")
        if undersampled:
            print(f"  Stage 0b kept {len(undersampled)} feature(s) with < {min_obs} adjacent pairs "
                  f"(score unreliable at that sample size): {undersampled}")
        if min_persistence <= 0.0:
            weak = sorted((s, c) for c, s in scores.items() if s == s)[:3]
            if weak:
                shown = ", ".join(f"{c} ({s:.3f})" for s, c in weak)
                print(f"  Stage 0b (diagnostic, no removal) weakest marginal persistence: {shown}")

    report = StageReport(stage="Stage0b_Persistence", n_before=n_before, n_after=len(kept), removed=removed)
    return X[kept], report
