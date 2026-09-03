"""Measure what a feature actually is, by rebuilding it on deliberately transformed price paths.

A feature is a function of a price path, so you can decide what it measures by feeding it paths whose
properties you control and watching what comes out. Two exact transforms are enough to place every
price-path feature (see ``_transforms``):

  * REFLECTION negates every log-return.  ``corr(f_reflected, f) ~ -1`` means the feature knows WHICH
    WAY the market went (ODD); ``~ +1`` means it knows only HOW MUCH (EVEN); matching a DIFFERENT
    column at ``~ +1`` means it is ONE-SIDED, half of an odd pair; anything else is MIXED, a defect.
  * RESCALING multiplies every log-deviation by ``c``.  The exponent ``log_c(IQR(f')/IQR(f))`` is ~1
    for a size measure (scale-carrying) and ~0 for a normalised one (scale-free).

Crossed, they define the three price-path axes (``registry._axis``). This module is what makes those
axes MEASURED rather than declared — without it the stamps in ``registry/_invariance.csv`` would be
just another set of hand-typed assertions, which is the exact failure the axis layer exists to fix.

WHAT THIS DOES NOT DO
---------------------
It cannot classify the LIQUIDITY axis. Liquidity features read a different input substrate (volume and
order flow), and reflecting the price path leaves them invariant, so the reflection test carries no
information about them. That asymmetry is real, not an omission.

The derivative-order test from the original lab probe (sinusoidal drift injection at three periods, to
separate features that see constant drift from those that see curvature) is deliberately NOT ported.
It exists to construct an ``acceleration`` axis, which was built, screened exhaustively to depth 3 on
6 majors and killed: WEAK at best (nsep 0.06-0.074), with its trend-joint indistinguishable from a
dwell-matched uninformative label (p 0.30-0.66). Do not re-add it here.

LIMITS OF THE EVIDENCE BEHIND THE SHIPPED STAMPS
------------------------------------------------
``registry/_invariance.csv`` was measured on 4 FX majors plus one index (FXPIG-Server M5, 80k bars).
Invariance is instrument-agnostic, so the CLASSIFICATION should hold everywhere. Axis USEFULNESS
demonstrably does not: indices gave 7 symbols and 7 different winning features on the trend axis, 6 of
7 at noise floor. Do not read a stamp as a claim that the feature is useful — only that it measures
what it measures.

USAGE
-----
    >>> from okmich_quant_research.features.invariance import probe_invariance, aggregate_stamps
    >>> frames = [probe_invariance(raw, feature_engineering, symbol=s) for s, raw in corpus.items()]
    >>> stamps = aggregate_stamps(pd.concat(frames), measured_on="FXPIG M5, 4 symbols, 80k bars, ...")
    >>> write_stamps_csv(stamps, INVARIANCE_CSV)

``feature_engineering`` is the same ``Callable[[pd.DataFrame], pd.DataFrame]`` contract
``HmmFeatureScreener`` takes, so any pool you can screen you can probe, with no adapter.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import pandas as pd

from ..registry import FeatureInvariance, Parity, ScaleClass
from ._classify import (MIN_OBS, SCALE_C, best_match, classify_cell, classify_parity, classify_scale,
                        cross_correlations, scale_exponent)
from ._transforms import reflect_ohlc, rescale_ohlc

FeatureEngineering = Callable[[pd.DataFrame], pd.DataFrame]

PROBE_COLUMNS = ("symbol", "feature", "refl_corr", "parity", "scale_exp", "scale_class", "cell",
                 "conjugate", "conjugate_corr", "n_valid")
STAMP_COLUMNS = ("feature", "parity", "scale_class", "conjugate", "measured_on")


def probe_invariance(raw: pd.DataFrame, feature_engineering: FeatureEngineering, *, symbol: str = "",
                     scale_c: float = SCALE_C, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Classify every column ``feature_engineering`` produces, on one price frame.

    Builds the pool three times — on ``raw``, on its reflection and on its rescaling — using the SAME
    builder each time, so any difference in the output is attributable to the transform and nothing
    else. Returns one row per feature with ``PROBE_COLUMNS``.

    Only columns present in all three frames are scored: a feature that throws on a transformed path
    (and is NaN-filled by a defensive builder) has no measurable parity, and reporting one would be a
    fabrication.
    """
    orig = feature_engineering(raw.copy())
    refl = feature_engineering(reflect_ohlc(raw))
    scal = feature_engineering(rescale_ohlc(raw, scale_c))

    cols = [c for c in orig.columns if c in refl.columns and c in scal.columns]
    if not cols:
        raise ValueError("feature_engineering produced no column present in all three variants; "
                         "nothing can be classified.")

    # ONE pairwise-complete matrix for the whole pool rather than a per-column loop: it yields both the
    # self term (f vs its own reflection) and the conjugate search (f's reflection vs every OTHER
    # column) from the same computation.
    cmat = cross_correlations(orig[cols], refl[cols], min_obs=min_obs)

    rows = []
    for col in cols:
        refl_corr = float(cmat.at[col, col]) if col in cmat.index else float("nan")
        conj_name, conj_corr = best_match(cmat, col)
        conj_is_self = conj_name == col
        exponent = scale_exponent(orig[col], scal[col], scale_c)
        parity = classify_parity(refl_corr, conj_corr, conj_is_self=conj_is_self)
        scale = classify_scale(exponent)
        rows.append({"symbol": symbol, "feature": col, "refl_corr": refl_corr, "parity": parity.value,
                     "scale_exp": exponent, "scale_class": scale.value,
                     "cell": classify_cell(parity, scale),
                     "conjugate": "" if conj_is_self else conj_name,
                     "conjugate_corr": float("nan") if conj_is_self else conj_corr,
                     "n_valid": int(orig[col].notna().sum())})
    return pd.DataFrame(rows, columns=list(PROBE_COLUMNS))


def aggregate_stamps(probe_df: pd.DataFrame, measured_on: str) -> dict[str, FeatureInvariance]:
    """Collapse per-symbol probe rows into one stamp per feature.

    Aggregation is by MEDIAN of the raw responses, re-classified — not by voting on the per-symbol
    verdicts. A parity that is not stable across symbols is not a property of the feature, and taking
    the median of the underlying correlation lets an unstable feature fall into MIXED on its own rather
    than being rescued by a majority.
    """
    if probe_df.empty:
        return {}
    def _median(series) -> float:
        """Median over the present values only.

        ``Series.median(skipna=True)`` on an all-NaN group emits a numpy "Mean of empty slice"
        RuntimeWarning while returning NaN anyway. An unmeasurable feature is an expected outcome here,
        not an anomaly, so it must not spray warnings across a corpus-sized run.
        """
        valid = series.dropna()
        return float(valid.median()) if len(valid) else float("nan")

    stamps: dict[str, FeatureInvariance] = {}
    for feature, group in probe_df.groupby("feature", sort=True):
        refl = _median(group["refl_corr"])
        exponent = _median(group["scale_exp"])
        conj_names = group["conjugate"].fillna("")
        conj_names = conj_names[conj_names != ""]
        conj = str(conj_names.value_counts().idxmax()) if len(conj_names) else ""
        conj_corr = _median(group["conjugate_corr"]) if len(conj_names) else float("nan")

        parity = classify_parity(refl, conj_corr, conj_is_self=(conj == ""))
        scale = classify_scale(exponent)
        stamps[str(feature)] = FeatureInvariance(parity=parity, scale_class=scale,
                                                 conjugate=conj if parity is Parity.ONE_SIDED else "",
                                                 measured_on=measured_on)
    return stamps


def stamp_summary(stamps: dict[str, FeatureInvariance]) -> pd.DataFrame:
    """One row per stamp — for eyeballing a fresh measurement before it is written to the registry."""
    return pd.DataFrame([{"feature": name, "parity": s.parity.value, "scale_class": s.scale_class.value,
                          "conjugate": s.conjugate, "cell": classify_cell(s.parity, s.scale_class)}
                         for name, s in sorted(stamps.items())])


def write_stamps_csv(stamps: dict[str, FeatureInvariance], path: str | Path) -> None:
    """Write stamps in the exact schema ``registry._invariance`` reads back.

    Regenerating the registry's stamp file is a library call rather than an ad-hoc script so the loop
    measure -> stamp -> screen stays reproducible.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(STAMP_COLUMNS))
        writer.writeheader()
        for name, s in sorted(stamps.items()):
            writer.writerow({"feature": name, "parity": s.parity.value, "scale_class": s.scale_class.value,
                             "conjugate": s.conjugate, "measured_on": s.measured_on})


def nearest_neighbour_redundancy(pool_a: pd.DataFrame, pool_b: pd.DataFrame,
                                 min_obs: int = MIN_OBS) -> pd.DataFrame:
    """For every column of ``pool_a``, its most-correlated column in ``pool_b`` and that ``|r|``.

    This is how "``momentum`` is ``trend`` at half the lookback" was measured: 23 of 38 momentum
    candidates sat at ``|r| >= 0.816`` — the ``MAX_VIF = 3.0`` near-duplicate line — against some trend
    candidate. Kept here because the same question is worth asking of any two pools before they are
    called separate axes.
    """
    xmat = cross_correlations(pool_a, pool_b, min_obs=min_obs)
    rows = []
    for name in xmat.index:
        r = xmat.loc[name].abs()
        has = bool(r.notna().any())
        rows.append({"feature": str(name), "nearest": str(r.idxmax()) if has else "",
                     "abs_r": float(r.max()) if has else float("nan")})
    return pd.DataFrame(rows, columns=["feature", "nearest", "abs_r"])


def unscored(stamps: dict[str, FeatureInvariance]) -> list[str]:
    """Names whose parity or scale could not be measured — a defect list, not a filler category."""
    return sorted(n for n, s in stamps.items()
                  if s.parity is Parity.UNSCORED or s.scale_class is ScaleClass.UNSCORED)
