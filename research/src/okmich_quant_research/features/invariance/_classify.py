"""Thresholds and classifiers that turn transform responses into a ``FeatureInvariance`` stamp.

Every constant here is a BAND, not a point estimate, because the measurement is exact but the feature
is not: a feature can be structurally odd yet land at -0.97 because of warm-up NaNs, clipping, or a
bounded output range.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import Parity, ScaleClass

#: Return multiplier for the homogeneity test; also the log base of the scale exponent.
SCALE_C = 2.0
#: |corr| below this is not a clean parity. 0.90 is deliberately far from 1.0: a genuinely odd feature
#: measures at -0.9999, and the one-sided pair measures at -0.69 and -0.44, so the band is wide open in
#: the middle and nothing real sits near the edge.
PARITY_BAND = 0.90
#: Homogeneity exponent at/above this carries scale; at/below SCALE_FREE it is scale-free. Between the
#: two is PARTIAL — reported honestly rather than forced into a cell.
SCALE_CARRY = 0.60
SCALE_FREE = 0.20
#: Minimum pairwise-complete observations before a correlation or an IQR is trusted.
MIN_OBS = 200
#: Correlation ceiling equivalent to MAX_VIF = 3.0, since VIF = 1/(1 - r**2) for a pair.
VIF3_R = float(np.sqrt(1.0 - 1.0 / 3.0))


def iqr(s: pd.Series) -> float:
    """Robust spread, or NaN when the column is degenerate.

    The degeneracy guard matters because this is the denominator of every response ratio. A
    near-constant column with a few extreme tail values has an IQR of ~0 and a real range of ~1, which
    turns a meaningless perturbation into a response of 1e11 and puts a broken feature at the top of the
    table (measured: ``path_structure._velocity_path.velocity_consistency``). Reporting it as unscored
    is the honest verdict, and it is also a defect worth seeing.
    """
    v = s.dropna().to_numpy()
    if v.size < MIN_OBS:
        return float("nan")
    q1, q3 = np.percentile(v, [25.0, 75.0])
    r = float(q3 - q1)
    if not np.isfinite(r) or r <= 0.0:
        return float("nan")
    span = float(np.nanmax(v) - np.nanmin(v))
    if span > 0.0 and r / span < 1e-6:
        return float("nan")
    return r


def scale_exponent(original: pd.Series, rescaled: pd.Series, c: float = SCALE_C) -> float:
    """``log_c(IQR(f_rescaled) / IQR(f_original))`` — ~1 carries scale, ~0 is scale-free."""
    i0, i1 = iqr(original), iqr(rescaled)
    if not (np.isfinite(i0) and np.isfinite(i1) and i0 > 0.0 and i1 > 0.0):
        return float("nan")
    return float(np.log(i1 / i0) / np.log(c))


def cross_correlations(a: pd.DataFrame, b: pd.DataFrame, min_obs: int = MIN_OBS) -> pd.DataFrame:
    """Pairwise-complete Pearson between every column of ``a`` (rows) and every column of ``b`` (cols).

    Columns are prefixed before concatenating because the two frames legitimately share column names —
    a pool against its own reflection is the whole point here. Without the prefix pandas aligns the two
    into one column and the self-correlation silently reads 1.0, which would classify every feature as
    EVEN and look entirely plausible.

    COST, accepted deliberately: correlating the concatenated frame computes the a-vs-a and b-vs-b
    blocks too, which are discarded — about 2x the pairs actually needed. The alternative is a
    hand-rolled cross-correlation, and the thing that would have to be reimplemented is precisely the
    pairwise-complete NaN handling (``min_periods`` over each pair's own overlap) that makes these
    numbers trustworthy on warm-up-truncated features. This is a research-time probe run rarely and
    offline; a 2x factor there is worth less than the risk of getting that subtlety wrong.
    """
    both = pd.concat([a.add_prefix("a|"), b.add_prefix("b|")], axis=1)
    c = both.corr(method="pearson", min_periods=min_obs)
    rows = [i for i in c.index if str(i).startswith("a|")]
    cols = [j for j in c.columns if str(j).startswith("b|")]
    out = c.loc[rows, cols]
    out.index = [str(i)[2:] for i in out.index]
    out.columns = [str(j)[2:] for j in out.columns]
    return out


def best_match(cmat: pd.DataFrame, col: str) -> tuple[str, float]:
    """For the TRANSFORMED column ``col``, the original column it most resembles, and that correlation.

    ``cmat`` is indexed by original columns and keyed by transformed ones, so the search runs DOWN
    column ``col``. When the answer is ``col`` itself the feature simply has a parity; when it is a
    different column the two are conjugates under reflection (plus_di/minus_di, aroon_up/aroon_down).
    """
    if col not in cmat.columns:
        return col, float("nan")
    s = cmat[col].abs()
    if not s.notna().any():
        return col, float("nan")
    name = str(s.idxmax())
    return name, float(cmat.at[name, col])


def classify_parity(refl_corr: float, conj_corr: float = float("nan"), conj_is_self: bool = True) -> Parity:
    """Parity of ``f``, with the ONE-SIDED case separated out from genuine confounding.

    A feature can fail the odd test without being defective. ``momentum.minus_di`` reflects onto
    ``plus_di`` rather than onto ``-minus_di``, and ``aroon_up`` reflects onto ``aroon_down`` — these are
    one-sided halves of an odd PAIR, so ``refl_corr`` lands mid-band (measured -0.687 and -0.436) even
    though nothing is confounded.

    That is a different finding from a feature that genuinely mixes direction with magnitude, and it has
    a different consequence: a one-sided feature's high state means "strong move THIS way" while its low
    state means "the other way OR no move at all", so a K=2 split on it is not an up/down partition.
    Detected by asking whether the reflected column matches some OTHER column in the same pool at
    ``>= PARITY_BAND``. Nothing else in the codebase can tell the two cases apart.
    """
    if not np.isfinite(refl_corr):
        return Parity.UNSCORED
    if refl_corr <= -PARITY_BAND:
        return Parity.ODD
    if refl_corr >= PARITY_BAND:
        return Parity.EVEN
    if (not conj_is_self) and np.isfinite(conj_corr) and abs(conj_corr) >= PARITY_BAND:
        return Parity.ONE_SIDED
    return Parity.MIXED


def classify_scale(exponent: float) -> ScaleClass:
    if not np.isfinite(exponent):
        return ScaleClass.UNSCORED
    if exponent >= SCALE_CARRY:
        return ScaleClass.CARRYING
    if exponent <= SCALE_FREE:
        return ScaleClass.FREE
    return ScaleClass.PARTIAL


def classify_cell(parity: Parity, scale: ScaleClass) -> str:
    """Human-readable invariance cell. Reporting only — eligibility is decided by ``registry.is_eligible``."""
    if parity is Parity.ONE_SIDED:
        return "one-sided (half of an odd pair)"
    if parity is Parity.ODD and scale is ScaleClass.CARRYING:
        return "odd/scale-carrying (signed drift)"
    if parity is Parity.ODD and scale is ScaleClass.FREE:
        return "odd/scale-free (normalised direction)"
    if parity is Parity.EVEN and scale is ScaleClass.CARRYING:
        return "even/scale-carrying (volatility)"
    if parity is Parity.EVEN and scale is ScaleClass.FREE:
        return "even/scale-free (path shape)"
    return "unclassified"
