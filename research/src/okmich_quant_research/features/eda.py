"""
Leakage-Safe Feature EDA Framework for Quantitative Features

Exploratory analysis of features from the ``okmich_quant_features`` package that cannot
silently contaminate a later backtest:

- Feature relevance (correlation / mutual information) with HAC-corrected inference
- Distribution analysis and normality tests
- Correlation analysis and multicollinearity detection
- Transformation recommendation, fitted on train and frozen for later windows

Leakage contract
----------------
Every statistic is computed on a *train* partition only. The last ``1 - train_threshold``
of the sample is a holdout that no analysis method reads unless the caller explicitly
passes ``scope=EDAScope.HOLDOUT`` / ``EDAScope.FULL``::

    |<---------- train_threshold = 0.75 ---------->|<---- 0.25 ---->|
    |  ...train rows...                   |PURGE h| |    HOLDOUT    |
                                                      never read

The trailing ``horizon`` bars are purged off the train edge so the last train label,
which spans ``horizon`` bars forward, cannot overlap the holdout.

:class:`EDAMode` selects what happens *inside* the train partition:

``HOLDOUT`` (default)
    One pass over the whole train partition.

``WALK_FORWARD``
    The train partition is subdivided into ``n_splits`` strictly-forward folds. Each fold
    reports its statistic in-sample *and* on its own held-out block, so you can read
    regime stability and in-sample-to-out-of-sample decay side by side. The
    ``train_threshold`` holdout is still never touched by any fold.

Note on the two splitters
-------------------------
Walk-forward folds (:func:`_walk_forward_folds`) are strictly causal: a fold's training
block always precedes its test block. :class:`_PurgedKFold`, reused from the screener for
:meth:`FeatureEDA.analyze_model_based_importance`, is *not* -- it draws training rows from
both sides of the test fold. That is standard and defensible for ranking feature
importance, which is not a backtest, but it is why the two are kept separate.
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, Iterator, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import QuantileTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from okmich_quant_features.utils.transform import (BoxCoxTransformer, LogitTransformer, LogTransformer,
                                                   YeoJohnsonTransformer)

from .screener._stage5 import _PurgedKFold

# Minimum finite observations before a statistic is considered estimable at all.
MIN_SAMPLES = 30

# Emitted verbatim by recommend_transformations() for heavy-tailed features. Kept as a
# literal (rather than a Transformation member) because downstream consumers such as
# okmich_quant_features.utils.transform.apply_transformation_recommendations parse the
# recommendation string and this exact token is part of that wire format.
QUANTILE_OR_RANK = "quantile/rank"


# ============================================================================
# ENUMS
# ============================================================================


class EDAMode(StrEnum):
    """How the train partition is consumed."""

    HOLDOUT = "holdout"
    WALK_FORWARD = "walk_forward"


class WFScheme(StrEnum):
    """Walk-forward fold geometry."""

    ANCHORED = "anchored"
    ROLLING = "rolling"


class TargetType(StrEnum):
    """Nature of the prediction target."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


class EDAScope(StrEnum):
    """Which partition an analysis method reads."""

    TRAIN = "train"
    HOLDOUT = "holdout"
    FULL = "full"


class ModelType(StrEnum):
    """Estimator family for model-based importance."""

    RF = "rf"
    LINEAR = "linear"


class CorrMethod(StrEnum):
    """Correlation estimator."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"


class Transformation(StrEnum):
    """Transformations :meth:`FeatureEDA.apply_transformation` can fit."""

    NONE = "none"
    LOGIT = "logit"
    LOG = "log"
    BOX_COX = "box-cox"
    YEO_JOHNSON = "yeo-johnson"
    QUANTILE = "quantile"
    RANK = "rank"
    STANDARDIZE = "standardize"


# ============================================================================
# HELPERS
# ============================================================================


def _finite_values(series: pd.Series) -> np.ndarray:
    """Numeric, finite, NaN-free view of ``series`` as a float array."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _finite_series(series: pd.Series) -> pd.Series:
    """Numeric, finite, NaN-free view of ``series`` keeping its index."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).dropna()


def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill then zero-fill a feature frame.

    ``ffill`` is causal (it only ever reads backwards). The ``fillna(0)`` that follows
    only touches leading NaNs that have no prior observation to carry forward.
    """
    return df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _valid_target_mask(y: pd.Series, target_type: TargetType) -> np.ndarray:
    """Boolean mask of usable target rows.

    A continuous target must additionally be finite: ``notna()`` alone lets ``inf`` through,
    which would inflate the reported training-row count relative to what the statistics
    actually consume.
    """
    if target_type == TargetType.CONTINUOUS:
        numeric = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
        return numeric.notna().to_numpy()
    return y.notna().to_numpy()


def _validate_index(index: pd.Index) -> None:
    """Reject an index that would make the positional train/holdout split meaningless.

    Every partition operation here is positional -- ``index[:split]`` is the train side -- so
    it only means "the earliest rows" if the index is sorted ascending. A descending index
    (a common CSV export shape) would otherwise put the *oldest* bars in the holdout and
    train the analysis on the future, silently and without error. Duplicate labels are
    rejected for a different reason: ``.loc`` on a duplicated label multiplies rows.
    """
    if not index.is_unique:
        duplicated = index[index.duplicated()].unique()
        raise ValueError(f"features/target index must be unique; {len(duplicated)} duplicated label(s), "
                         f"first few: {list(duplicated[:5])}")
    if not index.is_monotonic_increasing:
        raise ValueError("features/target index must be sorted ascending -- the train/holdout split is positional, "
                         "so an unsorted index would put non-chronological rows in the holdout and silently void "
                         "the leakage guarantee. Sort with df.sort_index() before constructing FeatureEDA.")


def _numeric_columns(df: pd.DataFrame, feature_names: List[str]) -> Tuple[List[str], List[str]]:
    """Split ``feature_names`` into numeric-dtype and non-numeric columns."""
    numeric, rejected = [], []
    for name in feature_names:
        (numeric if pd.api.types.is_numeric_dtype(df[name]) else rejected).append(name)
    return numeric, rejected


# Series-name suffixes preserved from the pre-leakage-audit implementation so downstream
# code that keys off the transformed column name keeps working.
_SUFFIX = {
    Transformation.LOGIT: "logit",
    Transformation.LOG: "log",
    Transformation.BOX_COX: "boxcox",
    Transformation.YEO_JOHNSON: "yeojohnson",
    Transformation.QUANTILE: "quantile",
    Transformation.RANK: "rank",
    Transformation.STANDARDIZE: "std",
}


# ============================================================================
# LEAKAGE MANIFEST
# ============================================================================


@dataclass(frozen=True)
class LeakageManifest:
    """Exactly which rows an EDA run was allowed to read.

    Attached to :attr:`FeatureEDA.manifest` and to every comprehensive report so a result
    can be audited long after the notebook cell that produced it has scrolled away.
    """

    mode: EDAMode
    wf_scheme: WFScheme
    train_threshold: float
    horizon: int
    n_splits: int
    embargo_pct: float
    n_total: int
    n_train: int
    n_purged: int
    n_holdout: int
    train_start: object
    train_end: object
    holdout_start: object
    holdout_end: object

    def describe(self) -> str:
        """Human-readable summary block."""
        wf_detail = f" ({self.wf_scheme}, n_splits={self.n_splits})" if self.mode == EDAMode.WALK_FORWARD else ""
        lines = [
            "LEAKAGE MANIFEST",
            "-" * 80,
            f"  mode              : {self.mode}{wf_detail}",
            f"  train_threshold   : {self.train_threshold}",
            f"  label horizon     : {self.horizon} bar(s)",
            f"  embargo_pct       : {self.embargo_pct}",
            f"  rows total        : {self.n_total}",
            f"  rows read (train) : {self.n_train}  [{self.train_start} .. {self.train_end}]",
            f"  rows purged       : {self.n_purged}  (trailing {self.horizon} bar(s) of train)",
            f"  rows held out     : {self.n_holdout}  [{self.holdout_start} .. {self.holdout_end}]  NEVER READ",
        ]
        return "\n".join(lines)


# ============================================================================
# FITTED TRANSFORMERS (the three the features package does not provide)
# ============================================================================


class _FittedStandardize:
    """Z-score against a frozen train mean and standard deviation."""

    def __init__(self) -> None:
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, X: pd.Series, y=None) -> "_FittedStandardize":
        values = _finite_values(X)
        if values.size == 0:
            raise ValueError("cannot fit standardize on a series with no finite values")
        self.mean_ = float(values.mean())
        std = float(values.std(ddof=1))
        # A constant train slice yields std == 0; fall back to 1.0 so transform returns
        # centred zeros rather than inf.
        self.std_ = std if std > 0 else 1.0
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        if self.mean_ is None:
            raise RuntimeError("Transformer must be fitted before transform")
        return ((X - self.mean_) / self.std_).rename(X.name)

    def fit_transform(self, X: pd.Series, y=None) -> pd.Series:
        return self.fit(X).transform(X)


class _FittedRank:
    """Percentile rank against a frozen train empirical distribution.

    ``Series.rank(pct=True)`` ranks every observation against every other row in the
    frame -- including rows that had not happened yet -- so a bar's rank moves when the
    future arrives. Freezing the train sample as a reference CDF and scoring later bars
    with ``searchsorted`` removes that look-ahead: a bar's rank depends only on the train
    distribution and on its own value.
    """

    def __init__(self) -> None:
        self.reference_: Optional[np.ndarray] = None

    def fit(self, X: pd.Series, y=None) -> "_FittedRank":
        values = _finite_values(X)
        if values.size == 0:
            raise ValueError("cannot fit rank on a series with no finite values")
        self.reference_ = np.sort(values)
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        if self.reference_ is None:
            raise RuntimeError("Transformer must be fitted before transform")
        values = pd.to_numeric(X, errors="coerce").to_numpy(dtype=float)
        ranks = np.searchsorted(self.reference_, values, side="right") / len(self.reference_)
        # searchsorted sorts NaN/inf to the top end; mask them back out rather than
        # letting a missing observation silently score as the 100th percentile.
        ranks = np.where(np.isfinite(values), ranks, np.nan)
        return pd.Series(ranks, index=X.index, name=X.name)

    def fit_transform(self, X: pd.Series, y=None) -> pd.Series:
        return self.fit(X).transform(X)


class _FittedQuantile:
    """Quantile transform against a frozen train quantile map."""

    def __init__(self, output_distribution: str = "normal", random_state: int = 42) -> None:
        self.output_distribution = output_distribution
        self.random_state = random_state
        self._transformer: Optional[QuantileTransformer] = None

    def fit(self, X: pd.Series, y=None) -> "_FittedQuantile":
        values = _finite_values(X)
        if values.size == 0:
            raise ValueError("cannot fit quantile on a series with no finite values")
        n_quantiles = int(min(1000, values.size))
        self._transformer = QuantileTransformer(n_quantiles=n_quantiles, output_distribution=self.output_distribution,
                                                random_state=self.random_state)
        self._transformer.fit(values.reshape(-1, 1))
        return self

    def transform(self, X: pd.Series, y=None) -> pd.Series:
        if self._transformer is None:
            raise RuntimeError("Transformer must be fitted before transform")
        values = pd.to_numeric(X, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        finite = np.isfinite(values)
        out = np.full(values.shape, np.nan, dtype=float)
        if finite.any():
            out[finite] = self._transformer.transform(values[finite].reshape(-1, 1)).ravel()
        return pd.Series(out, index=X.index, name=X.name)

    def fit_transform(self, X: pd.Series, y=None) -> pd.Series:
        return self.fit(X).transform(X)


def _warn_if_unsuitable(transformation: Transformation, values: pd.Series, feature: str) -> None:
    """Warn when a transformation is about to be fitted on data it will silently mangle.

    ``LogitTransformer`` clips its input to ``[eps, 1 - eps]`` and never validates the range.
    Pointed at an unbounded feature it therefore pins every negative value to ``logit(eps)``
    and every value above one to ``logit(1 - eps)`` -- roughly +/-16 -- flattening the majority
    of the sample onto two constants and destroying the signal, with no error raised.
    ``recommend_transformations`` only ever suggests logit for bounded features, but
    :meth:`FeatureEDA.apply_transformation` is public and takes whatever it is given.
    """
    if transformation != Transformation.LOGIT:
        return
    outside = float(((values < 0.0) | (values > 1.0)).mean())
    if outside > 0:
        warnings.warn(f"logit on '{feature}': {outside:.1%} of the fitted values fall outside [0, 1] and will be "
                      f"clipped to the epsilon bounds, flattening them onto two constants. Logit is for bounded "
                      f"ratios and oscillators -- use yeo-johnson or quantile for unbounded features.",
                      UserWarning, stacklevel=3)


def _build_transformer(transformation: Transformation, **kwargs):
    """Instantiate an unfitted transformer for ``transformation``.

    The power/log/logit families are reused from ``okmich_quant_features.utils.transform``
    so a spec fitted here is the same object a training pipeline would persist with joblib.
    """
    if transformation == Transformation.LOGIT:
        return LogitTransformer(epsilon=kwargs.get("epsilon", 1e-7))
    if transformation == Transformation.LOG:
        return LogTransformer()
    if transformation == Transformation.BOX_COX:
        return BoxCoxTransformer(standardize=kwargs.get("standardize", False))
    if transformation == Transformation.YEO_JOHNSON:
        return YeoJohnsonTransformer(standardize=kwargs.get("standardize", False))
    if transformation == Transformation.QUANTILE:
        return _FittedQuantile(output_distribution=kwargs.get("output_distribution", "normal"),
                               random_state=kwargs.get("random_state", 42))
    if transformation == Transformation.RANK:
        return _FittedRank()
    if transformation == Transformation.STANDARDIZE:
        return _FittedStandardize()
    raise ValueError(f"Unknown transformation: {transformation}")


@dataclass
class FittedTransformSpec:
    """Transformers fitted on one window, ready to apply to any other window.

    Returned by :meth:`FeatureEDA.fit_transformations`. Every parameter it holds -- a
    Box-Cox lambda, a log offset, a z-score mean, a quantile map, a rank reference CDF --
    was estimated on :attr:`fitted_on` and is frozen, so applying the spec to a later
    window cannot import information from that window back into the earlier one.
    """

    transformers: Dict[str, object] = field(default_factory=dict)
    transformations: Dict[str, Transformation] = field(default_factory=dict)
    fitted_on: Optional[pd.Index] = None
    failures: Dict[str, str] = field(default_factory=dict)

    @property
    def features(self) -> List[str]:
        """Features this spec can transform."""
        return list(self.transformers)

    def transform_feature(self, series: pd.Series, feature: str) -> pd.Series:
        """Apply the frozen transformer for ``feature`` to ``series``."""
        if feature not in self.transformers:
            raise KeyError(f"No fitted transformer for feature {feature!r}. Fitted: {self.features}")
        result = self.transformers[feature].transform(series)
        if not isinstance(result, pd.Series):
            result = pd.Series(np.asarray(result).ravel(), index=series.index, name=series.name)
        return result

    def transform(self, df: pd.DataFrame, replace_original: bool = True) -> pd.DataFrame:
        """Apply every fitted transformer to the matching column of ``df``."""
        out = df.copy()
        for feature, transformation in self.transformations.items():
            if feature not in df.columns:
                continue
            transformed = self.transform_feature(df[feature], feature)
            if replace_original:
                out[feature] = transformed
            else:
                out[f"{feature}_{_SUFFIX[transformation]}"] = transformed
        return out


# ============================================================================
# PARTITIONING
# ============================================================================


def _resolve_boundary(index: pd.Index, train_threshold: Union[float, object]) -> int:
    """Positional index of the first holdout row.

    ``train_threshold`` is a fraction of the sample when it is a float strictly inside
    ``(0, 1)``; anything else is treated as an index label and located with
    ``searchsorted``, so a timestamp boundary works too.
    """
    n = len(index)
    if n == 0:
        raise ValueError("cannot partition an empty index")

    if isinstance(train_threshold, (float, np.floating)):
        # A float is always a fraction. Falling through to the label path would let 1.5 be
        # searchsorted into a DatetimeIndex and fail with an opaque TypeError instead.
        fraction = float(train_threshold)
        if not 0.0 < fraction < 1.0:
            raise ValueError(f"a float train_threshold is a fraction and must lie strictly inside (0, 1), "
                             f"got {fraction}")
        split_at = int(n * fraction)
    else:
        try:
            split_at = int(index.searchsorted(train_threshold, side="left"))
        except TypeError as exc:
            raise ValueError(f"train_threshold={train_threshold!r} is neither a fraction in (0, 1) nor a label "
                             f"comparable with a {type(index).__name__}") from exc

    if split_at <= 0 or split_at >= n:
        raise ValueError(f"train_threshold={train_threshold!r} puts the boundary at position {split_at} of {n} rows, "
                         f"leaving an empty train or holdout partition")
    return split_at


def _walk_forward_folds(n: int, n_splits: int, horizon: int, embargo_bars: int,
                        scheme: WFScheme = WFScheme.ANCHORED,
                        train_window: Optional[int] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Strictly-forward folds over ``n`` positions.

    Every fold's training block ends before its test block begins, separated by a gap of
    ``horizon + embargo_bars``. The horizon component purges training rows whose label
    would still be open when the test block starts; the embargo component drops the
    serially-correlated bars immediately after that.

    ANCHORED grows the training block from position 0; ROLLING slides a fixed
    ``train_window`` (defaulting to one block).
    """
    block = n // (n_splits + 1)
    if block < 1:
        return

    gap = horizon + embargo_bars
    for k in range(n_splits):
        train_end = (k + 1) * block
        test_start = train_end + gap
        if test_start >= n:
            break
        test_end = n if k == n_splits - 1 else min(n, test_start + block)

        if scheme == WFScheme.ROLLING:
            window = train_window if train_window else block
            train_start = max(0, train_end - window)
        else:
            train_start = 0

        # Redundant with `gap` by construction, but stated explicitly so the purge
        # survives any future change to the gap arithmetic.
        train_stop = min(train_end, test_start - horizon)

        if train_stop - train_start < 1 or test_end - test_start < 1:
            continue
        yield np.arange(train_start, train_stop), np.arange(test_start, test_end)


# ============================================================================
# STATISTICS
# ============================================================================


def _hac_stats(x: np.ndarray, y: np.ndarray, maxlags: int) -> Tuple[float, float, float]:
    """Newey-West corrected t-statistic, p-value, and effective sample size.

    With an ``h``-bar forward target, consecutive labels overlap by ``h - 1`` bars and the
    iid p-values from ``scipy.stats.pearsonr`` overstate significance badly. Regressing the
    target on the feature with a HAC covariance at ``maxlags = h - 1`` prices that overlap in.

    ``n_eff`` is derived from how far the HAC standard error inflates the OLS one:
    ``n_eff = n * (se_ols / se_hac) ** 2``. That is a measured variance-inflation rather
    than the ``n / h`` rule of thumb, so it also picks up autocorrelation in the feature.
    """
    n = len(x)
    if n < MIN_SAMPLES or not np.isfinite(x).all() or not np.isfinite(y).all():
        return np.nan, np.nan, np.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan, np.nan, np.nan

    # A lag length approaching the sample size makes the Newey-West kernel degenerate (and
    # can leave the covariance non-positive-definite). Cap it well short of n; the cap only
    # binds when the label horizon is large relative to the window being scored, which is
    # exactly where an uncapped HAC would return noise dressed up as a statistic.
    lags = int(np.clip(maxlags, 0, max(0, n // 4)))

    design = sm.add_constant(x, has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ols = sm.OLS(y, design).fit()
            hac = sm.OLS(y, design).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    except Exception:
        return np.nan, np.nan, np.nan

    se_ols, se_hac = float(ols.bse[1]), float(hac.bse[1])
    n_eff = n * (se_ols / se_hac) ** 2 if se_hac > 0 else np.nan
    if np.isfinite(n_eff):
        n_eff = float(np.clip(n_eff, 1.0, n))
    return float(hac.tvalues[1]), float(hac.pvalues[1]), n_eff


def _numeric_target(y: pd.Series, target_type: TargetType) -> Optional[np.ndarray]:
    """Numeric encoding of ``y`` suitable for an OLS-based HAC test, or None.

    Continuous targets pass through. A binary categorical target is encoded 0/1, which
    makes the OLS slope the point-biserial correlation. Multi-class targets have no
    meaningful single-slope encoding, so they get no HAC statistic rather than a
    misleading one.
    """
    if target_type == TargetType.CONTINUOUS:
        return y.to_numpy(dtype=float)
    codes, uniques = pd.factorize(y)
    if len(uniques) == 2:
        return codes.astype(float)
    return None


def _relevance_stats(x: pd.Series, y: pd.Series, target_type: TargetType, maxlags: int) -> Dict[str, float]:
    """Correlation / mutual-information / HAC block for one feature against one target."""
    out: Dict[str, float] = {}
    x_values = x.to_numpy(dtype=float)

    if target_type == TargetType.CONTINUOUS:
        y_values = y.to_numpy(dtype=float)
        corr, p_val = stats.pearsonr(x_values, y_values)
        out["pearson_corr"] = float(corr)
        out["pearson_pval"] = float(p_val)
        out["abs_pearson"] = abs(float(corr))
        y_ranked = y_values
    else:
        # Rank correlation needs an ordering. A string-labelled categorical target would make
        # spearmanr raise, and because the caller drops a feature whose stat block fails, that
        # would empty the whole relevance table rather than just the one column.
        y_ranked = pd.factorize(y, sort=True)[0].astype(float)

    spearman, sp_pval = stats.spearmanr(x_values, y_ranked)
    out["spearman_corr"] = float(spearman)
    out["spearman_pval"] = float(sp_pval)
    out["abs_spearman"] = abs(float(spearman))

    feat_2d = x_values.reshape(-1, 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if target_type == TargetType.CONTINUOUS:
            mi = mutual_info_regression(feat_2d, y.to_numpy(dtype=float), random_state=42)[0]
        else:
            mi = mutual_info_classif(feat_2d, y.to_numpy(), random_state=42)[0]
    out["mutual_info"] = float(mi)

    y_numeric = _numeric_target(y, target_type)
    if y_numeric is None:
        out["hac_tstat"], out["hac_pval"], out["n_eff"] = np.nan, np.nan, np.nan
    else:
        out["hac_tstat"], out["hac_pval"], out["n_eff"] = _hac_stats(x_values, y_numeric, maxlags)
    return out


def _sign_consistency(values: List[float]) -> float:
    """Fraction of ``values`` sharing the sign of their mean; NaN when undecidable."""
    finite = [v for v in values if np.isfinite(v) and v != 0.0]
    if not finite:
        return np.nan
    reference = np.sign(np.mean(finite))
    if reference == 0:
        return np.nan
    return float(np.mean([np.sign(v) == reference for v in finite]))


def _decay(is_value: float, oos_value: float) -> float:
    """Fractional loss of effect size from in-sample to out-of-sample.

    ``0.0`` means the out-of-sample effect matched in-sample; ``-1.0`` means it vanished
    entirely; positive means it grew.
    """
    if not np.isfinite(is_value) or not np.isfinite(oos_value) or is_value == 0:
        return np.nan
    return float(abs(oos_value) / abs(is_value) - 1.0)


# ============================================================================
# CORE FRAMEWORK
# ============================================================================


class FeatureEDA:
    """
    Leakage-safe EDA framework for quantitative trading features.

    Every statistic is computed on the train partition only. The final
    ``1 - train_threshold`` of the sample is a holdout that no method reads unless the
    caller explicitly passes ``scope=EDAScope.HOLDOUT`` or ``EDAScope.FULL``.

    Parameters
    ----------
    features : pd.DataFrame
        DataFrame containing all features.
    target : pd.Series or np.ndarray
        Target variable (forward returns, labels, etc.). Must be *forward*-looking
        relative to the features; see ``horizon``.
    target_type : TargetType or str, default TargetType.CONTINUOUS
        'continuous' (regression) or 'categorical' (classification).
    feature_names : List[str], optional
        Subset of feature names to analyze. If None, analyzes all columns.
    mode : EDAMode or str, default EDAMode.HOLDOUT
        ``HOLDOUT`` runs one pass over the train partition. ``WALK_FORWARD`` subdivides the
        train partition into ``n_splits`` strictly-forward folds and reports each statistic
        in-sample and out-of-sample side by side.
    train_threshold : float or index label, default 0.75
        Boundary between train and holdout. A float in ``(0, 1)`` is a fraction of rows;
        anything else is treated as an index label (e.g. a timestamp).
    horizon : int, optional
        Number of forward bars the target spans. Drives the purge width at the train edge,
        the walk-forward fold gap, and the HAC lag length. Defaults to 1 with a warning --
        leaving it unset while passing a multi-bar forward return under-purges the boundary.
    n_splits : int, default 5
        Walk-forward folds, and CV folds for :meth:`analyze_model_based_importance`.
    embargo_pct : float, default 0.01
        Fraction of the partition embargoed after a test block.
    wf_scheme : WFScheme or str, default WFScheme.ANCHORED
        ``ANCHORED`` grows the training block from the start; ``ROLLING`` slides a fixed window.
    wf_train_window : int, optional
        Training window length in bars for ``ROLLING``. Defaults to one fold block.
    verbose : bool, default True
        If True, analysis methods print their reports (matching the notebook UX of
        :class:`~okmich_quant_research.features.period_stack_audit.PeriodStackAudit`).
        Set False for tests, batch pipelines, and model-selection loops.

    Examples
    --------
    >>> eda = FeatureEDA(features, fwd_returns, horizon=12)
    >>> eda.analyze_feature_relevance()          # train rows only, purged
    >>> spec = eda.fit_transformations()         # params frozen on train
    >>> oos = spec.transform(features.loc['2024-01-01':])
    """

    def __init__(self, features: pd.DataFrame, target: Union[pd.Series, np.ndarray],
                 target_type: Union[TargetType, str] = TargetType.CONTINUOUS,
                 feature_names: Optional[List[str]] = None, *,
                 mode: Union[EDAMode, str] = EDAMode.HOLDOUT,
                 train_threshold: Union[float, object] = 0.75, horizon: Optional[int] = None,
                 n_splits: int = 5, embargo_pct: float = 0.01,
                 wf_scheme: Union[WFScheme, str] = WFScheme.ANCHORED,
                 wf_train_window: Optional[int] = None, verbose: bool = True):
        self.target_type = TargetType(target_type)
        self.mode = EDAMode(mode)
        self.wf_scheme = WFScheme(wf_scheme)
        self.verbose = verbose

        if horizon is None:
            horizon = 1
            warnings.warn("horizon was not specified; assuming a 1-bar forward target. If your target spans more "
                          "bars, the train-edge purge and the HAC lag length are both too narrow -- pass "
                          "horizon=<bars> explicitly.", UserWarning, stacklevel=2)
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if n_splits < 1:
            raise ValueError(f"n_splits must be >= 1, got {n_splits}")
        if not 0.0 <= embargo_pct < 1.0:
            raise ValueError(f"embargo_pct must be in [0, 1), got {embargo_pct}")

        self.horizon = int(horizon)
        self.n_splits = int(n_splits)
        self.embargo_pct = float(embargo_pct)
        self.train_threshold = train_threshold
        self.wf_train_window = wf_train_window
        self.maxlags = max(0, self.horizon - 1)

        self.features = features.copy()
        if isinstance(target, np.ndarray):
            # A bare array carries no index. Wrapping it plain would give it a RangeIndex that
            # shares nothing with a DatetimeIndex-keyed feature frame, so the alignment below
            # would silently find no common rows; adopt the feature index instead.
            if len(target) != len(self.features):
                raise ValueError(f"an ndarray target must be the same length as features "
                                 f"({len(target)} vs {len(self.features)}); pass a Series to align by index")
            self.target = pd.Series(target, index=self.features.index)
        else:
            self.target = target.copy()

        if not self.features.columns.is_unique:
            duplicated = self.features.columns[self.features.columns.duplicated()].unique()
            raise ValueError(f"features has duplicate column name(s): {list(duplicated)}. Each name must be unique -- "
                             f"df[name] would otherwise return a DataFrame where a Series is expected.")

        _validate_index(self.features.index)
        _validate_index(self.target.index)

        # Align on the intersection, preserving the feature frame's row order.
        common_idx = self.features.index[self.features.index.isin(self.target.index)]
        if len(common_idx) == 0:
            raise ValueError("features and target share no index values")
        self.features = self.features.loc[common_idx]
        self.target = self.target.loc[common_idx]

        # The boundary is resolved on the raw aligned index BEFORE any NaN-target rows are
        # dropped. Filtering first (as the pre-leakage-audit version did) let the number of
        # missing targets inside the holdout shift the positional split point -- holdout
        # content leaking into the split geometry itself.
        split_at = _resolve_boundary(common_idx, self.train_threshold)
        train_raw = common_idx[:split_at]
        holdout_raw = common_idx[split_at:]

        # Purge the trailing `horizon` bars: a label opened at those rows is still running
        # when the holdout starts, so it is priced off holdout bars.
        purge = min(self.horizon, len(train_raw))
        train_purged = train_raw[: len(train_raw) - purge]
        if len(train_purged) == 0:
            raise ValueError(f"purging {self.horizon} bar(s) at the train edge leaves no training rows "
                             f"(train partition is only {len(train_raw)} rows)")

        # Unusable targets are dropped *within* the train partition, never across the boundary.
        train_target = self.target.loc[train_purged]
        self._train_index = train_purged[_valid_target_mask(train_target, self.target_type)]
        # Kept unfiltered: nothing reads holdout values unless a scope override asks, and
        # inspecting them here would make the manifest itself holdout-dependent.
        self._holdout_index = holdout_raw

        if len(self._train_index) == 0:
            raise ValueError("no training rows remain after dropping rows with a missing target")

        if feature_names is not None:
            requested = list(feature_names)
            self.feature_names = [f for f in requested if f in self.features.columns]
            missing = [f for f in requested if f not in self.features.columns]
            if missing:
                warnings.warn(f"{len(missing)} requested feature(s) are not columns of the frame and were dropped: "
                              f"{missing[:10]}", UserWarning, stacklevel=2)
        else:
            self.feature_names = list(self.features.columns)

        # Non-numeric columns cannot be correlated, VIF'd or fitted. Dropping them here with a
        # warning beats letting each analysis fail in its own idiosyncratic way further down
        # (a string column makes every CV fold raise, surfacing as "no fold produced a model").
        self.feature_names, non_numeric = _numeric_columns(self.features, self.feature_names)
        if non_numeric:
            warnings.warn(f"{len(non_numeric)} non-numeric feature(s) excluded from analysis: {non_numeric[:10]}",
                          UserWarning, stacklevel=2)
        if not self.feature_names:
            raise ValueError("no numeric features to analyze")

        n_train = len(self._train_index)
        self.embargo_bars = max(1, int(n_train * self.embargo_pct)) if self.embargo_pct > 0 else 0

        self._wf_folds: List[Tuple[np.ndarray, np.ndarray]] = []
        if self.mode == EDAMode.WALK_FORWARD:
            self._wf_folds = list(_walk_forward_folds(n_train, self.n_splits, self.horizon, self.embargo_bars,
                                                      self.wf_scheme, self.wf_train_window))
            if len(self._wf_folds) < 2:
                raise ValueError(f"walk-forward mode needs at least 2 constructible folds but got "
                                 f"{len(self._wf_folds)} from {n_train} training rows with n_splits="
                                 f"{self.n_splits}, horizon={self.horizon}, embargo_bars={self.embargo_bars}. "
                                 f"Lower n_splits, lower embargo_pct, or supply more data.")
            if len(self._wf_folds) < self.n_splits:
                warnings.warn(f"requested n_splits={self.n_splits} but only {len(self._wf_folds)} folds fit after "
                              f"purge and embargo; proceeding with {len(self._wf_folds)}.", UserWarning, stacklevel=2)
            # A window at least as long as every fold's training block clamps train_start to 0
            # on every fold, which is ANCHORED -- silently not what the caller asked for.
            if (self.wf_scheme == WFScheme.ROLLING and self.wf_train_window
                    and all(train_pos[0] == 0 for train_pos, _ in self._wf_folds)):
                warnings.warn(f"wf_train_window={self.wf_train_window} is at least as long as every fold's training "
                              f"block, so ROLLING produces exactly the same folds as ANCHORED here. Lower it to get "
                              f"genuinely sliding windows.", UserWarning, stacklevel=2)

        self.manifest = LeakageManifest(
            mode=self.mode, wf_scheme=self.wf_scheme, train_threshold=self.train_threshold, horizon=self.horizon,
            n_splits=len(self._wf_folds) if self._wf_folds else self.n_splits, embargo_pct=self.embargo_pct,
            n_total=len(common_idx), n_train=len(self._train_index), n_purged=purge, n_holdout=len(holdout_raw),
            train_start=self._train_index[0], train_end=self._train_index[-1],
            holdout_start=holdout_raw[0] if len(holdout_raw) else None,
            holdout_end=holdout_raw[-1] if len(holdout_raw) else None,
        )

        # Results storage
        self.relevance_results = {}
        self.distribution_results = {}
        self.correlation_results = {}
        self.transformation_results = {}

    # ------------------------------------------------------------------
    # Partitioning / scope
    # ------------------------------------------------------------------

    def _log(self, msg: str = "") -> None:
        if self.verbose:
            print(msg)

    @property
    def train_index(self) -> pd.Index:
        """Rows the analysis methods are allowed to read."""
        return self._train_index

    @property
    def holdout_index(self) -> pd.Index:
        """Rows withheld from every default analysis."""
        return self._holdout_index

    @property
    def train_features(self) -> pd.DataFrame:
        """Feature frame restricted to the train partition."""
        return self.features.loc[self._train_index]

    @property
    def train_target(self) -> pd.Series:
        """Target restricted to the train partition."""
        return self.target.loc[self._train_index]

    @property
    def wf_folds(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Walk-forward folds as positional indices into :attr:`train_index`."""
        return list(self._wf_folds)

    def _resolve_scope(self, scope: Optional[Union[EDAScope, str]]) -> EDAScope:
        """Default to TRAIN; warn loudly when a caller opts into holdout rows."""
        if scope is None:
            return EDAScope.TRAIN
        scope = EDAScope(scope)
        if scope != EDAScope.TRAIN:
            warnings.warn(f"scope={scope} reads holdout rows. Any feature decision made from this output is "
                          f"contaminated with respect to a backtest covering that period.",
                          UserWarning, stacklevel=3)
        return scope

    def _index_for(self, scope: EDAScope) -> pd.Index:
        """Row index a given scope may read, with missing targets dropped."""
        if scope == EDAScope.TRAIN:
            return self._train_index
        if scope == EDAScope.HOLDOUT:
            candidate = self._holdout_index
        else:
            # FULL means every aligned row, including the band purged off the train edge --
            # those rows exist, they are simply unusable for a train-only statistic.
            candidate = self.features.index
        return candidate[_valid_target_mask(self.target.loc[candidate], self.target_type)]

    def _slice(self, index: pd.Index) -> Tuple[pd.DataFrame, pd.Series]:
        """Feature frame and target for ``index``, with non-finite targets removed."""
        y = self.target.loc[index]
        if self.target_type == TargetType.CONTINUOUS:
            y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = y.dropna()
        return self.features.loc[y.index], y

    # ------------------------------------------------------------------
    # 1. FEATURE RELEVANCE ANALYSIS
    # ------------------------------------------------------------------

    def _relevance_single(self, index: pd.Index) -> pd.DataFrame:
        """Per-feature relevance block over exactly the rows in ``index``."""
        X, y = self._slice(index)
        if len(y) < MIN_SAMPLES:
            return pd.DataFrame()

        rows = []
        for feat in self.feature_names:
            if feat not in X.columns:
                continue
            raw = X[feat]
            if raw.isna().mean() > 0.5:
                continue
            # ffill is causal: it only ever carries an earlier observation forward.
            feat_data = _finite_series(raw.ffill())
            if len(feat_data) < MIN_SAMPLES or feat_data.nunique() < 2:
                continue
            target_data = y.loc[feat_data.index]
            if target_data.nunique() < 2:
                continue

            row = {"feature": feat}
            try:
                row.update(_relevance_stats(feat_data, target_data, self.target_type, self.maxlags))
            except Exception as exc:
                # Report rather than vanish: a feature that silently drops out of the table is
                # indistinguishable from one that was never passed in.
                self._log(f"  {feat}: relevance statistics failed ({type(exc).__name__}: {exc})")
                continue
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _agg(records: List[Dict[str, float]], key: str) -> Tuple[float, float]:
        """Mean and standard deviation of ``key`` across fold records."""
        values = [r.get(key, np.nan) for r in records]
        values = [float(v) for v in values if np.isfinite(v)]
        if not values:
            return np.nan, np.nan
        return float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    def _relevance_walk_forward(self) -> pd.DataFrame:
        """In-sample and out-of-sample relevance per feature, aggregated across folds."""
        # Only folds where BOTH the in-sample and out-of-sample block produced a stat block are
        # kept, so every aggregate -- and `decay` above all -- compares the same set of folds.
        # Averaging 5 in-sample folds against 3 out-of-sample ones would make decay an artefact
        # of which folds happened to be estimable.
        per_feature: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        for train_pos, test_pos in self._wf_folds:
            is_records = {r["feature"]: r for r in self._relevance_single(self._train_index[train_pos])
                          .to_dict("records")}
            oos_records = {r["feature"]: r for r in self._relevance_single(self._train_index[test_pos])
                           .to_dict("records")}
            # Iterate in feature_names order, not set-intersection order: a set of strings
            # iterates by hash, which varies with PYTHONHASHSEED and would make the row order
            # of the returned frame differ between processes.
            for feature in self.feature_names:
                if feature in is_records and feature in oos_records:
                    bucket = per_feature.setdefault(feature, {"is": [], "oos": []})
                    bucket["is"].append(is_records[feature])
                    bucket["oos"].append(oos_records[feature])

        primary = "pearson_corr" if self.target_type == TargetType.CONTINUOUS else "mutual_info"
        rows = []
        for feat, bucket in per_feature.items():
            if not bucket["is"] or not bucket["oos"]:
                continue
            row: Dict[str, object] = {"feature": feat}

            stat_keys = ["spearman_corr", "mutual_info"]
            if self.target_type == TargetType.CONTINUOUS:
                stat_keys.insert(0, "pearson_corr")
            for key in stat_keys:
                stem = key.replace("_corr", "")
                row[f"{stem}_is_mean"], row[f"{stem}_is_std"] = self._agg(bucket["is"], key)
                row[f"{stem}_oos_mean"], row[f"{stem}_oos_std"] = self._agg(bucket["oos"], key)

            row["hac_tstat_is_mean"], _ = self._agg(bucket["is"], "hac_tstat")
            row["n_eff_is_mean"], _ = self._agg(bucket["is"], "n_eff")

            primary_stem = primary.replace("_corr", "")
            row["decay"] = _decay(row[f"{primary_stem}_is_mean"], row[f"{primary_stem}_oos_mean"])
            row["oos_sign_consistency"] = _sign_consistency([r.get("spearman_corr", np.nan) for r in bucket["oos"]])
            row["n_folds"] = len(bucket["is"])

            # Compat aliases so the shared sort key and any downstream consumer that reads
            # `abs_pearson` / `mutual_info` keep working in both modes. They carry the
            # out-of-sample number, which is the one worth ranking on.
            if self.target_type == TargetType.CONTINUOUS:
                row["abs_pearson"] = abs(row["pearson_oos_mean"]) if np.isfinite(row["pearson_oos_mean"]) else np.nan
            row["abs_spearman"] = abs(row["spearman_oos_mean"]) if np.isfinite(row["spearman_oos_mean"]) else np.nan
            row["mutual_info"] = row["mutual_info_oos_mean"]
            rows.append(row)

        return pd.DataFrame(rows)

    def analyze_feature_relevance(self, n_top: int = 20, scope: Optional[Union[EDAScope, str]] = None) -> pd.DataFrame:
        """
        Feature relevance using correlation, mutual information, and HAC-corrected inference.

        In ``HOLDOUT`` mode this is one pass over the train partition. In ``WALK_FORWARD``
        mode each fold contributes an in-sample and an out-of-sample block, aggregated into
        ``*_is_mean`` / ``*_oos_mean`` columns plus ``decay`` and ``oos_sign_consistency``.

        The ``hac_pval`` column supersedes ``pearson_pval`` for selection: with an
        ``h``-bar forward target the iid p-value is overstated, and ``n_eff`` reports how
        many independent observations the sample is really worth.

        Parameters
        ----------
        n_top : int, default 20
            Number of top features to print.
        scope : EDAScope or str, optional
            Defaults to the train partition. Passing HOLDOUT or FULL reads withheld rows
            and warns.

        Returns
        -------
        pd.DataFrame
            Relevance scores, sorted most relevant first.
        """
        scope = self._resolve_scope(scope)
        self._log("=" * 80)
        self._log("FEATURE RELEVANCE ANALYSIS")
        self._log("=" * 80)

        if self.mode == EDAMode.WALK_FORWARD and scope == EDAScope.TRAIN:
            results_df = self._relevance_walk_forward()
        else:
            results_df = self._relevance_single(self._index_for(scope))

        if not results_df.empty:
            sort_col = "abs_pearson" if self.target_type == TargetType.CONTINUOUS else "mutual_info"
            if sort_col in results_df.columns:
                results_df = results_df.sort_values(sort_col, ascending=False,
                                                    kind="mergesort").reset_index(drop=True)

        self.relevance_results = results_df

        self._log(f"\nScope: {scope} | mode: {self.mode} | rows: {len(self._index_for(scope))}")
        self._log(f"\nTop {n_top} Most Relevant Features:")
        self._log("-" * 80)
        self._log(results_df.head(n_top).to_string(index=False) if not results_df.empty else "None")

        return results_df

    def _make_model(self, model_type: ModelType):
        """Instantiate the estimator for model-based importance."""
        if model_type == ModelType.RF:
            cls = RandomForestRegressor if self.target_type == TargetType.CONTINUOUS else RandomForestClassifier
            return cls(n_estimators=100, max_depth=10, min_samples_leaf=50, random_state=42, n_jobs=-1)
        if self.target_type == TargetType.CONTINUOUS:
            return LinearRegression()
        return LogisticRegression(max_iter=1000, random_state=42)

    @staticmethod
    def _extract_importance(model) -> np.ndarray:
        """Non-negative importance vector from a fitted estimator."""
        if hasattr(model, "feature_importances_"):
            return np.asarray(model.feature_importances_, dtype=float)
        coef = np.asarray(model.coef_, dtype=float)
        if coef.ndim > 1:
            # Multi-class logistic: average the magnitude across one-vs-rest rows.
            return np.abs(coef).mean(axis=0)
        return np.abs(coef)

    def analyze_model_based_importance(self, model_type: Union[ModelType, str] = ModelType.RF, n_top: int = 20,
                                       scope: Optional[Union[EDAScope, str]] = None) -> pd.DataFrame:
        """
        Feature importance from purged, embargoed cross-validation.

        The estimator is refitted on each CV training fold and the resulting importances are
        averaged, replacing the single in-sample fit this method used to report. ``importance_std``
        exposes how much a feature's rank depends on which fold you looked at.

        For the linear path, features are z-scored using each fold's *training* mean and
        standard deviation, so ``|coef|`` compares like with like instead of tracking whichever
        feature happens to have the largest units.

        Notes
        -----
        :class:`_PurgedKFold` draws training rows from both sides of the test fold, so this is
        not a causal backtest -- it is a ranking device whose folds are purged of label overlap.
        The strictly-forward folds used by ``WALK_FORWARD`` mode are separate.

        Parameters
        ----------
        model_type : ModelType or str, default ModelType.RF
            'rf' for Random Forest, 'linear' for Linear/Logistic Regression.
        n_top : int, default 20
            Number of top features to print.
        scope : EDAScope or str, optional
            Defaults to the train partition.

        Returns
        -------
        pd.DataFrame
            Columns ``feature``, ``importance``, ``importance_std``, ``n_folds``.
        """
        model_type = ModelType(model_type)
        scope = self._resolve_scope(scope)

        self._log("\n" + "=" * 80)
        self._log(f"MODEL-BASED FEATURE IMPORTANCE ({str(model_type).upper()}, PURGED CV)")
        self._log("=" * 80)

        X_raw, y = self._slice(self._index_for(scope))
        X = _clean_frame(X_raw[self.feature_names])

        cv = _PurgedKFold(n_splits=self.n_splits, horizon=self.horizon, embargo_pct=self.embargo_pct)
        fold_importances = []
        for train_pos, _ in cv.split(X):
            X_tr, y_tr = X.iloc[train_pos], y.iloc[train_pos]
            if len(X_tr) < MIN_SAMPLES:
                continue
            if self.target_type == TargetType.CATEGORICAL and y_tr.nunique() < 2:
                continue
            if model_type == ModelType.LINEAR:
                mu, sigma = X_tr.mean(), X_tr.std(ddof=1).replace(0.0, 1.0)
                X_tr = (X_tr - mu) / sigma
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = self._make_model(model_type).fit(X_tr, y_tr)
                fold_importances.append(self._extract_importance(model))
            except Exception as exc:
                self._log(f"  skipped a fold: {exc}")

        if not fold_importances:
            raise RuntimeError("no CV fold produced a fitted model; lower n_splits or supply more training data")

        stacked = np.vstack(fold_importances)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": stacked.mean(axis=0),
            "importance_std": stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros(stacked.shape[1]),
            "n_folds": len(stacked),
        }).sort_values("importance", ascending=False, kind="mergesort").reset_index(drop=True)

        self._log(f"\nFolds used: {len(stacked)} | rows: {len(X)}")
        self._log(f"\nTop {n_top} Most Important Features:")
        self._log("-" * 80)
        self._log(importance_df.head(n_top).to_string(index=False))

        return importance_df

    # ------------------------------------------------------------------
    # 2. DISTRIBUTION ANALYSIS
    # ------------------------------------------------------------------

    def analyze_distributions(self, n_features: int = 10,
                              scope: Optional[Union[EDAScope, str]] = None) -> pd.DataFrame:
        """
        Distribution statistics and normality tests, computed on the train partition.

        Parameters
        ----------
        n_features : int, default 10
            Retained for signature compatibility; every feature is described.
        scope : EDAScope or str, optional
            Defaults to the train partition.

        Returns
        -------
        pd.DataFrame
            Distribution statistics for all features.
        """
        scope = self._resolve_scope(scope)
        index = self._index_for(scope)

        self._log("\n" + "=" * 80)
        self._log("FEATURE DISTRIBUTION ANALYSIS")
        self._log("=" * 80)

        results = []
        for feat in self.feature_names:
            feat_data = _finite_series(self.features.loc[index, feat])
            if len(feat_data) < MIN_SAMPLES:
                continue

            feat_stats = {
                "feature": feat,
                "count": len(feat_data),
                "mean": feat_data.mean(),
                "std": feat_data.std(),
                "min": feat_data.min(),
                "max": feat_data.max(),
                "skewness": stats.skew(feat_data),
                "kurtosis": stats.kurtosis(feat_data),
            }

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Shapiro-Wilk is only meaningful up to a few thousand observations.
                if len(feat_data) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(feat_data)
                    feat_stats["shapiro_stat"] = shapiro_stat
                    feat_stats["shapiro_pval"] = shapiro_p
                    feat_stats["is_normal_shapiro"] = shapiro_p > 0.05

                jb_stat, jb_p = stats.jarque_bera(feat_data)
            feat_stats["jarque_bera_stat"] = jb_stat
            feat_stats["jarque_bera_pval"] = jb_p
            feat_stats["is_normal_jb"] = jb_p > 0.05

            feat_stats["is_bounded_0_1"] = bool(feat_data.min() >= 0 and feat_data.max() <= 1)
            feat_stats["is_strictly_positive"] = bool(feat_data.min() > 0)
            results.append(feat_stats)

        dist_df = pd.DataFrame(results)
        self.distribution_results = dist_df

        self._log("\nDistribution Summary:")
        self._log("-" * 80)
        self._log(f"Scope: {scope} | rows: {len(index)}")
        self._log(f"Total features analyzed: {len(dist_df)}")
        if dist_df.empty:
            return dist_df

        n_normal = int(dist_df["is_normal_jb"].sum())
        self._log(f"Normal distributions (Jarque-Bera): {n_normal} ({n_normal / len(dist_df) * 100:.1f}%)")
        self._log(f"Bounded [0,1] features: {int(dist_df['is_bounded_0_1'].sum())}")
        self._log(f"Strictly positive features: {int(dist_df['is_strictly_positive'].sum())}")

        self._log("\nFeatures with High Skewness (|skew| > 2):")
        high_skew = dist_df[dist_df["skewness"].abs() > 2].sort_values("skewness", key=abs, ascending=False)
        self._log(high_skew[["feature", "skewness", "kurtosis"]].head(10).to_string(index=False)
                  if len(high_skew) else "None")

        self._log("\nFeatures with High Kurtosis (|kurtosis| > 3):")
        high_kurt = dist_df[dist_df["kurtosis"].abs() > 3].sort_values("kurtosis", key=abs, ascending=False)
        self._log(high_kurt[["feature", "skewness", "kurtosis"]].head(10).to_string(index=False)
                  if len(high_kurt) else "None")

        return dist_df

    def _features_to_plot(self, features_to_plot: Optional[List[str]], n_features: int) -> List[str]:
        """Explicit selection, else the top features by the last relevance run."""
        if features_to_plot is not None:
            return features_to_plot
        if isinstance(self.relevance_results, pd.DataFrame) and len(self.relevance_results) > 0:
            return self.relevance_results.head(n_features)["feature"].tolist()
        return self.feature_names[:n_features]

    @staticmethod
    def _grid(n_plots: int, figsize: Tuple[int, int]):
        """Subplot grid flattened to a 1-D axes array."""
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).ravel()
        return fig, axes

    def plot_distributions(self, features_to_plot: Optional[List[str]] = None, n_features: int = 12,
                           figsize: Tuple[int, int] = (20, 15), scope: Optional[Union[EDAScope, str]] = None):
        """
        Histograms with a KDE overlay, drawn from the train partition only.

        Plotting the holdout would leak it into the researcher's judgement just as surely as
        into a fitted parameter, so the scope is stamped into the figure title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        scope = self._resolve_scope(scope)
        index = self._index_for(scope)
        selected = self._features_to_plot(features_to_plot, n_features)

        fig, axes = self._grid(len(selected), figsize)
        for idx, feat in enumerate(selected):
            if idx >= len(axes):
                break
            ax = axes[idx]
            feat_data = _finite_series(self.features.loc[index, feat])
            if feat_data.empty:
                ax.axis("off")
                continue

            ax.hist(feat_data, bins=50, density=True, alpha=0.6, color="steelblue", edgecolor="black")
            try:
                kde = stats.gaussian_kde(feat_data)
                x_range = np.linspace(feat_data.min(), feat_data.max(), 100)
                ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
                ax.legend()
            except Exception:
                pass

            ax.set_title(f"{feat}\nSkew: {stats.skew(feat_data):.2f}, Kurt: {stats.kurtosis(feat_data):.2f}",
                         fontsize=10)
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(alpha=0.3)

        for idx in range(len(selected), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.suptitle(f"Feature Distributions ({scope} rows)", fontsize=16, y=1.001)
        return fig

    def plot_qq_plots(self, features_to_plot: Optional[List[str]] = None, n_features: int = 12,
                      figsize: Tuple[int, int] = (20, 15), scope: Optional[Union[EDAScope, str]] = None):
        """
        Q-Q plots to assess normality, drawn from the train partition only.

        Returns
        -------
        matplotlib.figure.Figure
        """
        scope = self._resolve_scope(scope)
        index = self._index_for(scope)
        selected = self._features_to_plot(features_to_plot, n_features)

        fig, axes = self._grid(len(selected), figsize)
        for idx, feat in enumerate(selected):
            if idx >= len(axes):
                break
            ax = axes[idx]
            feat_data = _finite_series(self.features.loc[index, feat])
            if feat_data.empty:
                ax.axis("off")
                continue
            stats.probplot(feat_data, dist="norm", plot=ax)
            ax.set_title(f"{feat}", fontsize=10)
            ax.grid(alpha=0.3)

        for idx in range(len(selected), len(axes)):
            axes[idx].axis("off")

        plt.tight_layout()
        plt.suptitle(f"Q-Q Plots -- Normality Assessment ({scope} rows)", fontsize=16, y=1.001)
        return fig

    # ------------------------------------------------------------------
    # 3. CORRELATION & MULTICOLLINEARITY ANALYSIS
    # ------------------------------------------------------------------

    def _design_matrix(self, scope: EDAScope) -> pd.DataFrame:
        """Cleaned feature matrix for the rows a scope may read."""
        return _clean_frame(self.features.loc[self._index_for(scope), self.feature_names])

    def analyze_correlation(self, threshold: float = 0.8, method: Union[CorrMethod, str] = CorrMethod.PEARSON,
                            scope: Optional[Union[EDAScope, str]] = None
                            ) -> Tuple[pd.DataFrame, List[Tuple[str, str, float]]]:
        """
        Correlation analysis and redundant-feature detection on the train partition.

        Parameters
        ----------
        threshold : float, default 0.8
            Correlation threshold for flagging redundant features.
        method : CorrMethod or str, default CorrMethod.PEARSON
        scope : EDAScope or str, optional
            Defaults to the train partition.

        Returns
        -------
        Tuple[pd.DataFrame, List[Tuple[str, str, float]]]
            Correlation matrix and highly correlated pairs, strongest first.
        """
        method = CorrMethod(method)
        scope = self._resolve_scope(scope)

        self._log("\n" + "=" * 80)
        self._log(f"CORRELATION ANALYSIS ({str(method).upper()})")
        self._log("=" * 80)

        corr_matrix = self._design_matrix(scope).corr(method=str(method))
        self.correlation_results["matrix"] = corr_matrix
        self.correlation_results["method"] = method
        self.correlation_results["scope"] = scope

        high_corr_pairs = []
        columns = list(corr_matrix.columns)
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_val = corr_matrix.iloc[i, j]
                if np.isfinite(corr_val) and abs(corr_val) >= threshold:
                    high_corr_pairs.append((columns[i], columns[j], float(corr_val)))

        high_corr_pairs.sort(key=lambda pair: abs(pair[2]), reverse=True)
        self.correlation_results["high_corr_pairs"] = high_corr_pairs

        self._log(f"\nFound {len(high_corr_pairs)} highly correlated pairs (|corr| >= {threshold}):")
        self._log("-" * 80)
        if high_corr_pairs:
            self._log(f"{'Feature 1':<30} {'Feature 2':<30} {'Correlation':>12}")
            self._log("-" * 80)
            for f1, f2, corr in high_corr_pairs[:20]:
                self._log(f"{f1:<30} {f2:<30} {corr:>12.4f}")
            if len(high_corr_pairs) > 20:
                self._log(f"\n... and {len(high_corr_pairs) - 20} more pairs")

        return corr_matrix, high_corr_pairs

    def compute_vif(self, vif_threshold: float = 10.0, scope: Optional[Union[EDAScope, str]] = None) -> pd.DataFrame:
        """
        Variance Inflation Factor for multicollinearity detection, on the train partition.

        Parameters
        ----------
        vif_threshold : float, default 10.0
            VIF above which a feature is flagged.
        scope : EDAScope or str, optional
            Defaults to the train partition.

        Returns
        -------
        pd.DataFrame
            VIF score per feature, highest first.
        """
        scope = self._resolve_scope(scope)

        self._log("\n" + "=" * 80)
        self._log("VARIANCE INFLATION FACTOR (VIF) ANALYSIS")
        self._log("=" * 80)

        X = self._design_matrix(scope)
        # VIF must be computed against a design that includes an intercept. Without one, the
        # auxiliary regression is forced through the origin and its R^2 absorbs the feature's
        # mean, so two entirely independent features with large means (an ATR around 50 and a
        # price around 1800, say) report a VIF in the thousands instead of 1.0 -- and get
        # dropped as collinear. The constant occupies column 0, so feature i sits at i + 1.
        design = sm.add_constant(X.to_numpy(dtype=float), has_constant="add")
        vif_data = []
        for i, feat in enumerate(self.feature_names):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    vif = float(variance_inflation_factor(design, i + 1))
                if not np.isfinite(vif):
                    raise ValueError("undefined VIF")
                vif_data.append({"feature": feat, "vif": vif, "high_multicollinearity": vif > vif_threshold})
            except Exception:
                vif_data.append({"feature": feat, "vif": np.nan, "high_multicollinearity": False})

        vif_df = pd.DataFrame(vif_data).sort_values("vif", ascending=False, kind="mergesort").reset_index(drop=True)
        self.correlation_results["vif"] = vif_df

        self._log(f"\nFeatures with VIF > {vif_threshold} (High Multicollinearity):")
        self._log("-" * 80)
        high_vif = vif_df[vif_df["high_multicollinearity"]]
        self._log(high_vif.to_string(index=False) if len(high_vif) else "None - all features have acceptable VIF")

        return vif_df

    def plot_correlation_matrix(self, method: Union[CorrMethod, str] = CorrMethod.PEARSON,
                                figsize: Tuple[int, int] = (16, 14), cluster: bool = True,
                                scope: Optional[Union[EDAScope, str]] = None):
        """
        Correlation heatmap for the train partition.

        Returns
        -------
        matplotlib.figure.Figure or seaborn.matrix.ClusterGrid
        """
        method = CorrMethod(method)
        scope = self._resolve_scope(scope)
        corr_matrix = self._design_matrix(scope).corr(method=str(method))

        # `square` is meaningful for a plain heatmap but seaborn ignores it (and warns) for a
        # clustermap, whose aspect is driven by the dendrogram layout.
        heatmap_kwargs = dict(cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=False, fmt=".2f",
                              linewidths=0.5, cbar_kws={"label": "Correlation"})
        title = f"Correlation Matrix ({str(method).capitalize()}, {scope} rows)"

        if cluster:
            # A constant feature has undefined correlation with everything, and scipy's linkage
            # rejects NaN outright. Drop those rows/columns rather than crash mid-plot; they
            # carry no clustering information anyway.
            usable = corr_matrix.columns[corr_matrix.notna().sum() > 1]
            clusterable = corr_matrix.loc[usable, usable].fillna(0.0)
            dropped = len(corr_matrix.columns) - len(usable)
            if len(clusterable) < 2:
                raise ValueError(f"only {len(clusterable)} feature(s) have defined correlations; "
                                 f"nothing to cluster. Pass cluster=False for a plain heatmap.")
            if dropped:
                self._log(f"  {dropped} constant/undefined feature(s) omitted from the clustered heatmap.")
            clustergrid = sns.clustermap(clusterable, figsize=figsize, **heatmap_kwargs)
            clustergrid.fig.suptitle(f"Clustered {title}", fontsize=16, y=0.99)
            return clustergrid

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr_matrix, ax=ax, square=True, **heatmap_kwargs)
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # 4. TRANSFORMATION RECOMMENDATIONS
    # ------------------------------------------------------------------

    def recommend_transformations(self, scope: Optional[Union[EDAScope, str]] = None) -> pd.DataFrame:
        """
        Recommend a transformation per feature from its train-partition distribution.

        Returns
        -------
        pd.DataFrame
            Columns ``feature``, ``transformations`` (comma separated), ``reason``.
        """
        scope = self._resolve_scope(scope)
        index = self._index_for(scope)

        self._log("\n" + "=" * 80)
        self._log("TRANSFORMATION RECOMMENDATIONS")
        self._log("=" * 80)

        recommendations = []
        for feat in self.feature_names:
            feat_data = _finite_series(self.features.loc[index, feat])
            if len(feat_data) < MIN_SAMPLES:
                continue

            transformations: List[str] = []
            reasons: List[str] = []

            is_bounded_0_1 = bool(feat_data.min() >= 0 and feat_data.max() <= 1)
            is_strictly_positive = bool(feat_data.min() > 0)
            skewness = float(stats.skew(feat_data))
            kurtosis = float(stats.kurtosis(feat_data))

            if is_bounded_0_1 and (feat_data.max() - feat_data.min()) > 0.5:
                transformations.append(str(Transformation.LOGIT))
                reasons.append("Bounded [0,1] ratio/oscillator")

            if is_strictly_positive and skewness > 1.0:
                transformations.append(str(Transformation.LOG))
                reasons.append(f"Right-skewed (skew={skewness:.2f})")

            if abs(skewness) > 1.5 or abs(kurtosis) > 3:
                transformations.append(str(Transformation.BOX_COX if is_strictly_positive
                                           else Transformation.YEO_JOHNSON))
                reasons.append(f"Non-normal (skew={skewness:.2f}, kurt={kurtosis:.2f})")

            # Tukey fence. The previous version counted observations outside the 1st/99th
            # percentiles, which is ~2% of the sample by construction and so could never
            # clear the 5% trigger -- this rule never actually fired.
            q25, q75 = feat_data.quantile([0.25, 0.75])
            iqr = q75 - q25
            if iqr > 0:
                outlier_ratio = float(((feat_data < q25 - 3 * iqr) | (feat_data > q75 + 3 * iqr)).mean())
                if outlier_ratio > 0.05:
                    transformations.append(QUANTILE_OR_RANK)
                    reasons.append(f"Heavy outliers ({outlier_ratio * 100:.1f}%)")

            if feat_data.std() > 0:
                mean = feat_data.mean()
                cv = feat_data.std() / abs(mean) if mean != 0 else np.inf
                if cv > 2:
                    transformations.append(str(Transformation.STANDARDIZE))
                    reasons.append(f"High variance (CV={cv:.2f})")

            if not transformations:
                transformations.append(str(Transformation.NONE))
                reasons.append("Well-behaved distribution")

            recommendations.append({"feature": feat, "transformations": ", ".join(transformations),
                                    "reason": " | ".join(reasons)})

        rec_df = pd.DataFrame(recommendations)
        self.transformation_results = rec_df

        self._log("\nTransformation Summary:")
        self._log("-" * 80)
        if rec_df.empty:
            self._log("None")
            return rec_df

        all_transforms = []
        for trans_str in rec_df["transformations"]:
            all_transforms.extend(t.strip() for t in trans_str.split(","))
        for trans, count in Counter(all_transforms).most_common():
            self._log(f"  {trans:<20}: {count} features")

        self._log("\nFeatures Needing Logit Transformation:")
        logit_feats = rec_df[rec_df["transformations"].str.contains("logit")]
        self._log(logit_feats[["feature", "reason"]].to_string(index=False) if len(logit_feats) else "  None")

        self._log("\nFeatures Needing Log Transformation:")
        log_feats = rec_df[rec_df["transformations"].str.contains("log")
                           & ~rec_df["transformations"].str.contains("logit")]
        self._log(log_feats[["feature", "reason"]].to_string(index=False) if len(log_feats) else "  None")

        return rec_df

    @staticmethod
    def _primary_transformation(trans_str: str) -> Optional[Transformation]:
        """First fittable transformation in a recommendation string, or None."""
        for token in (t.strip() for t in str(trans_str).split(",")):
            if token == QUANTILE_OR_RANK:
                return Transformation.QUANTILE
            try:
                transformation = Transformation(token)
            except ValueError:
                continue
            if transformation != Transformation.NONE:
                return transformation
        return None

    def fit_transformations(self, recommendations: Optional[pd.DataFrame] = None,
                            scope: Optional[Union[EDAScope, str]] = None, **kwargs) -> FittedTransformSpec:
        """
        Fit each recommended transformation on the train partition and freeze its parameters.

        This is the leakage-safe replacement for calling ``fit_transform`` over a whole frame.
        Every parameter -- Box-Cox lambda, log offset, z-score mean and standard deviation,
        quantile map, rank reference CDF -- is estimated from train rows only, so applying the
        returned spec to a later window cannot carry that window's information backwards.

        Parameters
        ----------
        recommendations : pd.DataFrame, optional
            Output of :meth:`recommend_transformations`. Computed on demand if omitted.
        scope : EDAScope or str, optional
            Partition to fit on. Defaults to the train partition.
        **kwargs
            Forwarded to the transformer constructors (``epsilon``, ``standardize``,
            ``output_distribution``, ``random_state``).

        Returns
        -------
        FittedTransformSpec
            Call ``.transform(df)`` on any window, including the holdout.
        """
        scope = self._resolve_scope(scope)
        fit_index = self._index_for(scope)

        if recommendations is None:
            was_verbose, self.verbose = self.verbose, False
            try:
                recommendations = self.recommend_transformations(scope=scope)
            finally:
                self.verbose = was_verbose

        spec = FittedTransformSpec(fitted_on=fit_index)
        for record in recommendations.to_dict("records"):
            feature = record["feature"]
            if feature not in self.features.columns:
                continue
            transformation = self._primary_transformation(record.get("transformations", ""))
            if transformation is None:
                continue

            fit_slice = _finite_series(self.features.loc[fit_index, feature])
            if len(fit_slice) < MIN_SAMPLES:
                spec.failures[feature] = f"only {len(fit_slice)} finite train observations"
                continue
            _warn_if_unsuitable(transformation, fit_slice, feature)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spec.transformers[feature] = _build_transformer(transformation, **kwargs).fit(fit_slice)
                spec.transformations[feature] = transformation
            except Exception as exc:
                spec.failures[feature] = str(exc)

        self._log(f"\nFitted {len(spec.transformers)} transformer(s) on {len(fit_index)} {scope} rows.")
        if spec.failures:
            self._log(f"Failed to fit {len(spec.failures)}: {list(spec.failures)[:10]}")
        return spec

    def apply_transformation(self, feature: str, transformation: Union[Transformation, str],
                             scope: Optional[Union[EDAScope, str]] = None, **kwargs) -> pd.Series:
        """
        Apply one transformation to one feature, fitting its parameters on the train partition.

        The returned Series still spans every aligned row, as before, but the parameters behind
        it come from train rows only -- so the value at a given bar no longer moves when later
        bars arrive.

        Parameters
        ----------
        feature : str
            Feature name.
        transformation : Transformation or str
            'logit', 'log', 'box-cox', 'yeo-johnson', 'quantile', 'rank', 'standardize'.
        scope : EDAScope or str, optional
            Partition to fit on. Defaults to the train partition.
        **kwargs
            Forwarded to the transformer constructor.

        Returns
        -------
        pd.Series
            Transformed feature over all aligned rows.

        Notes
        -----
        ``box-cox`` needs strictly positive input at transform time as well as at fit time. A
        holdout that dips to zero or below will raise rather than silently refit -- use
        ``yeo-johnson`` for features that can change sign.
        """
        transformation = Transformation(transformation)
        series = self.features[feature]
        if transformation == Transformation.NONE:
            return series.copy()

        scope = self._resolve_scope(scope)
        fit_slice = _finite_series(self.features.loc[self._index_for(scope), feature])
        if fit_slice.empty:
            raise ValueError(f"feature '{feature}' has no finite values in the {scope} partition")

        _warn_if_unsuitable(transformation, fit_slice, feature)
        transformer = _build_transformer(transformation, **kwargs).fit(fit_slice)
        result = transformer.transform(series)
        if not isinstance(result, pd.Series):
            result = pd.Series(np.asarray(result).ravel(), index=series.index)
        return result.rename(f"{feature}_{_SUFFIX[transformation]}")

    # ------------------------------------------------------------------
    # COMPREHENSIVE REPORT
    # ------------------------------------------------------------------

    def generate_comprehensive_report(self, output_path: Optional[str] = None,
                                      n_top_features: int = 20) -> Dict:
        """
        Run every analysis on the train partition and collect the results.

        Parameters
        ----------
        output_path : str, optional
            If given, an HTML rendering of the report is written here. (The previous version
            accepted this argument and silently discarded it.)
        n_top_features : int, default 20
            Number of top features to highlight.

        Returns
        -------
        Dict
            All analysis frames plus the :class:`LeakageManifest` under ``'manifest'``.
        """
        self._log("\n" + "=" * 80)
        self._log("GENERATING COMPREHENSIVE EDA REPORT")
        self._log("=" * 80)
        self._log("")
        self._log(self.manifest.describe())

        self._log("\n[1/5] Analyzing feature relevance...")
        relevance_df = self.analyze_feature_relevance(n_top=n_top_features)

        self._log("\n[2/5] Analyzing distributions...")
        distribution_df = self.analyze_distributions(n_features=n_top_features)

        self._log("\n[3/5] Analyzing correlations...")
        corr_matrix, high_corr_pairs = self.analyze_correlation(threshold=0.8)

        self._log("\n[4/5] Computing VIF...")
        vif_df = self.compute_vif(vif_threshold=10.0)

        self._log("\n[5/5] Recommending transformations...")
        transformation_df = self.recommend_transformations()

        report = {
            "manifest": self.manifest,
            "relevance": relevance_df,
            "distribution": distribution_df,
            "correlation_matrix": corr_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "vif": vif_df,
            "transformations": transformation_df,
        }

        if output_path is not None:
            self._write_html_report(report, output_path)
            self._log(f"\nReport written to {output_path}")

        self._log("\n" + "=" * 80)
        self._log("REPORT GENERATION COMPLETE")
        self._log("=" * 80)

        return report

    @staticmethod
    def _write_html_report(report: Dict, output_path: str) -> None:
        """Render the report frames to a single self-contained HTML file."""
        parts = ["<html><head><meta charset='utf-8'><title>Feature EDA Report</title>",
                 "<style>body{font-family:system-ui,sans-serif;margin:2rem}",
                 "table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #ddd;padding:4px}",
                 "pre{background:#f6f6f6;padding:1rem}</style></head><body>",
                 "<h1>Feature EDA Report</h1>",
                 f"<pre>{report['manifest'].describe()}</pre>"]
        for key, value in report.items():
            if key == "manifest":
                continue
            parts.append(f"<h2>{key}</h2>")
            if isinstance(value, pd.DataFrame):
                parts.append(value.to_html(index=False))
            else:
                parts.append(f"<pre>{value}</pre>")
        parts.append("</body></html>")
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(parts))



# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def quick_eda(features: pd.DataFrame, target: Union[pd.Series, np.ndarray],
              target_type: Union[TargetType, str] = TargetType.CONTINUOUS, n_top: int = 20, **kwargs) -> FeatureEDA:
    """
    Quick leakage-safe EDA with default settings.

    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame.
    target : pd.Series or np.ndarray
        Target variable.
    target_type : TargetType or str, default TargetType.CONTINUOUS
    n_top : int, default 20
        Number of top features to analyze.
    **kwargs
        Forwarded to :class:`FeatureEDA` (``horizon``, ``mode``, ``train_threshold``, ...).

    Returns
    -------
    FeatureEDA
        EDA object with all results populated.
    """
    eda = FeatureEDA(features, target, target_type, **kwargs)
    eda.generate_comprehensive_report(n_top_features=n_top)
    return eda
