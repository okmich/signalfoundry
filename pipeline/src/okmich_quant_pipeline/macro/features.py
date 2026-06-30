"""Daily macro feature engineering — raw levels into model-ready conditioners.

Pure, IO-free. Turns the long raw frame (``reader.load_macro``) into a long feature
frame ``[date, feature, value, available_from_utc]``, one row per (date, feature).

Two design rules make this generic and leak-free:

1. **Compute on observation cadence, never on the broadcast intraday series.** A 20-point
   rolling window is 20 *observations* — 20 trading days for a daily series, 20 weeks for a
   weekly one. Windows are expressed in observations, so the same transforms apply to any
   cadence. Computing a "20-day" window on a forward-filled intraday series would instead
   window 20 bars (= 100 minutes on M5) — the classic exogenous-feature bug.

2. **Per-feature availability = max over its source series' availabilities.** A feature can't
   be consumed until every series it derives from is public. For single-source features this
   is just that series' stamp; for a cross-series feature (e.g. the VIX term ratio) it is the
   later of the two.

Adding a feature is data-only: append a ``FeatureRecipe``. Adding a series is data-only:
a ``SeriesSpec`` in ``_types`` plus any recipes referencing it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from okmich_quant_pipeline.macro._types import MacroSeries

# --------------------------------------------------------------------------- #
# Generic transforms (each takes one or more observation-indexed Series)
# --------------------------------------------------------------------------- #

def level(s: pd.Series) -> pd.Series:
    """Pass-through raw level."""
    return s


def zscore(s: pd.Series, window: int) -> pd.Series:
    """Trailing (causal) z-score over ``window`` observations; warmup region is NaN."""
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std()
    return (s - mean) / std


def change(s: pd.Series, periods: int) -> pd.Series:
    """Absolute change over ``periods`` observations: ``s - s.shift(periods)``."""
    return s - s.shift(periods)


def log_return(s: pd.Series, periods: int) -> pd.Series:
    """Log return over ``periods`` observations: ``log(s / s.shift(periods))``."""
    return np.log(s / s.shift(periods))


def ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Cross-series ratio (auto-aligned on the observation index)."""
    return num / den


def spread(a: pd.Series, b: pd.Series) -> pd.Series:
    """Cross-series difference ``a - b`` (auto-aligned), e.g. a yield-curve slope."""
    return a - b


# --------------------------------------------------------------------------- #
# Recipes
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureRecipe:
    """One output feature: its name, source series, and a transform of those series.

    ``fn`` receives one Series per entry in ``sources`` (in order) and returns the feature
    Series. Parameters (windows, periods) are bound into ``fn`` (e.g. via ``partial``), so
    each feature owns its own window — no single global window is imposed.
    """

    name: str
    sources: tuple[MacroSeries, ...]
    fn: Callable[..., pd.Series]


DEFAULT_RECIPES: tuple[FeatureRecipe, ...] = (
    FeatureRecipe("vix_level", (MacroSeries.VIX,), level),
    FeatureRecipe("vix_z20", (MacroSeries.VIX,), partial(zscore, window=20)),
    FeatureRecipe("vix_chg5", (MacroSeries.VIX,), partial(change, periods=5)),
    FeatureRecipe("vixts_ratio", (MacroSeries.VIX, MacroSeries.VIX_3M), ratio),
    FeatureRecipe("vixts_z20", (MacroSeries.VIX, MacroSeries.VIX_3M), lambda v, v3: zscore(ratio(v, v3), 20)),
    FeatureRecipe("credit_level", (MacroSeries.CREDIT_SPREAD,), level),
    FeatureRecipe("credit_z20", (MacroSeries.CREDIT_SPREAD,), partial(zscore, window=20)),
    FeatureRecipe("credit_chg5", (MacroSeries.CREDIT_SPREAD,), partial(change, periods=5)),
    FeatureRecipe("usd_ret5", (MacroSeries.USD_BROAD,), partial(log_return, periods=5)),
    FeatureRecipe("usd_z20", (MacroSeries.USD_BROAD,), partial(zscore, window=20)),
    # Rates: levels + 5-day changes; the 2s10s slope and its z-score (inversion regime).
    FeatureRecipe("us2y_level", (MacroSeries.US_2Y,), level),
    FeatureRecipe("us2y_chg5", (MacroSeries.US_2Y,), partial(change, periods=5)),
    FeatureRecipe("us10y_level", (MacroSeries.US_10Y,), level),
    FeatureRecipe("us10y_chg5", (MacroSeries.US_10Y,), partial(change, periods=5)),
    FeatureRecipe("curve_2s10s", (MacroSeries.US_10Y, MacroSeries.US_2Y), spread),
    FeatureRecipe("curve_2s10s_z20", (MacroSeries.US_10Y, MacroSeries.US_2Y), lambda a, b: zscore(spread(a, b), 20)),
    # Financial conditions (weekly): level + 4-week change. Cadence handled by the asof-merge.
    FeatureRecipe("nfci_level", (MacroSeries.NFCI,), level),
    FeatureRecipe("nfci_chg4", (MacroSeries.NFCI,), partial(change, periods=4)),
)


# Opt-in ICE BofA HY-OAS recipes — deliberately NOT in DEFAULT_RECIPES. The FRED series is
# licence-capped to a rolling ~3y window (no pre-2023 history), so folding it into the defaults would
# NaN-truncate any longer macro-joined dataset via the attach's drop_warmup. Concat explicitly
# (``DEFAULT_RECIPES + HY_OAS_RECIPES``) only for 2023+ work that wants the high-yield credit gauge.
HY_OAS_RECIPES: tuple[FeatureRecipe, ...] = (
    FeatureRecipe("hy_oas_level", (MacroSeries.HY_OAS,), level),
    FeatureRecipe("hy_oas_z20", (MacroSeries.HY_OAS,), partial(zscore, window=20)),
    FeatureRecipe("hy_oas_chg5", (MacroSeries.HY_OAS,), partial(change, periods=5)),
)


def compute_macro_features(raw: pd.DataFrame, recipes: tuple[FeatureRecipe, ...] = DEFAULT_RECIPES) -> pd.DataFrame:
    """Compute conditioning features from the long raw macro frame.

    Parameters
    ----------
    raw
        Long frame with columns ``date``, ``series``, ``value``, ``available_from_utc``
        (the output of ``reader.load_macro``).
    recipes
        Feature definitions. Defaults to ``DEFAULT_RECIPES``.

    Returns
    -------
    pd.DataFrame
        Long frame ``[date, feature, value, available_from_utc]``, warmup/NaN rows dropped,
        sorted by ``(feature, date)``.
    """
    # One groupby pass; date-indexed and sorted. Keys are the raw "series" strings, which a
    # MacroSeries (a StrEnum) indexes into transparently (member == its string value).
    grouped = {s: g.set_index("date").sort_index() for s, g in raw.groupby("series")}

    frames: list[pd.DataFrame] = []
    for recipe in recipes:
        missing = [s for s in recipe.sources if s not in grouped]
        if missing:
            raise KeyError(f"recipe '{recipe.name}' needs series {missing} not present in raw frame")

        feat = recipe.fn(*[grouped[s]["value"] for s in recipe.sources]).rename("value")

        # Availability = elementwise latest over the source series (a cross-series feature is only
        # public once every leg is); single-source reduces to that series' stamp. Kept as a
        # tz-aware Series throughout (no .to_numpy round-trip) so the UTC dtype is preserved natively.
        src_avail = pd.concat([grouped[s]["available_from_utc"] for s in recipe.sources], axis=1)
        feat_avail = src_avail.max(axis=1).reindex(feat.index).rename("available_from_utc")

        frame = pd.concat([feat, feat_avail], axis=1).reset_index(names="date")
        frame["feature"] = recipe.name
        frames.append(frame.dropna(subset=["value"]))

    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["feature", "date"]).reset_index(drop=True)
