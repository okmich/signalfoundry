"""Wrappers — attach exogenous features onto a DatasetBuilder output.

The only IO-touching pieces of the consume path; ``features``/``align`` stay pure. Kept as a
post-step on the dataset (NOT folded into ``dataset_builder``) so each attach is a reversible
``+ columns`` step on the same base frame — which makes the with-vs-without ablation trivial. The
macro-series attach and the event-timing attach are deliberately *separate* functions so
``± macro columns`` and ``± event columns`` ablate independently.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from okmich_quant_pipeline.macro.align import attach_exogenous
from okmich_quant_pipeline.macro.features import DEFAULT_RECIPES, FeatureRecipe
from okmich_quant_pipeline.macro.reader import load_macro_features
from okmich_quant_pipeline.news_calendar._types import ImpactTier
from okmich_quant_pipeline.news_calendar.features import compute_event_features
from okmich_quant_pipeline.news_calendar.reader import load_calendar


def attach_macro_to_dataset(dataset: pd.DataFrame, macro_path: Path, *,
                            recipes: tuple[FeatureRecipe, ...] = DEFAULT_RECIPES,
                            drop_warmup: bool = True, prefix: str = "macro_",
                            max_staleness: pd.Timedelta | None = None) -> pd.DataFrame:
    """Load macro features and attach them onto a processed dataset (UTC-indexed bars).

    Source ``dataset_builder._trim`` keys NaN-drops off the ``feat_/tm_/candle_/temporal_``
    prefixes and runs before this attach, so ``macro_*`` warmup NaNs are not auto-dropped.
    With ``drop_warmup=True`` (default) rows where any ``macro_*`` column is NaN are dropped
    here, keeping the modeling frame dense; set ``False`` for diagnostics. ``max_staleness``
    is forwarded to ``attach_exogenous`` (NaN out macro on bars whose latest macro observation
    is older than the bound — guards a stale store).
    """
    features = load_macro_features(macro_path, recipes)
    out = attach_exogenous(dataset, features, prefix=prefix, max_staleness=max_staleness)
    if drop_warmup:
        macro_cols = [c for c in out.columns if c.startswith(prefix)]
        out = out.dropna(subset=macro_cols)
    return out


def attach_events_to_dataset(dataset: pd.DataFrame, calendar_path: Path, *, prefix: str = "macro_event_",
                             tiers: tuple[ImpactTier, ...] = (ImpactTier.HIGH,), blackout_bars: int = 3,
                             bar_seconds: int = 300, horizon_minutes: int = 24 * 60) -> pd.DataFrame:
    """Attach per-bar event-timing columns (``{prefix}minutes_to_next`` / ``_minutes_since_last`` /
    ``_blackout``) onto a UTC-indexed dataset.

    Computed directly on ``dataset.index`` against the news calendar (not via the asof-merge), and
    saturated (no warmup NaN), so there is no ``drop_warmup`` step. Kept separate from
    ``attach_macro_to_dataset`` so the event channel ablates independently of the macro series.
    ``bar_seconds`` should match the dataset timeframe (5m → 300) so the blackout radius is correct.
    """
    calendar = load_calendar(calendar_path)
    feats = compute_event_features(dataset.index, calendar, tiers=tiers, blackout_bars=blackout_bars,
                                   bar_seconds=bar_seconds, horizon_minutes=horizon_minutes).add_prefix(prefix)
    return pd.concat([dataset, feats], axis=1)
