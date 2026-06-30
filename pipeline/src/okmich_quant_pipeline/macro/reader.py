"""Load the macro store (per-series parquets) back into the long frame.

``load_macro`` concatenates every ``{SERIES}.parquet`` in the store dir; ``load_macro_features``
additionally engineers the conditioning features. The no-lookahead broadcast onto intraday bars
lives in ``align.attach_exogenous`` / ``attach.attach_macro_to_dataset``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from okmich_quant_pipeline.macro._types import MacroSeries
from okmich_quant_pipeline.macro.features import DEFAULT_RECIPES, FeatureRecipe, compute_macro_features


def load_macro(store_dir: Path) -> pd.DataFrame:
    """Load every per-series parquet in ``store_dir`` and concat to the long frame.

    Returns columns ``date`` (datetime64[ns]), ``series`` (str), ``value`` (float),
    ``available_from_utc`` (datetime64[ns, UTC]), sorted by ``(series, date)``.
    """
    store_dir = Path(store_dir)
    frames = []
    for series in MacroSeries:
        path = store_dir / f"{series.value}.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        raise FileNotFoundError(f"No macro series parquets found in {store_dir} (run `fetch-macro-data --full`)")
    df = pd.concat(frames, ignore_index=True)
    df["available_from_utc"] = pd.to_datetime(df["available_from_utc"], utc=True)
    return df.sort_values(["series", "date"]).reset_index(drop=True)


def load_macro_features(store_dir: Path, recipes: tuple[FeatureRecipe, ...] = DEFAULT_RECIPES) -> pd.DataFrame:
    """Load the store and compute conditioning features in one step.

    Returns the long feature frame ``[date, feature, value, available_from_utc]``.
    """
    return compute_macro_features(load_macro(store_dir), recipes)


def pivot_values(df: pd.DataFrame) -> pd.DataFrame:
    """Wide view: one column per series, indexed by observation ``date`` (exploration only).

    Drops the ``available_from_utc`` causal stamp, so do the no-lookahead asof-merge on the long
    frame instead of feeding this to models.
    """
    wide = df.pivot_table(index="date", columns="series", values="value")
    return wide.reindex(columns=[s.value for s in MacroSeries])
