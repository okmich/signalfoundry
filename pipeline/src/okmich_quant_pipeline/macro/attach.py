"""Lab-side wrapper — attach macro features onto a DatasetBuilder output.

This is the only IO-touching piece of the consume path; ``features``/``align`` stay pure.
It runs lab-side (NOT inside source ``dataset_builder``) to preserve the lab -> source
dependency direction and keep the live pipeline untouched: the attach is a reversible
``+ columns`` step on the same base dataset, which also makes the macro-vs-no-macro
ablation trivial.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from okmich_quant_pipeline.macro.align import attach_exogenous
from okmich_quant_pipeline.macro.features import DEFAULT_RECIPES, FeatureRecipe
from okmich_quant_pipeline.macro.reader import load_macro_features


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
