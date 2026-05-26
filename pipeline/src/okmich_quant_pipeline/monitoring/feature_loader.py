"""OOS feature-baseline loader for the streaming monitor.

The feature-side drift gate (``score_feature_health``) needs a reference sample
of the OOS feature distribution. We do NOT persist it in ``metadata.json`` —
feature samples are large (~MB at typical training sizes) and deterministic
from (OHLCV, feature_engineering, transform_pipeline). The monitor re-derives
the baseline at runtime from the persisted OOS-window definition.
"""
from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from okmich_quant_ml.posterior_inference import FeatureHealthBaselines, fit_feature_health_baselines


def _resolve_callable(spec: str) -> Callable:
    """Resolve a 'pkg.module:func_name' spec to the actual callable."""
    if ":" not in spec:
        raise ValueError(f"_resolve_callable: spec must be 'pkg.module:func', got {spec!r}")
    module_path, func_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    if not hasattr(module, func_name):
        raise AttributeError(f"_resolve_callable: module {module_path!r} has no attribute {func_name!r}")
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"_resolve_callable: {spec!r} is not callable")
    return func


def derive_feature_baseline_from_oos_window(
        raw_data_dir: Path, symbol: str, oos_start_ts: str, oos_end_ts: str,
        feature_columns: list[str], feature_engineering_callable: str,
        transform_pipeline: Any) -> FeatureHealthBaselines:
    """Re-derive ``FeatureHealthBaselines`` from the OOS window in the data lake.

    Reads ``<raw_data_dir>/<symbol>.parquet``, slices to ``[oos_start_ts, oos_end_ts]``
    inclusive, runs the strategy's feature-engineering function, applies the persisted
    ``transform_pipeline`` (StandardScaler), and fits the feature baseline.

    The feature-engineering function is resolved dynamically from
    ``feature_engineering_callable`` (e.g. ``"my_pkg.features:compute"``) so the
    monitor stays family-agnostic. The function must accept
    ``(df: DataFrame, feature_columns: list[str]) -> DataFrame`` and return a frame
    containing at least ``feature_columns``.

    Raises ``FileNotFoundError`` if the parquet is missing, ``ValueError`` if the
    OOS slice is empty after timestamp filter, or downstream errors from the
    feature pipeline.
    """
    parquet_path = raw_data_dir / f"{symbol}.parquet"
    if not parquet_path.is_file():
        raise FileNotFoundError(
            f"derive_feature_baseline_from_oos_window: OHLCV parquet not found at {parquet_path}"
        )
    raw_df = pd.read_parquet(parquet_path)

    # Slice to OOS window. Use timestamps from metadata.json; both inclusive.
    oos_start = pd.Timestamp(oos_start_ts)
    oos_end = pd.Timestamp(oos_end_ts)
    # Normalise to common tz handling — match the index tz if any.
    if raw_df.index.tz is None:
        oos_start = oos_start.tz_convert(None) if oos_start.tz is not None else oos_start
        oos_end = oos_end.tz_convert(None) if oos_end.tz is not None else oos_end
    else:
        if oos_start.tz is None:
            oos_start = oos_start.tz_localize(raw_df.index.tz)
        else:
            oos_start = oos_start.tz_convert(raw_df.index.tz)
        if oos_end.tz is None:
            oos_end = oos_end.tz_localize(raw_df.index.tz)
        else:
            oos_end = oos_end.tz_convert(raw_df.index.tz)

    oos_raw = raw_df.loc[oos_start:oos_end]
    if len(oos_raw) == 0:
        raise ValueError(
            f"derive_feature_baseline_from_oos_window: OOS slice [{oos_start_ts}, {oos_end_ts}] is empty "
            f"in {parquet_path} (raw_df has {len(raw_df)} rows from {raw_df.index[0]} to {raw_df.index[-1]})"
        )

    compute_features = _resolve_callable(feature_engineering_callable)
    feat_df = compute_features(oos_raw.copy(), feature_columns).dropna(subset=feature_columns)
    if len(feat_df) == 0:
        raise ValueError(
            f"derive_feature_baseline_from_oos_window: no rows left after feature engineering + dropna "
            f"on OOS slice for {symbol}"
        )

    x_oos = feat_df[feature_columns].to_numpy(dtype=float)
    x_oos_scaled = transform_pipeline.transform(x_oos)
    return fit_feature_health_baselines(x_oos_scaled, feature_names=list(feature_columns))


def load_transform_pipeline(artifact_dir: Path) -> Any:
    """Load the persisted ``transform_pipeline.joblib`` from a model artefact dir."""
    pipeline_path = artifact_dir / "transform_pipeline.joblib"
    if not pipeline_path.is_file():
        raise FileNotFoundError(
            f"load_transform_pipeline: transform_pipeline.joblib not found at {pipeline_path}"
        )
    return joblib.load(pipeline_path)
