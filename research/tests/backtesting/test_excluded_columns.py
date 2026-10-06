"""Tests for the excluded-column selection in ClusteringComparisonPipeline.

``columns_scaling_exclude`` is the caller's list of non-feature columns, and it is reasonable for it
to name every such column of the source parquet. ``get_data`` deletes ``spread`` and ``real_volume``
before anything else sees the frame, so the two lists legitimately differ.

Selecting the config list verbatim therefore raised ``KeyError: "['spread'] not in index"`` -- and
raised it at the END of ``run()``, after every model had already been fitted, so a multi-hour fit was
lost to a column nothing downstream reads. These lock the intersection in place.
"""

import pandas as pd
import pytest

from okmich_quant_research.backtesting.cluster_comparison_pipeline import (
    ClusteringComparisonPipeline, ClusteringComparisonPipelineConfig)

OHLCV = ["open", "high", "low", "close", "tick_volume"]


def _pipeline(exclude):
    """A pipeline instance carrying only the config -- no data, no models, no fitting."""
    cfg = ClusteringComparisonPipelineConfig(columns_scaling_exclude=exclude)
    p = ClusteringComparisonPipeline.__new__(ClusteringComparisonPipeline)
    p.pipeline_config = cfg
    return p


def _frame(cols):
    return pd.DataFrame({c: [1.0, 2.0] for c in cols})


def test_excluded_present_drops_columns_the_loader_removed():
    # The exact shape that broke: config names `spread`, get_data has already deleted it.
    p = _pipeline(OHLCV + ["spread"])
    assert p._excluded_present(_frame(OHLCV)) == OHLCV


def test_excluded_present_is_usable_as_a_selector():
    p = _pipeline(OHLCV + ["spread", "real_volume"])
    df = _frame(OHLCV)
    assert list(df[p._excluded_present(df)].columns) == OHLCV


def test_excluded_present_preserves_config_order_not_frame_order():
    p = _pipeline(["close", "open"])
    assert p._excluded_present(_frame(OHLCV)) == ["close", "open"]


def test_excluded_present_handles_no_overlap():
    assert _pipeline(["spread"])._excluded_present(_frame(OHLCV)) == []


def test_config_default_omits_spread_because_get_data_drops_it():
    """The library's own default is already spread-free. That is not an oversight -- it is the
    reason the verbatim select survived this long, and why a caller who sensibly ADDS `spread`
    (it is a real column of the parquet) is the one who hits the failure."""
    assert ClusteringComparisonPipelineConfig().columns_scaling_exclude == OHLCV
    assert _pipeline(None)._excluded_present(_frame(OHLCV)) == OHLCV


def test_get_data_really_does_drop_spread():
    """The premise of the bug. If this ever changes, the intersection is still correct but the
    comment explaining WHY it is needed is not -- so assert the loader's behaviour directly."""
    src = ClusteringComparisonPipeline.get_data.__doc__ or ""
    import inspect
    body = inspect.getsource(ClusteringComparisonPipeline.get_data)
    assert 'drop(columns=["spread"]' in body, "get_data no longer drops spread -- revisit the guard"


def test_attach_excluded_columns_keeps_only_the_block_rows():
    """Right join, deliberately: a caller asking for the train block wants the train rows."""
    p = _pipeline(OHLCV + ["spread"])
    df = pd.DataFrame({c: [1.0, 2.0, 3.0, 4.0] for c in OHLCV})
    p.pipeline_config.append_excluded_col_in_result = True
    block = pd.DataFrame({"lbl": [0, 1]}, index=[2, 3])
    out = p._attach_excluded_columns(df, block)
    assert list(out.index) == [2, 3]
    assert "spread" not in out.columns
    assert set(OHLCV + ["lbl"]) == set(out.columns)
