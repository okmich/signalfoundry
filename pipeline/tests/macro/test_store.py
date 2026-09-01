"""Feature-store materialization + coverage-report tests.

Covers the S1 writer (`store.build_feature_store`, atomic write) and the S2 report
(`report.build_report`: cadence inference, gap detection, staleness, feature density).
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.macro._types import SERIES, MacroSeries
from okmich_quant_pipeline.macro.features import FeatureRecipe, level, ratio
from okmich_quant_pipeline.macro.report import (
    REPORT_FILENAME,
    Cadence,
    _infer_cadence,
    build_report,
)
from okmich_quant_pipeline.macro.store import FEATURES_FILENAME, build_feature_store


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _write_series(store_dir, series: MacroSeries, dates, values) -> None:
    """Write one per-series parquet shaped exactly like `reader.load_macro` expects."""
    df = pd.DataFrame({"date": pd.to_datetime(dates), "series": series.value, "value": values})
    df["available_from_utc"] = SERIES[series].availability.stamp(df)
    store_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(store_dir / f"{series.value}.parquet", index=False)


def _raw(series: str, dates, value: float = 1.0) -> pd.DataFrame:
    dates = pd.to_datetime(dates)
    out = pd.DataFrame({"series": series, "date": dates, "value": value})
    out["available_from_utc"] = pd.to_datetime(out["date"]).dt.tz_localize("UTC")
    return out


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "feature": pd.Series([], dtype=str),
                         "value": pd.Series([], dtype=float), "available_from_utc": pd.to_datetime([])})


# --------------------------------------------------------------------------- #
# S3.1 — store writer / atomic write
# --------------------------------------------------------------------------- #

def test_atomic_write_parquet_roundtrips_and_leaves_no_tmp(tmp_path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "x.parquet"
    atomic_write_parquet(df, path)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))  # unique temp consumed by the rename, none left behind
    pd.testing.assert_frame_equal(pd.read_parquet(path), df)


def test_atomic_write_parquet_no_partial_file_on_failure(tmp_path, monkeypatch) -> None:
    path = tmp_path / "x.parquet"

    def boom(*_a, **_k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", boom)
    with pytest.raises(RuntimeError):
        atomic_write_parquet(pd.DataFrame({"a": [1]}), path)
    assert not path.exists()  # os.replace never ran -> no file under the final name
    assert not list(tmp_path.glob("*.tmp"))  # failed write cleans up its temp


def test_build_feature_store_roundtrip(tmp_path) -> None:
    macro, out = tmp_path / "macro", tmp_path / "feat"
    dates = pd.bdate_range("2024-01-01", periods=60)
    _write_series(macro, MacroSeries.VIX, dates, 15.0 + np.arange(60) * 0.1)
    _write_series(macro, MacroSeries.VIX_3M, dates, 16.0 + np.arange(60) * 0.1)

    recipes = (
        FeatureRecipe("vix_level", (MacroSeries.VIX,), level),
        FeatureRecipe("vixts_ratio", (MacroSeries.VIX, MacroSeries.VIX_3M), ratio),
    )
    # last synthetic obs is Fri 2024-03-22; asof the next Monday keeps it fresh (3 calendar days).
    report = build_feature_store(macro, out, recipes=recipes, asof=dt.date(2024, 3, 25))

    fp = out / FEATURES_FILENAME
    assert fp.exists() and (out / REPORT_FILENAME).exists()
    feats = pd.read_parquet(fp)
    assert set(feats["feature"].unique()) == {"vix_level", "vixts_ratio"}
    # tz-aware UTC stamp survives the parquet round-trip (the causal key must not be dropped/naive).
    assert str(feats["available_from_utc"].dtype) == "datetime64[ns, UTC]"
    assert {s.series for s in report.series} == {"VIX", "VIX_3M"}
    assert report.has_stale is False


# --------------------------------------------------------------------------- #
# S3.2 — cadence inference
# --------------------------------------------------------------------------- #

def test_infer_cadence() -> None:
    assert _infer_cadence(pd.Series(pd.bdate_range("2024-01-01", periods=40))) is Cadence.BUSINESS_DAILY
    assert _infer_cadence(pd.Series(pd.date_range("2024-01-05", periods=20, freq="7D"))) is Cadence.WEEKLY
    assert _infer_cadence(pd.Series(pd.to_datetime(["2024-01-01"]))) is Cadence.IRREGULAR


# --------------------------------------------------------------------------- #
# S3.2 — gap detection + staleness (daily)
# --------------------------------------------------------------------------- #

def test_report_flags_internal_gap_but_not_long_weekend(tmp_path) -> None:
    # Two daily runs with a ~2-week hole between them; last obs == asof so it is not tail-stale.
    first = pd.bdate_range("2024-01-01", "2024-02-15")
    second = pd.bdate_range("2024-03-01", "2024-04-15")
    dates = first.append(second)
    rep = build_report(_raw("VIX", dates), _empty_features(), asof=dt.date(2024, 4, 15))
    sc = rep.series[0]
    assert sc.cadence == Cadence.BUSINESS_DAILY.value
    assert sc.n_gaps == 1 and sc.max_gap_days > 5
    assert sc.is_stale is False


def test_report_long_weekend_is_not_stale() -> None:
    # Last obs Fri 2024-05-24, asof Tue 2024-05-28 (Memorial Day Mon) -> 4 calendar days < 5.
    dates = pd.bdate_range("2024-04-01", "2024-05-24")
    rep = build_report(_raw("VIX", dates), _empty_features(), asof=dt.date(2024, 5, 28))
    sc = rep.series[0]
    assert sc.staleness_days == 4 and sc.is_stale is False


def test_report_staleness_true_when_feed_stalls() -> None:
    dates = pd.bdate_range("2024-01-01", periods=30)  # last obs ~2024-02-09
    rep = build_report(_raw("VIX", dates), _empty_features(), asof=dt.date(2024, 3, 1))
    assert rep.series[0].is_stale is True and rep.has_stale is True


def test_report_coverage_pct_is_high_for_clean_daily() -> None:
    dates = pd.bdate_range("2024-01-01", periods=100)
    rep = build_report(_raw("VIX", dates), _empty_features(), asof=dates[-1].date())
    sc = rep.series[0]
    assert sc.expected_obs == 100 and sc.coverage_pct == 100.0  # no holidays punched in synthetic bdays


# --------------------------------------------------------------------------- #
# S3.2 — weekly cadence must not false-flag weekday "gaps"
# --------------------------------------------------------------------------- #

def test_report_weekly_cadence_no_false_gaps() -> None:
    dates = pd.date_range("2024-01-05", periods=20, freq="7D")
    rep = build_report(_raw("NFCI", dates), _empty_features(), asof=dates[-1].date())
    sc = rep.series[0]
    assert sc.cadence == Cadence.WEEKLY.value
    assert sc.n_gaps == 0 and sc.is_stale is False


def test_report_weekly_staleness_true_after_missed_release() -> None:
    dates = pd.date_range("2024-01-05", periods=20, freq="7D")
    asof = (dates[-1] + pd.Timedelta(days=21)).date()  # 21 > weekly threshold 14
    rep = build_report(_raw("NFCI", dates), _empty_features(), asof=asof)
    assert rep.series[0].is_stale is True


# --------------------------------------------------------------------------- #
# S3.2 — feature density
# --------------------------------------------------------------------------- #

def test_feature_density_reflects_cadence() -> None:
    daily = pd.bdate_range("2024-01-01", periods=10)
    rows = [{"date": d, "feature": "vix_level", "value": 1.0, "available_from_utc": pd.Timestamp(d, tz="UTC")} for d in daily]
    rows += [{"date": d, "feature": "nfci_level", "value": 1.0, "available_from_utc": pd.Timestamp(d, tz="UTC")} for d in daily[::2]]
    feats = pd.DataFrame(rows)
    rep = build_report(_raw("VIX", daily), feats, asof=dt.date(2024, 2, 1))
    dens = {f.feature: f.density_pct for f in rep.features}
    assert dens["vix_level"] == 100.0  # present on every date
    assert dens["nfci_level"] == 50.0  # present on half


def test_report_to_dict_is_json_serializable() -> None:
    import json

    rep = build_report(_raw("VIX", pd.bdate_range("2024-01-01", periods=10)), _empty_features(),
                       asof=dt.date(2024, 1, 20))
    json.dumps(rep.to_dict())  # must not raise
