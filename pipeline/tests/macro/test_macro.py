"""Tests for the macro feature-engineering + no-lookahead asof-merge.

The leakage test (``test_no_lookahead``) is the reason this package exists. The cadence
tests prove the merge is generic across daily / weekly / irregular release schedules.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline.macro._types import (
    SERIES,
    BusinessDayLag,
    CalendarDayLag,
    ExplicitRelease,
    MacroSeries,
)
from okmich_quant_pipeline.macro.align import attach_exogenous
from okmich_quant_pipeline.macro.features import (
    DEFAULT_RECIPES,
    FeatureRecipe,
    compute_macro_features,
    zscore,
)
from okmich_quant_pipeline.macro.metastore import MacroMetastore
from okmich_quant_pipeline.macro.reader import load_macro
from okmich_quant_pipeline.macro.update import _merge


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _utc(ts: str) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def make_bars(timestamps: list[str]) -> pd.DataFrame:
    idx = pd.DatetimeIndex([_utc(t) for t in timestamps], name="timestamp")
    return pd.DataFrame({"close": np.arange(len(idx), dtype=float)}, index=idx)


def make_features(rows: list[tuple[str, float, str]]) -> pd.DataFrame:
    """rows = [(feature, value, available_from_utc), ...]"""
    return pd.DataFrame(
        [{"feature": f, "value": v, "available_from_utc": _utc(a)} for f, v, a in rows]
    )


def synth_raw(periods: int = 40) -> pd.DataFrame:
    """Synthetic raw long frame for all four series, stamped via real policies."""
    dates = pd.bdate_range("2024-01-01", periods=periods)
    base = {MacroSeries.VIX: 15.0, MacroSeries.VIX_3M: 17.0,
            MacroSeries.CREDIT_SPREAD: 1.5, MacroSeries.USD_BROAD: 120.0,
            MacroSeries.US_2Y: 1.0, MacroSeries.US_10Y: 3.0, MacroSeries.NFCI: -0.2}
    frames = []
    for s in MacroSeries:
        df = pd.DataFrame({"date": dates, "series": s.value,
                           "value": base[s] + np.arange(periods) * 0.1})
        df["available_from_utc"] = SERIES[s].availability.stamp(df)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# align — the core no-lookahead property
# --------------------------------------------------------------------------- #

def test_no_lookahead() -> None:
    feats = make_features([
        ("f", 10.0, "2024-01-02 22:00"),
        ("f", 20.0, "2024-01-03 22:00"),
        ("f", 30.0, "2024-01-04 22:00"),
    ])
    bars = make_bars([
        "2024-01-02 21:00",  # before first release -> NaN
        "2024-01-02 22:00",  # exactly at release  -> 10 (backward is <=)
        "2024-01-03 10:00",  # between             -> 10 (NOT 20)
        "2024-01-03 22:00",  # next release        -> 20
        "2024-01-04 10:00",  # between             -> 20
        "2024-01-05 10:00",  # after last          -> 30
    ])
    out = attach_exogenous(bars, feats)
    got = out["macro_f"].tolist()
    assert got[0] != got[0]  # NaN
    assert got[1:] == [10.0, 10.0, 20.0, 20.0, 30.0]


def test_boundary_is_inclusive_at_release_instant() -> None:
    feats = make_features([("f", 1.0, "2024-01-03 22:00"), ("f", 2.0, "2024-01-04 22:00")])
    bars = make_bars(["2024-01-03 21:59:59", "2024-01-03 22:00:00", "2024-01-03 22:00:01"])
    out = attach_exogenous(bars, feats)["macro_f"].tolist()
    assert out[0] != out[0]      # NaN — before release
    assert out[1] == 1.0          # exactly at release -> included
    assert out[2] == 1.0


def test_heterogeneous_availability_worked_example() -> None:
    # VIX(D) released D 22:00; CREDIT(D-1) released D 22:00; both next-day values not yet public.
    feats = make_features([
        ("vix", 13.0, "2024-01-02 22:00"),     # VIX(D=Tue)
        ("vix", 14.0, "2024-01-03 22:00"),     # VIX(D+1) — not public at D+1 10:00
        ("credit", 1.40, "2024-01-02 22:00"),  # CREDIT(D-1=Mon), released next bday (Tue)
        ("credit", 1.50, "2024-01-03 22:00"),  # CREDIT(D) — not public at D+1 10:00
    ])
    bars = make_bars(["2024-01-03 10:00"])  # D+1 10:00
    out = attach_exogenous(bars, feats)
    assert out["macro_vix"].iloc[0] == 13.0
    assert out["macro_credit"].iloc[0] == 1.40


def test_ffill_carries_over_weekend_gap() -> None:
    feats = make_features([
        ("f", 5.0, "2024-01-05 22:00"),   # Friday evening
        ("f", 6.0, "2024-01-08 22:00"),   # Monday evening
    ])
    bars = make_bars([
        "2024-01-06 12:00",  # Saturday  -> Friday value
        "2024-01-07 12:00",  # Sunday    -> Friday value
        "2024-01-08 10:00",  # Mon morn  -> Friday value (Mon not yet released)
        "2024-01-08 23:00",  # Mon eve   -> Monday value
    ])
    out = attach_exogenous(bars, feats)["macro_f"].tolist()
    assert out == [5.0, 5.0, 5.0, 6.0]


def test_namespace_and_shape_preserved() -> None:
    feats = make_features([("f", 1.0, "2024-01-02 22:00")])
    bars = make_bars(["2024-01-01 10:00", "2024-01-03 10:00"])
    out = attach_exogenous(bars, feats)
    assert list(out.columns) == ["close", "macro_f"]
    assert out.index.equals(bars.index)
    assert len(out) == len(bars)
    pd.testing.assert_series_equal(out["close"], bars["close"])  # untouched


def test_empty_features_is_noop() -> None:
    bars = make_bars(["2024-01-02 10:00"])
    empty = pd.DataFrame(columns=["feature", "value", "available_from_utc"])
    out = attach_exogenous(bars, empty)
    assert list(out.columns) == ["close"]


@pytest.mark.parametrize("bad_index", ["naive", "non_utc", "unsorted", "not_datetime"])
def test_tz_and_sort_guards(bad_index: str) -> None:
    feats = make_features([("f", 1.0, "2024-01-02 22:00")])
    if bad_index == "naive":
        bars = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-03 10:00"]))
    elif bad_index == "non_utc":
        bars = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2024-01-03 10:00"]).tz_localize("US/Eastern"))
    elif bad_index == "unsorted":
        bars = make_bars(["2024-01-04 10:00", "2024-01-03 10:00"])
    else:
        bars = pd.DataFrame({"close": [1.0]}, index=[0])
    with pytest.raises(ValueError):
        attach_exogenous(bars, feats)


# --------------------------------------------------------------------------- #
# align — genericity across cadence
# --------------------------------------------------------------------------- #

def test_mixed_time_resolution_merges() -> None:
    # Parquet bar indices are commonly datetime64[us]; stamps are [ns]. Must still merge.
    feats = make_features([("f", 7.0, "2024-01-02 22:00")])
    bars = make_bars(["2024-01-03 10:00", "2024-01-04 10:00"])
    bars.index = bars.index.as_unit("us")
    assert str(bars.index.dtype) == "datetime64[us, UTC]"
    out = attach_exogenous(bars, feats)
    assert out["macro_f"].tolist() == [7.0, 7.0]


def test_max_staleness_nans_stale_bars() -> None:
    feats = make_features([("f", 1.0, "2024-01-02 22:00"), ("f", 2.0, "2024-01-03 22:00")])
    bars = make_bars(["2024-01-03 23:00", "2024-01-10 10:00"])  # fresh, then ~6.5 days stale
    out = attach_exogenous(bars, feats, max_staleness=pd.Timedelta(days=2))["macro_f"].tolist()
    assert out[0] == 2.0          # within 2 days of the 01-03 22:00 release
    assert out[1] != out[1]       # beyond 2 days -> NaN, not a stale carry-forward
    # default (no bound) still carries forward
    out_def = attach_exogenous(bars, feats)["macro_f"].tolist()
    assert out_def == [2.0, 2.0]


def test_weekly_cadence_attaches() -> None:
    feats = make_features([
        ("w", 100.0, "2024-01-03 22:00"),  # Wednesday weekly print
        ("w", 101.0, "2024-01-10 22:00"),  # next Wednesday
    ])
    bars = make_bars(["2024-01-05 10:00", "2024-01-09 10:00", "2024-01-11 10:00"])
    out = attach_exogenous(bars, feats)["macro_w"].tolist()
    assert out == [100.0, 100.0, 101.0]  # week-1 value holds until week-2 print is public


def test_mixed_cadence_daily_plus_weekly() -> None:
    feats = make_features([
        ("daily", 1.0, "2024-01-02 22:00"),
        ("daily", 2.0, "2024-01-03 22:00"),
        ("daily", 3.0, "2024-01-04 22:00"),
        ("weekly", 50.0, "2024-01-03 22:00"),  # updates once
    ])
    bars = make_bars(["2024-01-03 10:00", "2024-01-04 10:00", "2024-01-05 10:00"])
    out = attach_exogenous(bars, feats)
    assert out["macro_daily"].tolist() == [1.0, 2.0, 3.0]   # updates daily
    weekly = out["macro_weekly"].tolist()
    assert weekly[0] != weekly[0]                # NaN before the weekly print is public
    assert weekly[1:] == [50.0, 50.0]            # then ffills across days


def test_irregular_explicit_release_attaches() -> None:
    # Event-driven series with arbitrary release instants (e.g. rate decisions).
    feats = make_features([
        ("rate", 5.25, "2024-01-31 19:00"),  # FOMC-style afternoon release
        ("rate", 5.50, "2024-03-20 18:00"),
    ])
    bars = make_bars(["2024-01-31 18:00", "2024-02-15 10:00", "2024-03-20 19:00"])
    out = attach_exogenous(bars, feats)["macro_rate"].tolist()
    assert out[0] != out[0]        # before first decision -> NaN
    assert out[1] == 5.25
    assert out[2] == 5.50


# --------------------------------------------------------------------------- #
# availability policies
# --------------------------------------------------------------------------- #

def test_business_day_lag_zero_same_evening() -> None:
    obs = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-05"])})  # Tue, Fri
    got = BusinessDayLag(0).stamp(obs).tolist()
    assert got == [_utc("2024-01-02 22:00"), _utc("2024-01-05 22:00")]


def test_business_day_lag_one_rolls_weekend() -> None:
    obs = pd.DataFrame({"date": pd.to_datetime(["2024-01-02", "2024-01-05"])})  # Tue, Fri
    got = BusinessDayLag(1).stamp(obs).tolist()
    assert got == [_utc("2024-01-03 22:00"), _utc("2024-01-08 22:00")]  # Fri -> Mon


def test_calendar_day_lag_ignores_weekends() -> None:
    obs = pd.DataFrame({"date": pd.to_datetime(["2024-01-05"])})  # Friday
    got = CalendarDayLag(3, hour_utc=20).stamp(obs).tolist()
    assert got == [_utc("2024-01-08 20:00")]  # +3 calendar days


def test_explicit_release_passthrough() -> None:
    obs = pd.DataFrame({"release_utc": ["2024-01-31 19:00", "2024-03-20 18:00"]})
    got = ExplicitRelease().stamp(obs)
    assert got.tolist() == [_utc("2024-01-31 19:00"), _utc("2024-03-20 18:00")]
    assert str(got.dt.tz) == "UTC"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def test_zscore_matches_reference_and_warmup_is_nan() -> None:
    s = pd.Series(np.arange(1, 51, dtype=float))
    z = zscore(s, 20)
    assert z.iloc[:19].isna().all()           # warmup
    ref = (s.iloc[19] - s.iloc[:20].mean()) / s.iloc[:20].std()
    assert np.isclose(z.iloc[19], ref)


def test_default_recipes_produce_expected_columns() -> None:
    feats = compute_macro_features(synth_raw())
    assert set(feats["feature"].unique()) == {r.name for r in DEFAULT_RECIPES}
    # warmup dropped: a z20 feature starts later than a level feature
    n_level = (feats["feature"] == "vix_level").sum()
    n_z20 = (feats["feature"] == "vix_z20").sum()
    assert n_z20 == n_level - 19


def test_cross_series_availability_is_latest_of_inputs() -> None:
    # A feature mixing a lag-0 and a lag-1 series must inherit the LATER availability.
    raw = synth_raw()
    recipe = FeatureRecipe("mix", (MacroSeries.VIX, MacroSeries.CREDIT_SPREAD), lambda a, b: a + b)
    feats = compute_macro_features(raw, recipes=(recipe,))
    # For each date, availability should equal the credit (lag-1) stamp, not the VIX (lag-0) one.
    credit_avail = (raw[raw.series == "CREDIT_SPREAD"].set_index("date")["available_from_utc"])
    merged = feats.set_index("date")["available_from_utc"]
    common = merged.index.intersection(credit_avail.index)
    assert (merged.loc[common].values == credit_avail.loc[common].values).all()


def test_end_to_end_attach_on_synthetic() -> None:
    raw = synth_raw()
    feats = compute_macro_features(raw)
    # All z20 features need ~20 business days of warmup (first valid ≈ 2024-01-26), so sample
    # well past it.
    bars = make_bars(["2024-02-14 10:00", "2024-02-20 10:00", "2024-02-22 10:00"])
    out = attach_exogenous(bars, feats)
    macro_cols = [c for c in out.columns if c.startswith("macro_")]
    assert len(macro_cols) == len(DEFAULT_RECIPES)
    assert out.loc[:, macro_cols].notna().all().all()  # all features warmed up by mid-Feb


# --------------------------------------------------------------------------- #
# store: metastore, incremental merge, per-series reader
# --------------------------------------------------------------------------- #

def test_metastore_roundtrip_and_atomic(tmp_path: Path) -> None:
    ms = MacroMetastore(tmp_path)
    assert ms.read() == {}
    assert ms.last_obs("VIX") is None
    ms.update_series("VIX", {"fred_id": "VIXCLS", "last_obs": "2026-06-19", "n_obs": 4171})
    ms.update_series("US_2Y", {"fred_id": "DGS2", "last_obs": "2026-06-18"})
    ms.update_series("VIX", {"last_obs": "2026-06-20"})  # incremental field update
    meta = ms.read()
    assert set(meta) == {"VIX", "US_2Y"}
    assert meta["VIX"] == {"fred_id": "VIXCLS", "last_obs": "2026-06-20", "n_obs": 4171}
    assert ms.last_obs("US_2Y") == "2026-06-18"
    assert (tmp_path / "_metadata.json").exists()
    assert not (tmp_path / "_metadata.json.tmp").exists()  # temp cleaned up


def test_merge_keeps_latest_on_revision() -> None:
    existing = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
                             "series": "X", "value": [1.0, 2.0]})
    # revision of 2026-01-02 + a new bar 2026-01-03
    new = pd.DataFrame({"date": pd.to_datetime(["2026-01-02", "2026-01-03"]),
                        "series": "X", "value": [2.5, 3.0]})
    merged = _merge(existing, new)
    assert merged["date"].is_monotonic_increasing
    assert not merged["date"].duplicated().any()
    assert merged.set_index("date")["value"].loc["2026-01-02"] == 2.5  # newest wins
    assert list(merged["value"]) == [1.0, 2.5, 3.0]


def test_load_macro_reads_and_concats_per_series(tmp_path: Path) -> None:
    def _one(series: MacroSeries, vals):
        dates = pd.bdate_range("2026-01-01", periods=len(vals))
        df = pd.DataFrame({"date": dates, "series": series.value, "value": vals})
        df["available_from_utc"] = SERIES[series].availability.stamp(df)
        df.to_parquet(tmp_path / f"{series.value}.parquet", index=False)

    _one(MacroSeries.VIX, [15.0, 16.0, 17.0])
    _one(MacroSeries.US_2Y, [1.0, 1.1, 1.2])
    out = load_macro(tmp_path)
    assert set(out["series"].unique()) == {"VIX", "US_2Y"}
    assert str(out["available_from_utc"].dtype) == "datetime64[ns, UTC]"
    assert (out.sort_values(["series", "date"]).reset_index(drop=True)["value"].to_list()
            == out["value"].to_list())  # already sorted by (series, date)


def test_load_macro_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_macro(tmp_path)
