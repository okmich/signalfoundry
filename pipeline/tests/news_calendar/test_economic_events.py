"""Economic-surprise tests (FF-native): value parsing, fetch mapping, causal surprise, attach.

All offline: the FF fetch is exercised on a synthetic blob; surprise math is checked on hand-built
event frames where every value is verifiable by hand.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.macro.attach import attach_surprise_to_dataset
from okmich_quant_pipeline.news_calendar import economic_events as ee
from okmich_quant_pipeline.news_calendar.fetchers import forexfactory as ff


# --------------------------------------------------------------------------- #
# value parsing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, expected", [
    ("164K", 164000.0), ("0.4%", 0.4), ("-0.1%", -0.1), ("1.2M", 1_200_000.0),
    ("2.5", 2.5), ("1,234K", 1_234_000.0),
])
def test_parse_value(raw: str, expected: float) -> None:
    assert ee._parse_value(raw) == expected


@pytest.mark.parametrize("raw", ["", "  ", "-", None])
def test_parse_value_blank_is_nan(raw) -> None:
    assert np.isnan(ee._parse_value(raw))


# --------------------------------------------------------------------------- #
# fetch mapping (synthetic FF blob)
# --------------------------------------------------------------------------- #

# Real release instants: NFP 2024-01-05 and CPI 2024-01-11, both 08:30 ET (13:30 UTC).
_FF_EVENTS = [
    {"country": "US", "name": "Non-Farm Employment Change", "dateline": 1704461400,
     "forecast": "160K", "actual": "200K", "previous": "150K"},
    {"country": "US", "name": "CPI m/m", "dateline": 1704979800,
     "forecast": "0.3%", "actual": "0.5%", "previous": "0.2%"},
    {"country": "US", "name": "Unemployment Rate", "dateline": 1704461400,
     "forecast": "3.8%", "actual": "3.7%", "previous": "3.9%"},
]


def _serve(monkeypatch, year: int, events: list[dict]) -> None:
    """Serve a full month of ``days`` per request (one entry per calendar day, as the live page has), ``events`` on
    the 9th of every month."""
    def page(url: str) -> str:
        start, end = next(w for w in ff._month_windows(year, year) if ff._range_url(*w) == url)
        days = [{"events": events if d.day == 9 else []} for d in pd.date_range(start, end)]
        return f"<script>window.calendarComponentStates[1] = {{\n  days: {json.dumps(days)},\n  more: 1\n}};</script>"
    monkeypatch.setattr(ee.ff, "get", lambda url, **_kw: type("R", (), {"text": page(url)})())
    monkeypatch.setattr(ee.ff, "_PAUSE_SECONDS", 0.0)


def test_fetch_events_with_data_maps_and_parses(monkeypatch) -> None:
    _serve(monkeypatch, 2024, _FF_EVENTS)
    df = ee.fetch_events_with_data(2024, 2024)  # 12 months x the same events -> deduped
    assert set(df["event_name"]) == {"US_NFP", "US_CPI"}  # unmapped "Unemployment Rate" dropped
    nfp = df[df["event_name"] == "US_NFP"].iloc[0]
    assert nfp["actual"] == 200000.0 and nfp["forecast"] == 160000.0
    assert len(df) == 2  # deduped across the repeated months
    assert list(df.columns) == ee.EVENT_COLUMNS


def test_fetch_events_with_data_stamps_release_times_and_keeps_both_of_a_double_release(monkeypatch) -> None:
    # 2025-12-16: October and November payrolls released together at 08:30 ET; ForexFactory lists October at 08:29.
    double = [
        {"country": "US", "name": "Non-Farm Employment Change", "dateline": 1765891800,
         "forecast": "64K", "actual": "64K", "previous": "-105K"},
        {"country": "US", "name": "Non-Farm Employment Change", "dateline": 1765891740,
         "forecast": "", "actual": "-105K", "previous": "119K"},
    ]
    _serve(monkeypatch, 2025, double)
    df = ee.fetch_events_with_data(2025, 2025)
    assert df["timestamp_utc"].tolist() == [pd.Timestamp("2025-12-16 13:30", tz="UTC")] * 2
    assert df["actual"].tolist() == [-105000.0, 64000.0]  # the older month first, the newer one last


def test_fetch_events_with_data_raises_when_forexfactory_disagrees(monkeypatch) -> None:
    # PCE recorded at 10:00 ET on 2024-06-28, which release_times stamps at 08:30.
    pce = [{"country": "US", "name": "Core PCE Price Index m/m", "dateline": 1719583200,
            "forecast": "0.1%", "actual": "0.1%", "previous": "0.3%"}]
    _serve(monkeypatch, 2024, pce)
    with pytest.raises(ValueError, match="US_PCE 2024-06-28: stamped 08:30"):
        ee.fetch_events_with_data(2024, 2024)


# --------------------------------------------------------------------------- #
# surprise computation — causality, per-type sigma, math
# --------------------------------------------------------------------------- #

def _events(event: str, misses: list[float], *, start: str = "2024-01-01") -> pd.DataFrame:
    ts = pd.date_range(start, periods=len(misses), freq="30D", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ts, "event_name": event,
                         "forecast": 0.0, "actual": misses, "previous": 0.0})


def test_surprise_is_causal_no_lookahead() -> None:
    ev = _events("US_CPI", [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.4, -0.3])
    s1 = ee.compute_surprise(ev, window=3, min_periods=2).set_index("available_from_utc")["value"]
    # Mutating the LAST release must not change any earlier surprise (sigma uses only prior misses).
    ev2 = ev.copy()
    ev2.loc[ev2.index[-1], "actual"] = 99.0
    s2 = ee.compute_surprise(ev2, window=3, min_periods=2).set_index("available_from_utc")["value"]
    common = s1.index.intersection(s2.index)[:-1]  # all but the mutated last instant
    assert (s1.loc[common].to_numpy() == s2.loc[common].to_numpy()).all()


def test_surprise_math_and_warmup() -> None:
    ev = _events("US_CPI", [1.0, 3.0, 5.0])  # misses = actual (forecast 0); window>=2 needs 2 priors
    out = ee.compute_surprise(ev, window=3, min_periods=2)
    # First two releases are warmup (sigma needs >=2 prior) -> dropped; only the 3rd survives.
    assert len(out) == 1
    sigma = float(np.std([1.0, 3.0], ddof=1))  # std of the two PRIOR misses
    assert out["value"].iloc[0] == pytest.approx(5.0 / sigma)


def test_surprise_sigma_is_per_event_type() -> None:
    # Each type's sigma must depend only on its own prior misses — perturbing one type's values must
    # not move the other's surprises.
    cpi = _events("US_CPI", [0.1, 0.3, 0.2, 0.5], start="2024-01-01")
    nfp = _events("US_NFP", [10.0, 90.0, 50.0, 50.0], start="2024-01-08")
    base = ee.compute_surprise(pd.concat([cpi, nfp], ignore_index=True), window=3, min_periods=2)
    cpi_only = base[base["available_from_utc"].isin(cpi["timestamp_utc"])].set_index("available_from_utc")["value"]

    nfp2 = _events("US_NFP", [999.0, -999.0, 500.0, -500.0], start="2024-01-08")  # wildly different
    perturbed = ee.compute_surprise(pd.concat([cpi, nfp2], ignore_index=True), window=3, min_periods=2)
    cpi_after = perturbed[perturbed["available_from_utc"].isin(cpi["timestamp_utc"])].set_index("available_from_utc")["value"]
    assert (cpi_only.to_numpy() == cpi_after.to_numpy()).all()  # CPI surprises unchanged by NFP


def test_surprise_zero_sigma_is_dropped_not_inf() -> None:
    # A flat run (identical misses → σ=0) then a real miss would be miss/0 = inf; it must be dropped.
    ev = _events("UK_BOE_RATE_DECISION", [0.0, 0.0, 0.0, 0.0, 0.5])
    out = ee.compute_surprise(ev, window=3, min_periods=2)
    assert np.isfinite(out["value"].to_numpy()).all()  # no inf leaked through


# --------------------------------------------------------------------------- #
# attach end-to-end
# --------------------------------------------------------------------------- #

def test_attach_surprise_to_dataset(tmp_path) -> None:
    ev = _events("US_CPI", [0.1, -0.2, 0.3, -0.1, 0.2, 0.0, 0.4, -0.3, 0.1, -0.2])
    path = tmp_path / "events.parquet"
    atomic_write_parquet(ev, path)

    idx = pd.date_range("2024-01-01", "2024-10-01", freq="1D", tz="UTC")
    ds = pd.DataFrame({"feat_x": range(len(idx))}, index=idx)
    out = attach_surprise_to_dataset(ds, path, window=3, min_periods=2)

    assert "macro_event_surprise" in out.columns
    assert len(out) == len(ds)
    assert out["macro_event_surprise"].notna().all()  # fill=0 -> dense
    # The final bar carries the most-recent computed surprise (ffilled).
    last = ee.compute_surprise(ev, window=3, min_periods=2)["value"].iloc[-1]
    assert out["macro_event_surprise"].iloc[-1] == pytest.approx(last)


def test_attach_surprise_fill_none_leaves_prehistory_nan(tmp_path) -> None:
    ev = _events("US_CPI", [0.1, -0.2, 0.3, -0.1, 0.2], start="2024-06-01")
    path = tmp_path / "events.parquet"
    atomic_write_parquet(ev, path)
    idx = pd.date_range("2024-01-01", "2024-12-01", freq="1D", tz="UTC")  # starts before any release
    ds = pd.DataFrame({"feat_x": range(len(idx))}, index=idx)
    out = attach_surprise_to_dataset(ds, path, window=3, min_periods=2, fill=None)
    assert out["macro_event_surprise"].iloc[0] != out["macro_event_surprise"].iloc[0]  # NaN pre-history
