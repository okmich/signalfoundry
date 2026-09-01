"""FRED series fetcher via the public fredgraph CSV endpoint (no API key).

The endpoint ``fred.stlouisfed.org/graph/fredgraph.csv?id={id}`` returns the
latest-vintage series as ``observation_date,{ID}`` with blank cells for
non-trading days. VIX/VIX3M are market closes (never revised); BAA10Y / DTWEXBGS
carry only minor revisions — for Phase 1 the latest vintage plus a conservative
publish stamp is sufficient. (Phase 2's revised series — yields, NFCI — will need a
FRED API key for true ALFRED point-in-time vintages.)

The fetcher is cadence-agnostic: FRED returns whatever cadence a series has (daily,
weekly, monthly) and the availability stamp comes from the series' policy
(``SeriesSpec.availability``), so weekly/monthly series drop in with no change here.
"""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from okmich_quant_pipeline.macro._types import SERIES, MacroSeries, SeriesSpec
from okmich_quant_pipeline.http import get

_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_id}&cosd={start}&coed={end}"


def fetch(series: MacroSeries, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download one macro ``series`` between ``start`` and ``end`` (inclusive).

    Returns the long-format schema documented in ``fetchers/__init__.py``.
    """
    spec = SERIES[series]
    url = _CSV_URL.format(fred_id=spec.fred_id, start=start.isoformat(), end=end.isoformat())
    resp = get(url)
    return _parse(resp.text, series, spec)


def _parse(csv_text: str, series: MacroSeries, spec: SeriesSpec) -> pd.DataFrame:
    """Parse fredgraph CSV text into the long-format macro schema."""
    raw = pd.read_csv(io.StringIO(csv_text), na_values=["."])
    # fredgraph returns exactly two columns: observation date, then the series id. A 200 with an
    # unexpected body (maintenance page, empty CSV) would otherwise raise an opaque IndexError below.
    if raw.shape[1] < 2:
        raise ValueError(f"unexpected fredgraph response for {spec.fred_id}: columns={list(raw.columns)}")
    date_col, value_col = raw.columns[0], raw.columns[1]
    out = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]),
        "series": series.value,
        "value": pd.to_numeric(raw[value_col], errors="coerce"),
    })
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["available_from_utc"] = spec.availability.stamp(out)
    return out
