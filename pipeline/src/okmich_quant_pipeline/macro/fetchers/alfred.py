"""Keyed FRED/ALFRED fetcher (point-in-time first-print vintages + full-history series).

The anonymous ``fredgraph.csv`` path (``fetchers/fred.py``) returns *latest-vintage* data and caps
some series (ICE HY-OAS) to a rolling 3-year window. This fetcher uses the keyed JSON API:

- ``output_type=4`` (**initial release only**) → each observation's **first-print** value, with the
  real release date in ``realtime_start``. This is the no-lookahead value markets reacted to; we use
  it for revision-sensitive series (NFCI). Availability comes from ``realtime_start`` (the true
  release date), not the heuristic publish-lag policy.
- ``output_type=1`` (latest) → full history for never-revised series that the anonymous CSV caps
  (ICE HY-OAS). Availability falls back to the series' lag policy, same as the CSV path.

The API key is a secret (query-param auth — FRED has no header auth), so it is scrubbed from any
exception before it can propagate into a log.
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from okmich_quant_pipeline.http import get
from okmich_quant_pipeline.macro._types import SERIES, PUBLISH_HOUR_UTC, MacroSeries, SeriesSpec

_API_URL = "https://api.stlouisfed.org/fred/series/observations"
_INITIAL_RELEASE = 4  # FRED output_type for first-print-only observations
# Full realtime span (FRED's min..max). Required for output_type=4: the default realtime is
# today-only ("No vintage dates exist for the specified real-time period"), so we must ask for the
# whole vintage history and let output_type=4 pick each observation's initial release.
_REALTIME_MIN = "1776-07-04"
_REALTIME_MAX = "9999-12-31"


def fetch(series: MacroSeries, start: dt.date, end: dt.date, *, api_key: str, output_type: int) -> pd.DataFrame:
    """Download one ``series`` between ``start`` and ``end`` (inclusive) via the keyed JSON API.

    ``output_type=4`` returns first-print values (vintage); anything else returns latest. Returns
    the long-format schema documented in ``fetchers/__init__.py``.
    """
    spec = SERIES[series]
    url = (f"{_API_URL}?series_id={spec.fred_id}&api_key={api_key}&file_type=json"
           f"&observation_start={start.isoformat()}&observation_end={end.isoformat()}"
           f"&output_type={output_type}")
    if output_type == _INITIAL_RELEASE:
        url += f"&realtime_start={_REALTIME_MIN}&realtime_end={_REALTIME_MAX}"
    try:
        resp = get(url)
    except Exception as exc:  # never let the key-bearing URL reach a caller's log
        raise RuntimeError(f"ALFRED fetch failed for {spec.fred_id}: {str(exc).replace(api_key, '***')}") from None
    return _parse(resp.text, series, spec, vintage=(output_type == _INITIAL_RELEASE))


def _parse(json_text: str, series: MacroSeries, spec: SeriesSpec, *, vintage: bool) -> pd.DataFrame:
    """Parse a FRED ``series/observations`` JSON body into the long-format macro schema."""
    payload = json.loads(json_text)
    obs = payload.get("observations")
    if not obs:
        raise ValueError(f"ALFRED returned no observations for {spec.fred_id}")

    raw = pd.DataFrame(obs)
    out = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "series": series.value,
        "value": pd.to_numeric(raw["value"], errors="coerce"),
        "_rt_start": pd.to_datetime(raw["realtime_start"]),
    }).dropna(subset=["value"]).reset_index(drop=True)

    if vintage:
        # First-print availability = the date FRED published the initial release, stamped at a
        # conservative end-of-day UTC hour (realtime_start is a date, not a wall-clock time). This
        # is the real release date, strictly better than the publish-lag heuristic.
        out["available_from_utc"] = (out["_rt_start"].dt.normalize() + pd.Timedelta(hours=PUBLISH_HOUR_UTC)).dt.tz_localize("UTC")
    else:
        # Latest-vintage (never-revised) series: realtime_start is meaningless per-obs, so fall back
        # to the series' lag policy, exactly like the CSV path.
        out["available_from_utc"] = spec.availability.stamp(out)
    return out.drop(columns="_rt_start")
