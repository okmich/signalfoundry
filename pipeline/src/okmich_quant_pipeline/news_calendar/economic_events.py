r"""Economic-event surprises (ForexFactory-native) — the event channel's surprise feature.

A self-contained data layer, distinct from the schedule calendar (``build.py``): ForexFactory's
event blob carries ``forecast`` / ``actual`` / ``previous`` for the high-impact US releases (and
ECB/BoE rate decisions), all in matching headline units, with the real release timestamp in
``dateline``. We store those raw values, then derive

    surprise = (actual − forecast) / σ_trailing_per_event_type

standardized causally per event type (only prior releases of that type feed σ — no lookahead). The
result is a per-release series stamped at the release instant, so it rides the existing
``align.attach_exogenous`` backward asof-merge: each bar carries the most-recent release's surprise,
ffilled until the next release (and decaying alongside the event-timing ``minutes_since_last``).

FF's ``actual`` *is* the released headline number, so the surprise is correct by construction — no
ALFRED first-print reconstruction (which is provably off for change/MoM% headlines; see the sprint
notes). ``fetch-economic-events`` (this module's ``main``) refreshes the store.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from okmich_quant_pipeline._io import atomic_write_parquet
from okmich_quant_pipeline.news_calendar._types import EventName
from okmich_quant_pipeline.news_calendar.fetchers import forexfactory as ff

logger = logging.getLogger(__name__)

DEFAULT_EVENTS_PATH = Path(r"E:\data_dump\calendars\economic_events.parquet")
DEFAULT_YEAR_MIN = 2011  # FF carries forecast/actual back to ~2011 (verified); earlier is sparse.
DEFAULT_SIGMA_WINDOW = 24
DEFAULT_SIGMA_MIN_PERIODS = 12

# FF numeric suffixes → multiplier. Units are consistent within an event type (forecast and actual
# share them), so the absolute scale is irrelevant to the z-scored surprise — but scale anyway so a
# mixed "1.2M"/"1200K" pair would still reconcile.
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

# (FF country, FF event name) → canonical EventName. US primaries only (the headline markets trade);
# Core/YoY/Prelim variants are a deliberate v1 omission. ECB/BoE rate decisions carry a forecast too.
_DATA_NAME_MAP: dict[tuple[str, str], EventName] = {
    ("US", "Non-Farm Employment Change"): EventName.US_NFP,
    ("US", "CPI m/m"): EventName.US_CPI,
    ("US", "PPI m/m"): EventName.US_PPI,
    ("US", "Advance GDP q/q"): EventName.US_GDP,
    ("US", "Core PCE Price Index m/m"): EventName.US_PCE,
    ("US", "Retail Sales m/m"): EventName.US_RETAIL_SALES,
    ("EZ", "Main Refinancing Rate"): EventName.EU_ECB_RATE_DECISION,
    ("UK", "Official Bank Rate"): EventName.UK_BOE_RATE_DECISION,
}


def _parse_value(raw: object) -> float:
    """Parse an FF value string (``'164K'``, ``'0.4%'``, ``'-0.1%'``, ``''``) to float; NaN if blank."""
    if raw is None:
        return float("nan")
    s = str(raw).strip().replace(",", "").replace("%", "")
    if not s or s == "-":
        return float("nan")
    mult = _SUFFIX.get(s[-1].upper(), 1.0)
    if mult != 1.0:
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return float("nan")


def fetch_events_with_data(year_min: int, year_max: int) -> pd.DataFrame:
    """Fetch mapped events with forecast/actual/previous from ForexFactory across the year span.

    Returns ``[timestamp_utc, event_name, actual, forecast, previous]`` sorted by time, deduped on
    ``(timestamp_utc, event_name)``.
    """
    rows: list[dict] = []
    for start, end in ff._quarter_windows(year_min, year_max):
        days = ff._extract_days(ff.get(ff._range_url(start, end)).text)
        for day in days:
            for ev in day.get("events", []):
                key = (ev.get("country"), ev.get("name"))
                ts = ev.get("dateline")
                if key not in _DATA_NAME_MAP or ts is None:
                    continue
                rows.append({
                    "timestamp_utc": pd.Timestamp(dt.datetime.fromtimestamp(ts, dt.timezone.utc)),
                    "event_name": _DATA_NAME_MAP[key].value,
                    "actual": _parse_value(ev.get("actual")),
                    "forecast": _parse_value(ev.get("forecast")),
                    "previous": _parse_value(ev.get("previous")),
                })
    if not rows:
        raise RuntimeError(f"economic_events: parsed zero mapped events in [{year_min}, {year_max}]")
    return (pd.DataFrame(rows).drop_duplicates(subset=["timestamp_utc", "event_name"])
            .sort_values("timestamp_utc").reset_index(drop=True))


def load_economic_events(path: Path | str = DEFAULT_EVENTS_PATH) -> pd.DataFrame:
    """Load the stored economic-events parquet, with ``timestamp_utc`` as tz-aware UTC."""
    df = pd.read_parquet(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def compute_surprise(events: pd.DataFrame, *, window: int = DEFAULT_SIGMA_WINDOW,
                     min_periods: int = DEFAULT_SIGMA_MIN_PERIODS) -> pd.DataFrame:
    """Derive the standardized surprise feature frame for ``align.attach_exogenous``.

    ``surprise = (actual − forecast) / σ`` where σ is the trailing std of the miss over the prior
    ``window`` releases *of the same event type* (``shift(1)`` → strictly causal, warmup → dropped).
    Returns long ``[feature, value, available_from_utc]`` with a single ``surprise`` feature stamped
    at each release instant (per-type columns are a one-line change: ``feature = event_name``).
    """
    df = events.dropna(subset=["actual", "forecast"]).sort_values("timestamp_utc").reset_index(drop=True)
    df["miss"] = df["actual"] - df["forecast"]
    sigma = df.groupby("event_name")["miss"].transform(lambda m: m.shift(1).rolling(window, min_periods=min_periods).std())
    # σ==0 (a run of identical misses, e.g. a rate held flat) would make miss/σ = ±inf — drop those
    # (surprise is undefined when there is no recent variation to standardize against).
    df["surprise"] = (df["miss"] / sigma).replace([np.inf, -np.inf], np.nan)
    out = df.dropna(subset=["surprise"]).reset_index(drop=True)
    return pd.DataFrame({
        "feature": "surprise",
        "value": out["surprise"],
        "available_from_utc": pd.to_datetime(out["timestamp_utc"], utc=True),  # tz-aware: the asof key
    })


def build_economic_events(year_min: int, year_max: int, out_path: Path) -> pd.DataFrame:
    """Fetch and atomically persist the economic-events store. Returns the frame."""
    events = fetch_events_with_data(year_min, year_max)
    atomic_write_parquet(events, Path(out_path))
    logger.info(f"Wrote {len(events)} economic events -> {out_path}")
    logger.info("By event_name:\n" + events["event_name"].value_counts().to_string())
    return events


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    parser = argparse.ArgumentParser(description="Fetch the economic-events surprise store (ForexFactory forecast/actual/previous).")
    parser.add_argument("--year-min", type=int, default=DEFAULT_YEAR_MIN)
    parser.add_argument("--year-max", type=int, default=dt.date.today().year)
    parser.add_argument("--out", default=str(DEFAULT_EVENTS_PATH), help=f"Output parquet (default: {DEFAULT_EVENTS_PATH})")
    args = parser.parse_args()
    build_economic_events(args.year_min, args.year_max, Path(args.out))


if __name__ == "__main__":
    main()
