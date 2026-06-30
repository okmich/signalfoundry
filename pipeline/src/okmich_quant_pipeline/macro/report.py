"""Coverage / gap / staleness reporting for the macro feature store.

Pure (the only IO is ``write_report``). Turns the raw per-series frame and the engineered feature
frame into a structured health report so a materialize run can surface holes, stale feeds, and
thin features without a human eyeballing parquet.

Cadence-aware by inference, not by hard-coded series knowledge: each series' cadence is read off
its observed date spacing (``_infer_cadence``), so daily and weekly series get the right "expected
observations" and gap/staleness thresholds through the same path — and a future monthly/irregular
series needs no special case.
"""
from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from okmich_quant_pipeline.macro._io import atomic_write_text

REPORT_FILENAME = "_report.json"

# Per-cadence threshold (calendar days) above which a spacing between consecutive observations is a
# real hole, and last-obs age counts as stale. A business-daily series spans at most a long holiday
# weekend (Fri->Tue = 4 days) without a hole, so >5 flags a genuine gap; weekly (NFCI) tolerates one
# period plus its release lag (~13 days), so >14 flags a missed release. Both are overridable.
DAILY_STALE_DAYS = 5
WEEKLY_STALE_DAYS = 14


class Cadence(enum.StrEnum):
    """Observation cadence inferred from a series' date spacing."""

    BUSINESS_DAILY = "business_daily"
    WEEKLY = "weekly"
    IRREGULAR = "irregular"


@dataclass(frozen=True)
class SeriesCoverage:
    """Coverage / gap / staleness for one raw series."""

    series: str
    cadence: str
    first_obs: str
    last_obs: str
    n_obs: int
    expected_obs: int | None
    coverage_pct: float | None
    max_gap_days: int
    n_gaps: int
    staleness_days: int
    is_stale: bool


@dataclass(frozen=True)
class FeatureCoverage:
    """Density of one engineered feature over the materialized frame's date span."""

    feature: str
    n_obs: int
    first_obs: str
    last_obs: str
    density_pct: float


@dataclass(frozen=True)
class MacroReport:
    """Full materialize-run health report."""

    asof: str
    series: list[SeriesCoverage]
    features: list[FeatureCoverage]

    @property
    def has_stale(self) -> bool:
        return any(s.is_stale for s in self.series)

    def to_dict(self) -> dict:
        return {
            "asof": self.asof,
            "has_stale": self.has_stale,
            "series": [asdict(s) for s in self.series],
            "features": [asdict(f) for f in self.features],
        }

    def summary_lines(self) -> list[str]:
        lines = [f"macro report asof={self.asof}  has_stale={self.has_stale}"]
        for s in self.series:
            cover = f"{s.coverage_pct:.1f}%" if s.coverage_pct is not None else "n/a"
            flag = "  STALE" if s.is_stale else ""
            lines.append(
                f"  {s.series:<14} {s.cadence:<14} obs={s.n_obs:<5} cover={cover:<7} "
                f"max_gap={s.max_gap_days}d gaps={s.n_gaps} last={s.last_obs} stale={s.staleness_days}d{flag}"
            )
        lines.append(f"  {len(self.features)} features materialized")
        return lines


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #

def _infer_cadence(dates: pd.Series) -> Cadence:
    """Infer cadence from the median calendar-day spacing of sorted observation dates."""
    diffs = dates.sort_values().diff().dropna().dt.days
    if diffs.empty:
        return Cadence.IRREGULAR
    med = diffs.median()
    if med <= 1.5:  # business-daily: most spacings are 1, weekends pull the median only slightly
        return Cadence.BUSINESS_DAILY
    if 5 <= med <= 9:  # weekly
        return Cadence.WEEKLY
    return Cadence.IRREGULAR


def _threshold_days(cadence: Cadence, daily: int, weekly: int) -> int | None:
    """Gap/staleness threshold (calendar days) for a cadence; ``None`` ⇒ not flagged."""
    if cadence is Cadence.BUSINESS_DAILY:
        return daily
    if cadence is Cadence.WEEKLY:
        return weekly
    return None


def _expected_obs(cadence: Cadence, first: pd.Timestamp, last: pd.Timestamp) -> int | None:
    """Expected observation count over ``[first, last]`` for a cadence; ``None`` if undefined."""
    if cadence is Cadence.BUSINESS_DAILY:
        return len(pd.bdate_range(first, last))
    if cadence is Cadence.WEEKLY:
        return (last - first).days // 7 + 1
    return None


def _series_coverage(series: str, dates: pd.Series, asof: pd.Timestamp, daily: int, weekly: int) -> SeriesCoverage:
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    first, last = dates.iloc[0], dates.iloc[-1]
    cadence = _infer_cadence(dates)
    threshold = _threshold_days(cadence, daily, weekly)

    diffs = dates.diff().dropna().dt.days
    max_gap = int(diffs.max()) if not diffs.empty else 0
    n_gaps = int((diffs > threshold).sum()) if threshold is not None else 0

    expected = _expected_obs(cadence, first, last)
    coverage = round(100.0 * len(dates) / expected, 1) if expected else None

    staleness = int((asof.normalize() - last.normalize()).days)
    is_stale = threshold is not None and staleness > threshold

    return SeriesCoverage(
        series=series, cadence=cadence.value,
        first_obs=first.date().isoformat(), last_obs=last.date().isoformat(),
        n_obs=len(dates), expected_obs=expected, coverage_pct=coverage,
        max_gap_days=max_gap, n_gaps=n_gaps, staleness_days=staleness, is_stale=is_stale,
    )


def _feature_coverage(features: pd.DataFrame) -> list[FeatureCoverage]:
    """Per-feature density over the union of all feature observation dates.

    Daily features approach 100%; a weekly feature (e.g. NFCI) is legitimately ~20% — density is a
    cadence signal, not a defect, but a daily feature dropping well below its peers flags warmup or
    source gaps worth a look.
    """
    n_dates = features["date"].nunique()
    out: list[FeatureCoverage] = []
    for name, g in features.groupby("feature"):
        dates = pd.to_datetime(g["date"])
        density = round(100.0 * len(g) / n_dates, 1) if n_dates else 0.0
        out.append(FeatureCoverage(
            feature=str(name), n_obs=len(g),
            first_obs=dates.min().date().isoformat(), last_obs=dates.max().date().isoformat(),
            density_pct=density,
        ))
    return sorted(out, key=lambda f: f.feature)


def build_report(raw: pd.DataFrame, features: pd.DataFrame, *, asof: "pd.Timestamp | object",
                 daily_stale_days: int = DAILY_STALE_DAYS, weekly_stale_days: int = WEEKLY_STALE_DAYS) -> MacroReport:
    """Build the coverage/gap/staleness report from the raw and feature frames.

    ``asof`` is the reference "today" for staleness (a ``date``/``datetime``/``Timestamp``).
    """
    asof_ts = pd.Timestamp(asof)
    series = [
        _series_coverage(str(name), g["date"], asof_ts, daily_stale_days, weekly_stale_days)
        for name, g in raw.groupby("series")
    ]
    return MacroReport(
        asof=asof_ts.date().isoformat(),
        series=sorted(series, key=lambda s: s.series),
        features=_feature_coverage(features),
    )


def write_report(report: MacroReport, out_dir: Path | str) -> Path:
    """Write the report as ``{out_dir}/_report.json`` (atomic). Returns the path."""
    path = Path(out_dir) / REPORT_FILENAME
    atomic_write_text(json.dumps(report.to_dict(), indent=4), path)
    return path
