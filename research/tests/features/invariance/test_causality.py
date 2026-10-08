"""The catalogue is MEASURED look-ahead-free, not just flagged so.

Every bindable catalogue entry is computed on the doji/tie-rich fixture and on its first ``CUT`` rows; on the shared
rows the two must be identical. This is the guard that would have caught the 2026-10-07 finding — 7 entries flagged
``causal=True`` that read whole-series doji caps and volume bins — and it keeps the next one out.

The planted-leak controls prove the audit can see a leak at all: a guard that never fails is not a guard.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volume import tick_volume_how_zscore
from okmich_quant_features.volume._volume import VolumePhase
from okmich_quant_research.features.invariance import AuditStatus, check_function, compare_truncated, truncation_audit

from ._fixture import make_ohlcv

CUT = 1600
#: Defaults whose warm-up the 8-day fixture cannot cover would leave columns UNCHECKABLE (all NaN) rather than tested.
#: CTL's 15% omega never completes a leg on a 5-minute FX-scale path; 0.2% gives the labeller legs to label. The
#: weekly phase z-score needs weeks of history; its daily phase with a short warm-up exercises the same code path.
OVERRIDES = {"trend.continuous_trend_labeling": {"omega": 0.002},
             "trend.continuous_trend.ctl_trend_features": {"omega": 0.002},
             "trend.core_trend_features": {"continuous_omega": 0.002},
             "volume.tick_volume_phase_zscore": {"phase": VolumePhase.DAILY, "min_periods": 2}}
#: Entries whose warm-up outlasts the fixture: checked separately on a longer series below.
KNOWN_UNCHECKABLE = {"volume.tick_volume_how_zscore"}
#: Measured 2026-10-08: 488 causal columns. A floor well below that catches an audit that silently binds nothing.
MIN_CAUSAL_COLUMNS = 400


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_ohlcv()


@pytest.fixture(scope="module")
def audit(raw) -> pd.DataFrame:
    return truncation_audit(raw, cut=CUT, overrides=OVERRIDES)


# ── the guard ─────────────────────────────────────────────────────────────────────────────────────

def test_catalogue_is_lookahead_free(audit):
    leaks = audit[(audit.status == AuditStatus.LEAK) & (audit.causal_flag == True)]  # noqa: E712 — object column
    assert leaks.empty, ("catalogue entries flagged causal=True changed when later bars were appended:\n"
                         + leaks[["column", "changed_rows", "max_abs_diff"]].to_string())


def test_audit_actually_checks_the_catalogue(audit):
    assert (audit.status == AuditStatus.CAUSAL).sum() >= MIN_CAUSAL_COLUMNS


def test_no_entry_errors_on_the_fixture(audit):
    """BLOCKED must mean structurally unbindable (cross-market, event-indexed, ...), never a crash on doji/tie bars."""
    errors = audit[(audit.status == AuditStatus.BLOCKED) & audit.reason.str.startswith("error:")]
    assert errors.empty, errors[["entry", "reason"]].to_string()


def test_only_known_entries_are_uncheckable(audit):
    unchecked = set(audit.loc[audit.status == AuditStatus.UNCHECKABLE, "entry"])
    assert unchecked <= KNOWN_UNCHECKABLE, sorted(unchecked - KNOWN_UNCHECKABLE)


def test_weekly_phase_zscore_is_causal_given_its_warm_up():
    long_raw = make_ohlcv(n=8064)                                       # four weeks of 5-minute bars
    report = check_function(lambda d: tick_volume_how_zscore(d["tick_volume"], min_periods=2), long_raw, cut=6500,
                            name="tick_volume_how_zscore")
    assert (report.status == AuditStatus.CAUSAL).all(), report.to_string()


# ── the guard can see a leak ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,fn", [
    ("negative shift", lambda d: d["close"].shift(-1)),
    ("centred window", lambda d: d["close"].rolling(21, center=True).mean()),
    ("whole-series rank", lambda d: d["close"].rank(pct=True)),
    ("whole-series percentile cap", lambda d: d["tick_volume"].clip(upper=d["tick_volume"].quantile(0.99))),
])
def test_planted_leak_is_caught(raw, name, fn):
    report = check_function(fn, raw, cut=CUT, name=name)
    assert (report.status == AuditStatus.LEAK).all(), report.to_string()


def test_short_reach_leak_is_caught_where_it_crosses_the_cut(raw):
    """A backward fill reaches only to the next valid bar, so it shows only when the cut lands inside a gap — the
    documented limit of a single cut. Placed there, the audit sees it."""
    def fn(d):
        return d["close"].where(d["spread"] > 6).bfill()
    gaps = np.flatnonzero(~(raw["spread"] > 6).to_numpy()[CUT - 200:CUT]) + CUT - 200
    cut = int(gaps[-1]) + 1                                               # the last row of the truncated run is a gap
    assert (check_function(fn, raw, cut=cut).status == AuditStatus.LEAK).all()


@pytest.mark.parametrize("name,fn", [
    ("trailing window", lambda d: d["close"].rolling(21).mean()),
    ("expanding rank", lambda d: d["close"].expanding().rank(pct=True)),
    ("forward fill", lambda d: d["close"].where(d["spread"] > 6).ffill()),
])
def test_causal_control_passes(raw, name, fn):
    report = check_function(fn, raw, cut=CUT, name=name)
    assert (report.status == AuditStatus.CAUSAL).all(), report.to_string()


def test_nan_in_one_run_counts_as_a_change():
    full = pd.Series([1.0, 2.0, np.nan, 4.0])
    part = pd.Series([1.0, 2.0, 3.0, np.nan])
    shared, changed, max_abs = compare_truncated(full, part, cut=4)
    assert (shared, changed, max_abs) == (2, 2, 0.0)


def test_cut_outside_the_frame_is_rejected(raw):
    with pytest.raises(ValueError):
        truncation_audit(raw, cut=len(raw), entries=[])
