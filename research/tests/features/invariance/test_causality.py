"""The catalogue is MEASURED look-ahead-free, not just flagged so.

Every bindable catalogue entry is computed on the doji/tie-rich fixture and on its first ``CUT`` rows; on the shared
rows the two must be identical. This is the guard that would have caught the 2026-10-07 finding — 7 entries flagged
``causal=True`` that read whole-series doji caps and volume bins — and it keeps the next one out.

The start cut is its mirror: the catalogue is also computed without its first ``START`` rows, and after ``SETTLE``
bars of history the two must agree — the guard for the 2026-10-08 finding (the MFI close passthrough and cumsum
flows, levels anchored at the frame's first bar). Anchored columns found then and left for the analyst are listed.

The planted-leak controls prove the audit can see a leak at all: a guard that never fails is not a guard.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_features.volume import tick_volume_how_zscore
from okmich_quant_features.volume._volume import VolumePhase
from okmich_quant_research.features.invariance import (AuditStatus, check_function, check_function_start,
                                                       compare_truncated, start_cut_audit, truncation_audit)

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


# ── start cut: the value must not depend on where the history begins ──────────────────────────────

START, SETTLE = 300, 1200
#: Measured 2026-10-08 after the MFI flows were fixed; each is a definition decision for the analyst, not a slip:
KNOWN_ANCHORED = {
    "microstructure.order_flow.vwap_anchored": "bound without an anchor, so a cumulative VWAP from the first bar",
    "volume.vwap_adjusted_roc": "ROC of a cumulative VWAP from the first bar",
    "volume.vwap_dev_momentum": "deviation from a cumulative VWAP from the first bar",
    "volume.core_volume_features@vwap_adjusted_roc": "as volume.vwap_adjusted_roc",
    "volume.core_volume_features@vwap_dev_momentum": "as volume.vwap_dev_momentum",
    "volume.ad@0": "the accumulation/distribution line, a cumsum",
    "volume.tick_volume_phase_zscore": "expanding per-phase baseline, by design",
    "directional_change.idc_parse@t_dc0": "a bar position in the frame",
    "momentum.dv2": "rolling(2).mean() rounding flips ties in its strict percent rank on ~1% of bars",
}
#: identity defaults (p1 = p2 = p3 = 1): as a catalogue entry it returns the close.
KNOWN_PASSTHROUGH = {"momentum._williamblau.triple_ema"}
#: event-sparse or long-warm-up columns with fewer than MIN_SHARED compared rows on the fixture.
KNOWN_START_UNCHECKABLE = KNOWN_UNCHECKABLE | {"directional_change.idc_parse"}


@pytest.fixture(scope="module")
def start_audit(raw) -> pd.DataFrame:
    return start_cut_audit(raw, start=START, settle=SETTLE, overrides=OVERRIDES)


def test_catalogue_has_no_new_anchored_column(start_audit):
    anchored = set(start_audit.loc[start_audit.status == AuditStatus.ANCHORED, "column"])
    assert anchored <= set(KNOWN_ANCHORED), sorted(anchored - set(KNOWN_ANCHORED))


def test_mfi_flows_are_no_longer_anchored(start_audit):
    mfi = start_audit[start_audit.entry.isin(["volume.mfi_features", "volume.mfi_volume_features"])]
    assert len(mfi) > 50 and (mfi.status == AuditStatus.START_FREE).all(), mfi[mfi.status != AuditStatus.START_FREE]


def test_catalogue_has_no_new_passthrough(start_audit):
    through = set(start_audit.loc[start_audit.passthrough.fillna("") != "", "entry"])
    assert through <= KNOWN_PASSTHROUGH, sorted(through - KNOWN_PASSTHROUGH)


def test_start_audit_actually_checks_the_catalogue(start_audit):
    assert (start_audit.status == AuditStatus.START_FREE).sum() >= MIN_CAUSAL_COLUMNS
    unchecked = set(start_audit.loc[start_audit.status == AuditStatus.UNCHECKABLE, "entry"])
    assert unchecked <= KNOWN_START_UNCHECKABLE, sorted(unchecked - KNOWN_START_UNCHECKABLE)


@pytest.mark.parametrize("name,fn", [
    ("cumsum level", lambda d: np.log(d["close"]).diff().cumsum()),
    ("bar position", lambda d: pd.Series(np.arange(len(d), dtype=float), index=d.index)),
    ("expanding mean", lambda d: d["tick_volume"].expanding().mean()),
])
def test_planted_anchor_is_caught(raw, name, fn):
    report = check_function_start(fn, raw, start=START, settle=SETTLE, name=name)
    assert (report.status == AuditStatus.ANCHORED).all(), report.to_string()


@pytest.mark.parametrize("name,fn", [
    ("trailing window", lambda d: d["close"].rolling(21).mean()),
    ("trailing sum", lambda d: d["tick_volume"].rolling(60).sum()),
    ("EMA forgets its seed", lambda d: d["close"].ewm(span=50, adjust=False).mean()),
])
def test_start_free_control_passes(raw, name, fn):
    report = check_function_start(fn, raw, start=START, settle=SETTLE, name=name)
    assert (report.status == AuditStatus.START_FREE).all(), report.to_string()


def test_passthrough_is_named(raw):
    report = check_function_start(lambda d: d["close"] * 1.0, raw, start=START, settle=SETTLE, name="close")
    assert report.passthrough.tolist() == ["close"]
    report = check_function_start(lambda d: d["close"] * 2.0, raw, start=START, settle=SETTLE, name="scaled")
    assert report.passthrough.tolist() == [""]          # (close.shift(1) would be: the fixture's open IS it)


def test_start_cut_without_room_is_rejected(raw):
    with pytest.raises(ValueError):
        start_cut_audit(raw, start=START, settle=len(raw), entries=[])
