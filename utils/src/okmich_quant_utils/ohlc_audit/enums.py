"""Fixed vocabularies for the OHLC audit (StrEnum, per the repo's code style)."""
from __future__ import annotations

import enum


class AssetClass(enum.StrEnum):
    FX = "fx"
    METAL = "metal"
    US_INDEX = "us_index"
    EU_INDEX = "eu_index"
    ASIA_INDEX = "asia_index"
    CRYPTO = "crypto"


class SlotState(enum.StrEnum):
    """What the calendar expects in one bar slot."""

    OPEN = "open"                  # a bar is expected
    THIN = "thin"                  # open, but bars are often missing (no ticks, e.g. FX rollover): absence is minor
    CLOSED = "closed"              # outside the weekly schedule (weekend, overnight, lunch): a bar here is out of session
    HOLIDAY = "holiday"            # trading day closed by a holiday rule: absence expected, a bar is tolerated
    EARLY_CLOSE = "early_close"    # after a holiday early close: absence expected, a bar is tolerated
    MAINTENANCE = "maintenance"    # declared broker maintenance: absence expected, a bar is tolerated


class HolidayEffect(enum.StrEnum):
    CLOSED = "closed"
    EARLY_CLOSE = "early_close"


class OverrideStatus(enum.StrEnum):
    """A per-symbol clock deviation. Only CONFIRMED overrides restamp bars; PROPOSED ones are dropped and refetched."""

    CONFIRMED = "confirmed"
    PROPOSED = "proposed"


class WindowKind(enum.StrEnum):
    """Why a registry window is bad."""

    SPLICE = "splice"                      # export splice: shifted duplicates and interleave
    INTERLEAVE = "interleave"              # two feeds alternating bar by bar
    MIXED_CLOCK = "mixed_clock"            # bars from two clocks in one window, no single shift fixes it
    OUT_OF_SESSION = "out_of_session"      # bars where the market is closed (e.g. re-stamped Friday data on Saturday)
    BAD_PRICES = "bad_prices"              # prices disagree with an independent reference (e.g. crypto vs Binance)
    REPLAY = "replay"                      # bars are exact copies of bars from another time, re-stamped
    UNVERIFIED_CLOCK = "unverified_clock"  # clock cannot be confirmed
    OUTAGE = "outage"                      # bars missing inside a session (< 1 day)
    HOLE = "hole"                          # bars missing for >= 1 day
    SUSPECT = "suspect"                    # looks wrong, needs a human look


class WindowAction(enum.StrEnum):
    DROP = "drop"          # remove the bars, then refetch the window
    REFETCH = "refetch"    # nothing to remove (missing bars): refetch the window
    INSPECT = "inspect"    # report only


class Timeframe(enum.IntEnum):
    """Bar size in minutes (the broker folder name under ``raw/<broker>/``)."""

    M5 = 5


class Severity(enum.StrEnum):
    CRITICAL = "critical"  # data present but wrong (clock, replay, out of session, broken OHLC)
    MAJOR = "major"        # data missing for >= 30 min inside a session, or a broker-wide anomaly
    MINOR = "minor"        # short gap, thin slot, spike suspect, spread, stale run
    INFO = "info"          # recognised structural closure or market event


SEVERITY_RANK = {Severity.CRITICAL: 3, Severity.MAJOR: 2, Severity.MINOR: 1, Severity.INFO: 0}


class FindingKind(enum.StrEnum):
    MISSING_STAMP = "missing_stamp"                # D1: the row has no timestamp (NaT): quarantined, dropped
    DUPLICATE_STAMP = "duplicate_stamp"            # D1
    NON_MONOTONIC = "non_monotonic"                # D1
    OFF_GRID = "off_grid"                          # D1
    FUTURE_STAMP = "future_stamp"                  # D1
    CLOCK_OVERLAP = "clock_overlap"                # D1: two raw stamps map to one UTC slot (seam, splice)
    DST_FOLD = "dst_fold"                          # D1: bar in a raw hour the server clock passed twice (may be merged)
    OHLC_VIOLATION = "ohlc_violation"              # D2
    NON_FINITE = "non_finite"                      # D2
    NON_POSITIVE = "non_positive"                  # D2
    OUTAGE = "outage"                              # D3: OPEN slots missing, < 1 day
    HOLE = "hole"                                  # D3: OPEN slots missing, >= 1 day
    THIN_NO_TICK = "thin_no_tick"                  # D3: THIN slots missing for >= 30 min
    STRUCTURAL_CLOSURE = "structural_closure"      # D3: holiday / early-close window recognised
    OUT_OF_SESSION = "out_of_session"              # D4
    OVERFULL = "overfull"                          # D5
    REPEATED_BLOCK = "repeated_block"              # D6
    INTERLEAVE = "interleave"                      # D7
    CLOCK_MISALIGNED = "clock_misaligned"          # D8a
    EVENT_STAMP_DRIFT = "event_stamp_drift"        # D8b
    SPIKE_SUSPECT = "spike_suspect"                # D9
    MARKET_EVENT = "market_event"                  # D9: same move on several symbols
    STALE_RUN = "stale_run"                        # D10
    ZERO_VOLUME = "zero_volume"                    # D10
    SPREAD_INVALID = "spread_invalid"              # D11: negative spread (impossible)
    SPREAD_ZERO = "spread_zero"                    # D11: zero spread (INFO: real on raw accounts, or a recording quirk)
    HISTORY_REWRITE = "history_rewrite"            # D12: bars changed / vanished / inserted vs the last snapshot
    BAD_PRICES = "bad_prices"                      # D13: disagrees with an independent reference
    REGISTERED = "registered"                      # a registry window applying to the symbol
    PROPOSED_OVERRIDE = "proposed_override"        # a PROPOSED clock override window (dropped and refetched)


class Action(enum.StrEnum):
    """What remediation does with a finding (SPEC §6)."""

    DROP = "drop"          # remove the bars, then refetch
    REFETCH = "refetch"    # missing bars or a suspect bar: ask the broker again
    FLAG = "flag"          # keep, set a quality flag
    NONE = "none"          # informational


class Verdict(enum.StrEnum):
    CLEAN = "clean"
    USABLE_AFTER_REMEDIATION = "usable_after_remediation"
    REVIEW = "review"


class QualityFlag(enum.IntFlag):
    """Per-bar quality flags written next to remediated bars (SPEC §6.7). Hygiene masks, not model features."""

    NONE = 0
    AFTER_GAP = 1 << 0             # first bar after missing OPEN slots: its open carries the gap return
    NEAR_DROPPED = 1 << 1          # within 12 bars of a dropped window
    SPIKE_SUSPECT = 1 << 2         # single-symbol spike (refetch decides in P4)
    MARKET_EVENT = 1 << 3          # same move on several same-class symbols: real
    SPREAD_INVALID = 1 << 4        # negative spread
    SPREAD_ZERO = 1 << 5           # zero spread
    THIN_SLOT = 1 << 6             # bar in a declared THIN range (e.g. FX rollover hour)
    STALE_RUN = 1 << 7             # close unchanged for a long run
    ZERO_VOLUME = 1 << 8           # tick_volume == 0
    DST_FOLD = 1 << 9              # raw hour the server clock passed twice
    INTERLEAVE_SUSPECT = 1 << 10   # alternation not confirmed as a feed event
    REFERENCE_DIVERGENCE = 1 << 11  # MAJOR D13 divergence kept for inspection
    HELD_FOR_REVIEW = 1 << 12      # inside an unregistered judgment finding (clock / bad prices): kept, not promoted
    RESTAMPED = 1 << 13            # refetched bar restamped by the overlap handshake (the broker's clock moved)
    REFETCHED = 1 << 14            # bar came from a verified refetch (a filled hole or a replaced spike)
    SPIKE_CONFIRMED = 1 << 15      # spike suspect the broker served again unchanged: what a live system saw, kept


class RemediationStatus(enum.StrEnum):
    VERIFIED = "verified"          # staged copy re-audits with no CRITICAL detector findings
    REVIEW = "review"              # held: unregistered judgment findings need a decision in the registry
    FAILED_VERIFY = "failed_verify"  # CRITICAL findings remain after the mechanical fixes (a bug or a new defect type)
    EMPTY = "empty"                # every bar was dropped: nothing is staged, and nothing may be promoted for this symbol


class TargetKind(enum.StrEnum):
    """Why a window is refetched (SPEC §6.4-6.5)."""

    MISSING = "missing"    # expected bars absent from the raw file (D3 outage / hole)
    DROPPED = "dropped"    # bars removed by remediation (registry drop, proposed clock override, mechanical defect)
    SPIKE = "spike"        # single-symbol spike suspect: ask the broker for the bar again


class AttemptResult(enum.StrEnum):
    """Outcome of one refetch attempt. EMPTY, REJECTED_VERIFY and CONFIRMED count towards PERMANENT; the rest don't."""

    FILLED = "filled"                    # every missing slot filled with verified broker bars
    PARTIAL = "partial"                  # some slots filled (progress, not counted)
    REPLACED = "replaced"                # spike: the broker's bar differs and no longer spikes
    CONFIRMED = "confirmed"              # the broker serves the same bar(s) again: an unchanged spike, or the dropped bars
    EMPTY = "empty"                      # the broker served no bars for the missing slots
    REJECTED_VERIFY = "rejected_verify"  # overlap handshake or post-splice verification failed
    TRANSIENT = "transient"              # terminal / IPC / timeout error after in-run retries (not counted)
    NO_MARGIN = "no_margin"              # file start/end: no trusted bars on one side, so not asked (not counted)


COUNTED_RESULTS = frozenset({AttemptResult.EMPTY, AttemptResult.REJECTED_VERIFY, AttemptResult.CONFIRMED})


class HoleStatus(enum.StrEnum):
    """Hole-ledger state of one refetch target."""

    OPEN = "open"              # refetched on the next fetch run
    PARTIAL = "partial"        # some slots filled; the rest stays open
    FILLED = "filled"          # resolved with verified broker bars (kept in the state folder's fill store)
    PERMANENT = "permanent"    # 3 counted failed attempts >= 24 h apart: never refetched unless forced
    RESOLVED = "resolved"      # no longer a target: the raw export now has the bars
