"""FXPIG-Server profile (MT5 "NY close" server reached through the FXIFY demo account).

Clock evidence: memory ``fxpig-broker-clock`` and the 2026-11-11 audit. Era 4 starts at the 2026-10-29 export splice
(raw 20:45); the splice window itself is a registry DROP. Per-symbol deviations are PROPOSED: detected by cross-pair
return alignment, not yet confirmed, so remediation drops and refetches them instead of restamping.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import datetime as dt

from ..calendar import (Anchor, DailySession, HolidayRule, SessionTemplate, day_after, eve_of_first, explicit, financial,
                        fixed, minus, union)
from ..clock import ClockEra, ClockOverride, DstSwitch, EraShiftClock
from ..profile import BrokerProfile, FetchSource, ReferenceSpec, SymbolSpec
from ..enums import AssetClass, OverrideStatus, SlotState

T = pd.Timestamp
t = dt.time
NY, BERLIN, HK, UTC = "America/New_York", "Europe/Berlin", "Asia/Hong_Kong", "UTC"
WEEKDAYS = frozenset(range(5))

ERAS = (
    ClockEra(T("2000-01-01"), 7.0),            # FXIFY-old history (merged from 5-history) and FXIFY to the seam
    ClockEra(T("2024-06-05 00:00"), 9.0),
    ClockEra(T("2026-08-30 00:00"), 6.0),
    ClockEra(T("2026-10-29 20:45"), 7.0),      # export splice; raw 20:45 -> 10-30 00:15 is corrupt (registry)
)

_AUD = frozenset({"AUDUSD", "AUDCAD", "AUDJPY"})
_CRYPTO = frozenset({"BTCUSDT", "ETHUSDT"})
OVERRIDES = (
    ClockOverride(_CRYPTO, T("2026-08-29 18:05"), T("2026-08-30 00:00"), 6.0, OverrideStatus.CONFIRMED,
                  "crypto feed switched to NY+6 at Sat 16:05-18:05 raw (interleave, registry), not at the era boundary; "
                  "measured bar by bar against Binance"),
    ClockOverride(frozenset({"USDCAD"}), T("2026-08-23 00:00"), T("2026-08-29 06:00"), 6.0, OverrideStatus.PROPOSED,
                  "week stamped on the NY+6 clock one week early (lag -36 bars vs USDCHF)"),
    ClockOverride(_AUD, T("2026-09-13 00:00"), T("2026-09-19 06:00"), 9.0, OverrideStatus.PROPOSED,
                  "AUD pairs still on the old NY+9 clock (lag +36 bars vs GBPUSD/EURCAD/EURJPY)"),
    ClockOverride(_AUD, T("2026-09-30 00:00"), T("2026-10-17 06:00"), 9.0, OverrideStatus.PROPOSED,
                  "AUD pairs back on the old NY+9 clock (lag +36 bars)"),
)

#: The crypto feed (the only FXPIG symbols trading over the weekend) applies US DST changes at its own moment. These
#: were measured bar by bar against Binance UTC klines (2026-11-11). Other symbols are closed at every switch, so the
#: default NY moment is harmless for them. 2026-11-01 is unmeasured (post-cutoff, Binance export ends 2026-10-06).
DST_SWITCHES = (
    DstSwitch(_CRYPTO, dt.date(2024, 3, 10), T("2024-03-10 06:40"), "offset 2->3 h; raw gap 05:40->06:40"),
    DstSwitch(_CRYPTO, dt.date(2024, 11, 3), T("2024-11-02 04:45"), "offset 5->4 h, a day early"),
    DstSwitch(_CRYPTO, dt.date(2025, 3, 9), T("2025-03-09 06:00"), "inside the bad-price window 06:00-20:40 (registry)"),
    DstSwitch(_CRYPTO, dt.date(2025, 11, 2), T("2025-11-01 20:00"), "offset 5->4 h; repeated raw hour kept first pass"),
    DstSwitch(_CRYPTO, dt.date(2026, 3, 8), T("2026-03-08 07:00"), "offset 4->5 h; raw 07:00-08:00 mixed (registry)"),
)

# ---- sessions ------------------------------------------------------------------------------------------------------
# Derived from the bars (the lab's ohlc_audit validation/derive_sessions.py, 2026-11-11). Every non-crypto FXPIG symbol trades on a New York
# schedule, including DE30/HKG50/JPN225 (23 h CFDs, not home-exchange hours). What differs is the holiday calendar:
# - US indices, JPN225, metals follow CME: full closure on Good Friday / Christmas / New Year, early close on the other
#   NYSE holidays and on Dec 24 / Jul 3 / the day after Thanksgiving. Early-close anchors sit at 12:45 NY; observed last
#   bars run 12:45-14:30 by year and holiday, and bars inside an early-close window are tolerated.
# - DE30 closes for Xetra holidays: from 22:00 Berlin on the previous trading day to 00:15 UTC on the next.
# - HKG50 closes for HKEX holidays: from 03:00 HKT the morning after the previous trading day (end of the night
#   session) to 09:15 HKT on the next trading day. HKEX half-days close at 12:00 HKT.
# - FX closes for the Dec 25 and Jan 1 trading days; some years it reopens 20-80 min before 17:00. The 17:00-18:00 NY
#   rollover hour is THIN: crosses often print nothing for 30-60 min there (widespread from Aug 2026).
# - On US holidays and CME early-close days, DE30 stops at 22:00 Berlin (the Eurex close) instead of 17:00 NY. HKG50
#   stops anywhere from 16:30 HKT to 17:00 NY and reopens at 09:15 HKT; Memorial Day is the long case.
# - HKG50 also closes for HKEX weather closures (typhoon / black rainstorm). These are listed as they happen, since
#   they can't be predicted.

_XNYS_FULL = financial("XNYS", include=["Good Friday", "Christmas", "New Year"])
_XNYS_EARLY = financial("XNYS", exclude=["Good Friday", "Christmas", "New Year"])
_THANKSGIVING = financial("XNYS", include=["Thanksgiving"])
_CME_EARLY_EXTRA = minus(union(fixed(12, 24), fixed(7, 3), day_after(_THANKSGIVING)), financial("XNYS"))
_MOURNING_2025 = explicit("2025-01-09")      # NYSE closed for the Carter funeral; CME equity closed at 09:30 NY
_CME_OUTAGE_2025 = explicit("2025-11-28")    # CME Globex halt (data-centre cooling failure), Thu 21:40 -> Fri 08:30 NY


def _cme_rules(early_extra: bool = True, equity: bool = True) -> tuple[HolidayRule, ...]:
    rules = [
        HolidayRule("XNYS full closure (trading day)", _XNYS_FULL, Anchor(-1, t(18), NY), Anchor(0, t(17), NY)),
        HolidayRule("XNYS early close", minus(_XNYS_EARLY, _MOURNING_2025) if equity else _XNYS_EARLY,
                    Anchor(0, t(12, 45), NY), Anchor(0, t(17), NY), SlotState.EARLY_CLOSE),
        HolidayRule("CME Globex outage 2025-11-28", _CME_OUTAGE_2025, Anchor(-1, t(21, 40), NY), Anchor(0, t(8, 30), NY)),
    ]
    if early_extra:
        rules.append(HolidayRule("CME early close (Dec 24, Jul 3, Black Friday)", _CME_EARLY_EXTRA, Anchor(0, t(12, 45), NY),
                                 Anchor(0, t(17), NY), SlotState.EARLY_CLOSE))
    if equity:
        rules.append(HolidayRule("CME equity early close 2025-01-09", _MOURNING_2025, Anchor(0, t(9, 30), NY), Anchor(0, t(17), NY),
                                 SlotState.EARLY_CLOSE))
    return tuple(rules)


_CFD_23H = DailySession(t(18), t(17), 1, WEEKDAYS)
_US_SHORT_DAYS = union(_XNYS_EARLY, _CME_EARLY_EXTRA)            # US holidays the CFDs trade through, plus CME early closes
_HKEX_WEATHER = explicit("2023-09-01", "2023-09-08", "2023-10-09", "2024-09-06")   # Saola, black rainstorm, Koinu, Yagi

SESSIONS = {
    "fx": SessionTemplate("fx", NY, DailySession(t(17), t(17), 1, WEEKDAYS), (
        HolidayRule("FX Christmas / New Year (trading day)", union(fixed(12, 25, False), fixed(1, 1, False)),
                    Anchor(-1, t(17), NY), Anchor(0, t(17), NY)),
    ), thin_ranges=((t(17), t(18)),)),
    "metal": SessionTemplate("metal", NY, DailySession(t(18), t(17), 1, WEEKDAYS, {4: t(16, 50)}), _cme_rules(equity=False)),
    "us_index": SessionTemplate("us_index", NY, _CFD_23H, _cme_rules()),
    "jpn225": SessionTemplate("jpn225", NY, _CFD_23H, _cme_rules()),
    "de30": SessionTemplate("de30", NY, _CFD_23H, (
        HolidayRule("Xetra holiday", financial("XETR"), Anchor(-1, t(22), BERLIN, business=True), Anchor(1, t(0, 15), UTC, business=True)),
        HolidayRule("US short day: Eurex-close stop", _US_SHORT_DAYS, Anchor(0, t(21, 55), BERLIN), Anchor(0, t(17), NY), SlotState.EARLY_CLOSE),
        HolidayRule("CME Globex outage 2025-11-28", _CME_OUTAGE_2025, Anchor(-1, t(21, 40), NY), Anchor(0, t(8, 30), NY)),
    )),
    "hkg50": SessionTemplate("hkg50", NY, _CFD_23H, (
        HolidayRule("HKEX holiday", financial("XHKG"), Anchor(-1, t(3), HK, business=True, then_days=1), Anchor(1, t(9, 15), HK, business=True)),
        HolidayRule("HKEX weather closure", _HKEX_WEATHER, Anchor(-1, t(17, 15), HK, business=True), Anchor(1, t(9, 15), HK, business=True)),
        HolidayRule("US short day: stop after the HK day session", _US_SHORT_DAYS, Anchor(0, t(16, 30), HK), Anchor(1, t(9, 15), HK, business=True),
                    SlotState.EARLY_CLOSE),
        HolidayRule("HKEX half day", minus(union(fixed(12, 24), fixed(12, 31), eve_of_first(financial("XHKG", include=["Chinese New Year"]))),
                                            financial("XHKG")),
                    Anchor(0, t(12), HK), Anchor(1, t(9, 15), HK, business=True), SlotState.EARLY_CLOSE),
    )),
    "crypto": SessionTemplate("crypto", UTC, DailySession(t(0), None, 0, frozenset(range(7)))),
}

_GROUPS: dict[AssetClass, tuple[str, list[str]]] = {
    AssetClass.FX: ("fx", ["AUDCAD", "AUDJPY", "AUDUSD", "EURCAD", "EURCHF", "EURGBP", "EURJPY", "EURUSD", "GBPCAD",
                           "GBPJPY", "GBPUSD", "USDCAD", "USDCHF", "USDJPY"]),
    AssetClass.METAL: ("metal", ["XAUUSD", "XAGUSD"]),
    AssetClass.US_INDEX: ("us_index", ["DJ30", "US500", "USTEC", "RUS2000"]),
    AssetClass.CRYPTO: ("crypto", ["BTCUSDT", "ETHUSDT"]),
}
_SINGLES = {"DE30": (AssetClass.EU_INDEX, "de30"), "HKG50": (AssetClass.ASIA_INDEX, "hkg50"),
            "JPN225": (AssetClass.ASIA_INDEX, "jpn225")}


def _symbols() -> dict[str, SymbolSpec]:
    out = {}
    for asset_class, (session, names) in _GROUPS.items():
        for n in names:
            out[n] = SymbolSpec(n, f"{n}.r", asset_class, session)
    for n, (asset_class, session) in _SINGLES.items():
        out[n] = SymbolSpec(n, f"{n}.r", asset_class, session)
    return out


#: Reference series for D13, relative to the raw root: CCXT Binance klines written by fetch-crypto-data.
_REFERENCES = {
    "BTCUSDT": ("crypto/binance/ohlcv/5m/BTC_USDT_USDT.parquet", "Binance BTC/USDT 5m"),
    "ETHUSDT": ("crypto/binance/ohlcv/5m/ETH_USDT_USDT.parquet", "Binance ETH/USDT 5m"),
}


def build(raw_root: Path | None = None) -> BrokerProfile:
    """FXPIG-Server 5m profile. ``raw_root`` is ``<market-data root>/raw``, or None when bars are supplied in memory."""
    refs = {s: ReferenceSpec(raw_root / rel, note=note) for s, (rel, note) in _REFERENCES.items()} if raw_root else {}
    return BrokerProfile(name="FXPIG-Server", bar_minutes=5, clock=EraShiftClock(NY, ERAS, OVERRIDES, DST_SWITCHES),
                         symbols=_symbols(), sessions=SESSIONS, fetch=FetchSource("fxify", "demo"), raw_root=raw_root, references=refs)
