"""OHLC audit for raw broker bar data (SPEC.md; README.md for usage). CLI: ``raw-data-audit``.

    from okmich_quant_utils.ohlc_audit import audit_broker, get_profile

    rep = audit_broker("FXPIG-Server", data_dir=r"D:\\data_dump\\market_data", registry_path="fxpig_bad_windows.csv")
    rep.summary_frame(); rep.findings_frame(); rep.events
    run = remediate_broker("FXPIG-Server", state_dir=r"D:\\quant_state", registry_path="fxpig_bad_windows.csv")
    with MT5FetchAdapter.for_profile(get_profile("FXPIG-Server")) as broker:          # P4: refetch, read-only
        run = remediate_broker("FXPIG-Server", state_dir=..., registry_path=..., adapter=broker)

Run inputs, never stored in a profile:
- the market-data root: argument, else ``OKMICH_QUANT_MARKET_DATA_DIR``, else fail;
- the state folder: argument, else ``OKMICH_QUANT_DATA_STATE_DIR``, else fail, and never inside the data tree;
- the bad-window registry (CSV);
- the blind-after cutoff;
- the report folder (temp by default).
Nothing is ever written into the market-data tree.
"""
from .audit import BrokerAuditReport, SymbolReport, audit_broker, audit_symbol
from .calendar import Anchor, DailySession, HolidayRule, SessionTemplate, expected_grid
from .clock import ClockEra, ClockOverride, DstSwitch, EraShiftClock, ZoneClock
from .enums import (Action, AssetClass, AttemptResult, FindingKind, HolidayEffect, HoleStatus, OverrideStatus, QualityFlag, RemediationStatus,
                    Severity, SlotState, TargetKind, Timeframe, Verdict, WindowAction, WindowKind)
from .fetch import FetchAdapter, FetchError, FetchUnavailableError, MT5FetchAdapter, TransientFetchError
from .findings import Finding
from .handshake import Handshake, handshake
from .ledger import HoleLedger, Target
from .profile import (MARKET_DATA_DIR_VAR, BrokerProfile, FetchSource, MarketDataDirNotConfiguredError, ReferenceSpec, SymbolSpec,
                      resolve_market_data_dir)
from .profiles import PROFILES, get_profile
from .refetch import RefetchConfig, RefetchRun
from .registry import BadWindow, load_registry
from .remediate import RemediationRun, SymbolRemediation, remediate_broker
from .snapshot import SnapshotCoverageError, history_diff, snapshot_broker
from .state import STATE_DIR_VAR, StateDirNotConfiguredError, StateLayout, resolve_state_dir
from .symbol_data import BAR_COLUMNS, NotABarSeries

__all__ = [
    "audit_broker", "audit_symbol", "BrokerAuditReport", "SymbolReport", "Finding",
    "get_profile", "PROFILES", "BrokerProfile", "SymbolSpec", "FetchSource", "ReferenceSpec",
    "resolve_market_data_dir", "MARKET_DATA_DIR_VAR", "MarketDataDirNotConfiguredError", "NotABarSeries", "BAR_COLUMNS",
    "EraShiftClock", "ZoneClock", "ClockEra", "ClockOverride", "DstSwitch",
    "SessionTemplate", "DailySession", "HolidayRule", "Anchor", "expected_grid",
    "load_registry", "BadWindow",
    "remediate_broker", "RemediationRun", "SymbolRemediation", "snapshot_broker", "history_diff", "SnapshotCoverageError",
    "MT5FetchAdapter", "FetchAdapter", "FetchError", "FetchUnavailableError", "TransientFetchError", "RefetchConfig", "RefetchRun",
    "HoleLedger", "Target", "Handshake", "handshake", "AttemptResult", "HoleStatus", "TargetKind",
    "resolve_state_dir", "STATE_DIR_VAR", "StateDirNotConfiguredError", "StateLayout", "QualityFlag", "RemediationStatus",
    "Action", "AssetClass", "FindingKind", "HolidayEffect", "OverrideStatus", "Severity", "SlotState", "Timeframe", "Verdict",
    "WindowAction", "WindowKind",
]
