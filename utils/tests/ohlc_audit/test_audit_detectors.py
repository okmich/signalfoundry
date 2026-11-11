"""SPEC A1: synthetic injections are found with the right kind; a clean synthetic broker yields no findings."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.audit import audit_broker
from okmich_quant_utils.ohlc_audit.clock import ClockEra, EraShiftClock
from okmich_quant_utils.ohlc_audit.detectors.reference import reference_check
from okmich_quant_utils.ohlc_audit.enums import AssetClass, FindingKind, Severity, Verdict
from okmich_quant_utils.ohlc_audit.profile import BrokerProfile, FetchSource, ReferenceSpec, SymbolSpec
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.symbol_data import SymbolData

NY = "America/New_York"
SYMS = ("EURUSD", "GBPUSD", "AUDUSD")
START, WEEKS = pd.Timestamp("2024-04-07 17:00"), 6         # no FX holiday inside


def _profile(tmp_path, references=None) -> BrokerProfile:
    symbols = {s: SymbolSpec(s, s, AssetClass.FX, "fx") for s in SYMS}
    if references:
        symbols["BTCUSDT"] = SymbolSpec("BTCUSDT", "BTCUSDT", AssetClass.CRYPTO, "crypto")
    return BrokerProfile("SYNTH", 5, EraShiftClock(NY, (ClockEra(pd.Timestamp("2000-01-01"), 7.0),)), symbols,
                         {"fx": SESSIONS["fx"], "crypto": SESSIONS["crypto"]}, FetchSource("x", "y"), references=references or {})


def _ny_slots() -> pd.DatetimeIndex:
    out = []
    for w in range(WEEKS):
        o = START + pd.Timedelta(weeks=w)
        out.append(pd.date_range(o, o + pd.Timedelta(days=5), freq="5min", inclusive="left"))   # Sun 17:00 -> Fri 17:00 NY
    return out[0].append(out[1:])


def _frames(seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ny = _ny_slots()
    raw = ny + pd.Timedelta(hours=7)                       # NY + 7 h server clock
    factor = rng.normal(0, 1e-4, len(ny))
    out = {}
    for i, s in enumerate(SYMS):
        r = factor + rng.normal(0, 5e-5, len(ny))
        close = 1.1 * (i + 1) * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]]
        wick = np.abs(rng.normal(0, 3e-5, (2, len(ny)))) * close
        out[s] = pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0], "low": np.minimum(open_, close) - wick[1],
                               "close": close, "tick_volume": rng.integers(50, 200, len(ny)).astype(float),
                               "spread": rng.integers(5, 10, len(ny)).astype(np.int32)}, index=pd.DatetimeIndex(raw, name="raw"))
    return out


def _non_info(rep):
    return [f for r in rep.symbols.values() for f in r.detector_findings() if f.severity is not Severity.INFO]


def test_clean_synthetic_broker_has_no_findings(tmp_path):
    rep = audit_broker(profile=_profile(tmp_path), frames=_frames())
    assert _non_info(rep) == []
    assert all(r.verdict is Verdict.CLEAN for r in rep.symbols.values())


def _inject(frames):
    eur, gbp, aud = (frames[s].copy() for s in SYMS)
    wed = pd.Timestamp("2024-04-17 10:00") + pd.Timedelta(hours=7)          # raw
    eur = eur.drop(eur.loc[wed:wed + pd.Timedelta(minutes=115)].index)       # 2 h outage
    src = eur.loc[wed + pd.Timedelta(hours=3):wed + pd.Timedelta(hours=3, minutes=55)]
    dst_idx = eur.loc[wed + pd.Timedelta(hours=5):wed + pd.Timedelta(hours=5, minutes=55)].index
    eur.loc[dst_idx, ["open", "high", "low", "close"]] = src[["open", "high", "low", "close"]].to_numpy()    # repeated block
    fri = pd.Timestamp("2024-04-26 15:00") + pd.Timedelta(hours=7)
    sat = eur.loc[fri:fri + pd.Timedelta(minutes=115)].copy()
    sat.index = sat.index + pd.Timedelta(hours=13, minutes=30)               # re-stamped onto Saturday
    eur = pd.concat([eur, sat]).sort_index()
    stale_at = pd.Timestamp("2024-05-08 03:00") + pd.Timedelta(hours=7)
    stale_idx = eur.loc[stale_at:stale_at + pd.Timedelta(minutes=95)].index
    eur.loc[stale_idx, ["open", "high", "low", "close"]] = eur.loc[stale_idx[0], "close"]          # stale run
    wk0, wk1 = pd.Timestamp("2024-04-28 17:00") + pd.Timedelta(hours=7), pd.Timestamp("2024-05-03 17:00") + pd.Timedelta(hours=7)
    shifted = gbp.loc[wk0:wk1 - pd.Timedelta(minutes=5)].copy()
    gbp = gbp.drop(shifted.index)
    shifted.index = shifted.index + pd.Timedelta(hours=3)                    # a whole week stamped 3 h late
    gbp = pd.concat([gbp, shifted]).sort_index()
    il = pd.Timestamp("2024-04-23 09:00") + pd.Timedelta(hours=7)
    il_idx = aud.loc[il:il + pd.Timedelta(minutes=55)].index[::2]
    aud.loc[il_idx, "close"] = aud["close"].shift(36).loc[il_idx] * 1.004   # alternate bars from a feed 3 h away, offset
    aud.loc[il_idx, "high"] = aud.loc[il_idx, ["open", "close", "high"]].max(axis=1)
    aud.loc[il_idx, "low"] = aud.loc[il_idx, ["open", "close", "low"]].min(axis=1)
    sp = pd.Timestamp("2024-05-14 05:00") + pd.Timedelta(hours=7)
    aud.loc[sp, "high"] = aud.loc[sp, "close"] * 1.02                        # spike wick
    dup = aud.loc[[pd.Timestamp("2024-04-10 11:00") + pd.Timedelta(hours=7)]]
    aud = pd.concat([aud, dup]).sort_index(kind="stable")                    # duplicate stamp
    return {"EURUSD": eur, "GBPUSD": gbp, "AUDUSD": aud}


@pytest.mark.parametrize("symbol, kind, at_raw", [
    ("EURUSD", FindingKind.OUTAGE, "2024-04-17 17:30"),
    ("EURUSD", FindingKind.REPEATED_BLOCK, "2024-04-17 22:30"),
    ("EURUSD", FindingKind.OUT_OF_SESSION, "2024-04-27 12:00"),
    ("EURUSD", FindingKind.STALE_RUN, "2024-05-08 10:30"),
    ("GBPUSD", FindingKind.EVENT_STAMP_DRIFT, "2024-05-01 12:00"),
    ("GBPUSD", FindingKind.CLOCK_MISALIGNED, "2024-05-01 12:00"),
    ("AUDUSD", FindingKind.INTERLEAVE, "2024-04-23 16:20"),
    ("AUDUSD", FindingKind.SPIKE_SUSPECT, "2024-05-14 12:00"),
    ("AUDUSD", FindingKind.DUPLICATE_STAMP, "2024-04-10 18:00"),
])
def test_injected_defects_are_found(tmp_path, symbol, kind, at_raw):
    rep = audit_broker(profile=_profile(tmp_path), frames=_inject(_frames()))
    t = pd.Timestamp(at_raw)
    hits = [f for f in rep.symbols[symbol].detector_findings() if f.kind is kind and f.start_raw <= t < f.end_raw + pd.Timedelta(minutes=5)]
    assert hits, f"{kind} not found on {symbol} at {at_raw}"


def test_injected_run_needs_review(tmp_path):
    rep = audit_broker(profile=_profile(tmp_path), frames=_inject(_frames()))
    assert rep.symbols["GBPUSD"].verdict is Verdict.REVIEW          # unregistered clock shift: a human decides
    assert rep.symbols["EURUSD"].verdict is not Verdict.CLEAN


def test_reference_check_flags_bad_prices(tmp_path):
    utc = pd.date_range("2024-04-01", periods=12 * 288, freq="5min", tz="UTC")
    rng = np.random.default_rng(1)
    ref_close = 60000 * np.exp(np.cumsum(rng.normal(0, 8e-4, len(utc))))
    path = tmp_path / "ref.parquet"
    pd.DataFrame({"close": ref_close}, index=utc).to_parquet(path)
    profile = _profile(tmp_path, {"BTCUSDT": ReferenceSpec(path, note="synthetic")})
    x = ref_close * (1 + rng.normal(0, 1e-4, len(utc)))
    bad = slice(5 * 288, 5 * 288 + 24)
    x[bad] *= 1.01                                                   # 100 bp off for 2 hours
    raw = (utc.tz_convert(NY).tz_localize(None) + pd.Timedelta(hours=7))
    df = pd.DataFrame({"open": x, "high": x * 1.0002, "low": x * 0.9998, "close": x, "tick_volume": 100.0, "spread": 5}, index=raw)
    found = reference_check(SymbolData.build(profile, "BTCUSDT", df), profile.references["BTCUSDT"])
    assert [f.severity for f in found if f.kind is FindingKind.BAD_PRICES] == [Severity.CRITICAL]


def test_audit_is_deterministic(tmp_path):
    """Same input, same findings, row for row (remediation idempotency, SPEC A10, builds on this)."""
    frames = _inject(_frames())
    runs = [audit_broker(profile=_profile(tmp_path), frames={k: v.copy() for k, v in frames.items()}).findings_frame() for _ in range(2)]
    pd.testing.assert_frame_equal(runs[0], runs[1])
