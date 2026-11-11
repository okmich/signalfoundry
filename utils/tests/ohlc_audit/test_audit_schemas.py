"""Different bar schemas: extra columns are never an issue; non-bar datasets are rejected with a clear error.

Schemas, as the fetchers write them:
- MT5 (fetch-mt5-data): open, high, low, close, tick_volume, spread; index = broker wall clock (fake +02:00 label).
- IB (fetch-ib-data): open, high, low, close, volume, average (VWAP), barCount; index ``date`` in UTC. volume and
  average are -1 when IB has none (FX midpoint bars).
- CCXT crypto (fetch-crypto-data; Binance, Bybit): open, high, low, close, volume; index ``date`` in UTC. The same for
  mark / index / premium-index klines. A premium index is a basis around 0, so its prices can be <= 0.
- CCXT funding_rate / open_interest: not bars.
"""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.audit import audit_broker
from okmich_quant_utils.ohlc_audit.clock import ZoneClock
from okmich_quant_utils.ohlc_audit.enums import AssetClass, FindingKind, Severity
from okmich_quant_utils.ohlc_audit.profile import BrokerProfile, FetchSource, SymbolSpec
from okmich_quant_utils.ohlc_audit.profiles.fxpig import SESSIONS
from okmich_quant_utils.ohlc_audit.symbol_data import NotABarSeries, SymbolData

NY = "America/New_York"


def _profile(tmp_path, specs) -> BrokerProfile:
    return BrokerProfile("SCHEMA", 5, ZoneClock("UTC"), {s.name: s for s in specs}, SESSIONS, FetchSource("x", "y"))


def _walk(n, start, scale, rng):
    close = start * np.exp(np.cumsum(rng.normal(0, scale, n)))
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0, scale / 3, (2, n))) * close
    return open_, np.maximum(open_, close) + wick[0], np.minimum(open_, close) - wick[1], close


def _ib_frame(rng, fx_midpoint=False) -> pd.DataFrame:
    """CME-hours futures bars (Sun 18:00 -> Fri 17:00 NY, break 17:00-18:00), as fetch-ib-data stores them (UTC)."""
    days = pd.date_range("2024-04-07", "2024-05-10", freq="D")
    ny = []
    for d in days:
        if d.weekday() in (4, 5):                          # sessions open Sun-Thu evenings
            continue
        ny.append(pd.date_range(d + pd.Timedelta(hours=18), d + pd.Timedelta(hours=41), freq="5min", inclusive="left"))
    ny = ny[0].append(ny[1:]).tz_localize(NY)
    o, h, lo, c = _walk(len(ny), 5000.0, 4e-4, rng)
    vol = rng.integers(10, 500, len(ny)).astype(float)
    avg = (h + lo + c) / 3
    if fx_midpoint:
        vol[:], avg[:] = -1.0, -1.0
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": vol, "average": avg,
                         "barCount": rng.integers(5, 80, len(ny))}, index=pd.DatetimeIndex(ny.tz_convert("UTC"), name="date"))


def _ccxt_frame(rng, start=60000.0, scale=8e-4, premium=False) -> pd.DataFrame:
    utc = pd.date_range("2024-04-01", "2024-05-01", freq="5min", inclusive="left", tz="UTC", name="date")
    if premium:
        c = rng.normal(-2e-4, 3e-4, len(utc))
        o = np.r_[c[0], c[:-1]]
        h, lo = np.maximum(o, c) + 5e-5, np.minimum(o, c) - 5e-5
    else:
        o, h, lo, c = _walk(len(utc), start, scale, rng)
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": 0.0 if premium else rng.gamma(2, 50, len(utc))}, index=utc)


def _non_info(rep):
    return [f for r in rep.symbols.values() for f in r.detector_findings() if f.severity is not Severity.INFO]


def test_ib_schema_with_volume_vwap_barcount_is_clean(tmp_path):
    rng = np.random.default_rng(0)
    specs = [SymbolSpec("MES", "MES", AssetClass.US_INDEX, "us_index"), SymbolSpec("EURUSD_MID", "EURUSD_MID", AssetClass.US_INDEX, "us_index")]
    rep = audit_broker(profile=_profile(tmp_path, specs), frames={"MES": _ib_frame(rng), "EURUSD_MID": _ib_frame(rng, fx_midpoint=True)})
    assert _non_info(rep) == []                            # volume / average / barCount, and their -1 sentinels, are no issue


def test_ccxt_schemas_are_clean(tmp_path):
    rng = np.random.default_rng(1)
    specs = [SymbolSpec("BTC", "BTC", AssetClass.CRYPTO, "crypto"), SymbolSpec("BTC_MARK", "BTC_MARK", AssetClass.CRYPTO, "crypto"),
             SymbolSpec("BTC_PREMIUM", "BTC_PREMIUM", AssetClass.CRYPTO, "crypto", positive_prices=False)]
    frames = {"BTC": _ccxt_frame(rng), "BTC_MARK": _ccxt_frame(rng).assign(volume=0.0), "BTC_PREMIUM": _ccxt_frame(rng, premium=True)}
    assert _non_info(audit_broker(profile=_profile(tmp_path, specs), frames=frames)) == []


def test_positive_prices_setting_matters_for_a_premium_index(tmp_path):
    rng = np.random.default_rng(2)
    spec = SymbolSpec("BTC_PREMIUM", "BTC_PREMIUM", AssetClass.CRYPTO, "crypto")          # wrongly left at positive_prices=True
    rep = audit_broker(profile=_profile(tmp_path, [spec]), frames={"BTC_PREMIUM": _ccxt_frame(rng, premium=True)})
    assert any(f.kind is FindingKind.NON_POSITIVE for f in _non_info(rep))


def test_extra_columns_do_not_change_findings(tmp_path):
    rng = np.random.default_rng(3)
    spec = SymbolSpec("BTC", "BTC", AssetClass.CRYPTO, "crypto")
    base = _ccxt_frame(rng)
    base = base.drop(base.index[1000:1030])                # a 2.5 h outage, to have something to find
    extra = base.assign(real_volume=base["volume"] * 3, vwap=base["close"], trades=7, note="x", nan_col=np.nan)
    kinds = [sorted((f.kind.value, str(f.start_raw)) for f in _non_info(audit_broker(profile=_profile(tmp_path, [spec]), frames={"BTC": df})))
             for df in (base, extra)]
    assert kinds[0] == kinds[1] and kinds[0]


@pytest.mark.parametrize("frame, msg", [
    (pd.DataFrame({"funding_rate": [1e-4, 2e-4]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-01 08:00"], tz="UTC")), "not an OHLC bar series"),
    (pd.DataFrame({"open_interest_amount": [1.0], "open_interest_value": [2.0]}, index=pd.DatetimeIndex(["2024-01-01"], tz="UTC")), "not an OHLC"),
    (pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}, index=[0]), "DatetimeIndex"),
    (pd.DataFrame({"open": ["a"], "high": [1.0], "low": [1.0], "close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])), "non-numeric"),
])
def test_non_bar_datasets_are_rejected_clearly(tmp_path, frame, msg):
    profile = _profile(tmp_path, [SymbolSpec("X", "X", AssetClass.CRYPTO, "crypto")])
    with pytest.raises(NotABarSeries, match=msg):
        SymbolData.build(profile, "X", frame)


def test_zone_clock_utc_is_identity():
    idx = pd.DatetimeIndex(["2024-03-10 06:55", "2024-11-03 05:30"])
    assert (ZoneClock("UTC").to_utc(idx, "X") == idx.tz_localize("UTC")).all()
    assert ZoneClock("UTC").ambiguous_raw("X") == []
    assert len(ZoneClock(NY).ambiguous_raw("X")) > 10      # one repeated hour per US fall-back
