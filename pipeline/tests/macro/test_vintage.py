"""Vintage sprint tests — FRED API key (env), ALFRED first-print fetcher, source dispatch, HY-OAS.

All offline: the key is read from ``$FRED_API_KEY``; the ALFRED fetcher is exercised on a recorded
JSON fixture; no test hits the network.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from okmich_quant_pipeline.macro import update as update_mod
from okmich_quant_pipeline.macro._types import SERIES, MacroSeries
from okmich_quant_pipeline.macro.fetchers import alfred
from okmich_quant_pipeline.macro.fetchers.alfred import _parse
from okmich_quant_pipeline.macro.metastore import MacroMetastore

_VALID = "9d6" + "a" * 27 + "bc"  # 32 lowercase-alnum chars; not a real key
_FIX = Path(__file__).parent / "fixtures"


def _fix(name: str) -> str:
    return (_FIX / name).read_text()


# --------------------------------------------------------------------------- #
# S0.1 — FRED API key from $FRED_API_KEY (env only; secret, never logged)
# --------------------------------------------------------------------------- #

def test_fred_api_key_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", f"  {_VALID}  ")  # surrounding whitespace stripped
    assert update_mod._fred_api_key() == _VALID


def test_fred_api_key_missing_raises_naming_the_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        update_mod._fred_api_key()
    assert "FRED_API_KEY" in str(exc.value)  # error names the env var, carries no key value


def test_fred_api_key_blank_env_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRED_API_KEY", "   ")  # blank is not a key
    with pytest.raises(RuntimeError):
        update_mod._fred_api_key()


# --------------------------------------------------------------------------- #
# S0.2 — ALFRED fetcher (parsed on recorded fixtures; no network)
# --------------------------------------------------------------------------- #

def test_alfred_parse_vintage_availability_from_realtime_start() -> None:
    spec = SERIES[MacroSeries.NFCI]
    df = _parse(_fix("alfred_nfci_first_release.json"), MacroSeries.NFCI, spec, vintage=True)
    assert list(df.columns) == ["date", "series", "value", "available_from_utc"]
    row = df[df["date"] == pd.Timestamp("2024-01-05")].iloc[0]
    assert row["value"] == -0.50717                                        # first-print value
    # availability = the real first-print date (2024-01-10), stamped EOD UTC — not the +6d heuristic.
    assert row["available_from_utc"] == pd.Timestamp("2024-01-10 22:00", tz="UTC")
    assert str(df["available_from_utc"].dtype) == "datetime64[ns, UTC]"


def test_alfred_parse_nonvintage_uses_lag_policy() -> None:
    spec = SERIES[MacroSeries.CREDIT_SPREAD]
    df = _parse(_fix("alfred_baa10y_latest.json"), MacroSeries.CREDIT_SPREAD, spec, vintage=False)
    # Non-vintage path stamps via the series' availability policy, identical to the CSV fetcher.
    expected = spec.availability.stamp(df[["date"]])
    assert (df["available_from_utc"].to_numpy() == expected.to_numpy()).all()


def test_alfred_parse_empty_observations_raises() -> None:
    with pytest.raises(ValueError, match="no observations"):
        _parse('{"observations": []}', MacroSeries.NFCI, SERIES[MacroSeries.NFCI], vintage=True)


def test_alfred_fetch_scrubs_key_from_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    key = "k" * 32

    def _boom(url, **_kw):
        raise RuntimeError(f"500 Server Error for url ...api_key={key}&output_type=4")

    monkeypatch.setattr(alfred, "get", _boom)
    with pytest.raises(RuntimeError) as exc:
        alfred.fetch(MacroSeries.NFCI, dt.date(2024, 1, 1), dt.date(2024, 1, 2), api_key=key, output_type=4)
    msg = str(exc.value)
    assert key not in msg and "***" in msg  # the key-bearing URL must never reach a log


def test_alfred_fetch_uses_fixture_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # Stub the HTTP layer with the recorded body to exercise fetch() (URL build + parse) offline.
    monkeypatch.setattr(alfred, "get", lambda url, **_kw: type("R", (), {"text": _fix("alfred_nfci_first_release.json")})())
    df = alfred.fetch(MacroSeries.NFCI, dt.date(2024, 1, 1), dt.date(2024, 2, 29), api_key=_VALID, output_type=4)
    assert len(df) and df["available_from_utc"].notna().all()


# --------------------------------------------------------------------------- #
# S1 — source dispatch + vintage idempotency
# --------------------------------------------------------------------------- #

def _frame(series: MacroSeries, dates: list[str], values: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"date": pd.to_datetime(dates), "series": series.value, "value": values})
    df["available_from_utc"] = df["date"].dt.tz_localize("UTC")
    return df


def test_update_dispatches_api_for_hy_oas_csv_for_others(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(update_mod.alfred, "fetch",
                        lambda series, start, end, *, api_key, output_type: calls.__setitem__("alfred", (series, output_type)) or _frame(series, ["2024-01-05"], [3.5]))
    monkeypatch.setattr(update_mod.fred, "fetch",
                        lambda series, start, end: calls.__setitem__("fred", series) or _frame(series, ["2024-01-05"], [1.0]))
    monkeypatch.setattr(update_mod, "_fred_api_key", lambda: "k" * 32)
    ms = MacroMetastore(tmp_path)

    update_mod.update_series(MacroSeries.HY_OAS, tmp_path, ms, full=True, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), overlap_days=60)
    assert calls["alfred"] == (MacroSeries.HY_OAS, 1) and "fred" not in calls  # API, latest (vintage=False)

    # A CSV series must use fred.fetch and never touch the key loader.
    monkeypatch.setattr(update_mod, "_fred_api_key", lambda: pytest.fail("key loaded for a CSV series"))
    update_mod.update_series(MacroSeries.VIX, tmp_path, ms, full=True, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), overlap_days=60)
    assert calls["fred"] == MacroSeries.VIX


def test_update_vintage_series_requests_output_type_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No production series is vintage=True today (NFCI stayed on CSV), so flip HY_OAS to vintage to
    # prove the dispatch maps vintage -> output_type=4.
    captured: dict[str, int] = {}
    monkeypatch.setitem(update_mod.SERIES, MacroSeries.HY_OAS, replace(SERIES[MacroSeries.HY_OAS], vintage=True))
    monkeypatch.setattr(update_mod.alfred, "fetch",
                        lambda series, start, end, *, api_key, output_type: captured.__setitem__("ot", output_type) or _frame(series, ["2024-01-05"], [3.5]))
    monkeypatch.setattr(update_mod, "_fred_api_key", lambda: "k" * 32)
    update_mod.update_series(MacroSeries.HY_OAS, tmp_path, MacroMetastore(tmp_path), full=True, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), overlap_days=60)
    assert captured["ot"] == 4


def test_api_series_refetch_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(update_mod, "_fred_api_key", lambda: "k" * 32)
    monkeypatch.setattr(update_mod.alfred, "fetch",
                        lambda series, start, end, *, api_key, output_type: _frame(series, ["2024-01-05", "2024-01-12"], [3.5, 3.6]))
    ms = MacroMetastore(tmp_path)
    r1 = update_mod.update_series(MacroSeries.HY_OAS, tmp_path, ms, full=True, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), overlap_days=60)
    # Incremental re-fetch returns the same values — count and values unchanged (store stays stable).
    r2 = update_mod.update_series(MacroSeries.HY_OAS, tmp_path, ms, full=False, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 31), overlap_days=60)
    assert r1["n_obs"] == r2["n_obs"] == 2
    out = pd.read_parquet(tmp_path / "HY_OAS.parquet").set_index("date")["value"]
    assert out.loc["2024-01-05"] == 3.5


def test_hy_oas_recipes_are_opt_in_not_default() -> None:
    from okmich_quant_pipeline.macro.features import DEFAULT_RECIPES, HY_OAS_RECIPES
    default_names = {r.name for r in DEFAULT_RECIPES}
    assert {r.name for r in HY_OAS_RECIPES} == {"hy_oas_level", "hy_oas_z20", "hy_oas_chg5"}
    assert not (default_names & {r.name for r in HY_OAS_RECIPES})  # never shipped in the defaults
