import pandas as pd
import pytest

from okmich_quant_utils.ohlc_audit.profiles import get_profile
from okmich_quant_utils.ohlc_audit.registry import COLUMNS, in_windows, load_registry, windows_for
from okmich_quant_utils.ohlc_audit.enums import WindowAction

HEADER = ",".join(COLUMNS)


def _csv(tmp_path, *rows):
    p = tmp_path / "w.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return p


def test_registry_parses_all_and_symbol_lists(tmp_path):
    p = _csv(tmp_path, "a,ALL,2024-01-01 00:00,2024-01-01 01:00,splice,drop,e,2026-11-11,2026-11-11",
             "b,EURUSD;GBPUSD,2024-01-02 00:00,2024-01-02 01:00,outage,refetch,e,2026-11-11,2026-11-11")
    reg = load_registry(p)
    assert reg[0].symbols is None and reg[0].applies_to("XAUUSD")
    assert reg[1].applies_to("GBPUSD") and not reg[1].applies_to("XAUUSD")
    raw = pd.DatetimeIndex(["2024-01-01 00:30", "2024-01-01 01:00", "2024-01-02 00:10"])
    assert list(in_windows(raw, windows_for(reg, "EURUSD", {WindowAction.DROP}))) == [True, False, False]


@pytest.mark.parametrize("rows, msg", [
    (("a,ALL,2024-01-01,2024-01-02,splice,drop,e,2026-11-11,2026-11-11", "a,ALL,2024-01-03,2024-01-04,splice,drop,e,2026-11-11,2026-11-11"), "duplicate"),
    (("a,ALL,2024-01-02,2024-01-01,splice,drop,e,2026-11-11,2026-11-11",), "to_raw <= from_raw"),
])
def test_registry_rejects_bad_rows(tmp_path, rows, msg):
    with pytest.raises(ValueError, match=msg):
        load_registry(_csv(tmp_path, *rows))


def test_fxpig_profile_is_consistent():
    p = get_profile("FXPIG-Server", require_data=False)
    assert len(p.symbols) == 25
    assert all(s.session in p.sessions for s in p.symbols.values())
    assert p.fetch.env_broker == "fxify" and p.fetch.env_account == "demo"
    assert p.raw_root is None and p.references == {}       # run inputs, not profile data
