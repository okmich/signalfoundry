"""Position managers: MT5 semantics on the async interface, with None for unset levels."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_core.price_buffer import PriceBuffer
from okmich_quant_crypto.position_manager import get_position_manager, wilder_atr

from .conftest import make_cfg


def _pm(pm_cfg, **kw):
    return get_position_manager(make_cfg(position_manager=pm_cfg), **kw)


def _pos(qty=1.0, entry=100.0, current=100.0, sl=None, tp=None):
    return {"position_id": "p1", "position": qty, "price_open": entry, "avg_cost": entry, "price_current": current,
            "stop_loss": sl, "take_profit": tp}


def test_fixed_percent_sets_initial_levels_once():
    pm = _pm({"type": "fixed_percent", "sl": 2, "tp": 4})
    assert pm._evaluate_position(_pos()) == (False, pytest.approx(98), pytest.approx(104))
    assert pm._evaluate_position(_pos(sl=98, tp=104)) == (False, None, None)
    short = pm._evaluate_position(_pos(qty=-1))
    assert short == (False, pytest.approx(102), pytest.approx(96))


def test_fixed_point_needs_point_size_and_uses_it():
    pm = _pm({"type": "fixed_point", "sl": 50, "tp": 100, "point_size": 0.1})
    assert pm.initial_levels(100.0, True) == (pytest.approx(95), pytest.approx(110))


def test_trailing_only_tightens_and_only_in_profit():
    pm = _pm({"type": "fixed_percent_with_trailing", "sl": 2, "tp": 10, "trailing": 1})
    assert pm._evaluate_position(_pos(current=99, sl=98, tp=110)) == (False, None, None)      # not in profit
    _, sl, _ = pm._evaluate_position(_pos(current=105, sl=98, tp=110))
    assert sl == pytest.approx(105 * 0.99)
    assert pm._evaluate_position(_pos(current=103, sl=103.95, tp=110)) == (False, None, None)  # never loosens


def test_trailing_for_shorts_with_unset_stop_works():
    # MT5 compares against sl=0.0 and so never trails a short whose stop is unset; None fixes that.
    pm = _pm({"type": "fixed_percent_with_trailing", "sl": 0, "tp": 0, "trailing": 1})
    _, sl, _ = pm._evaluate_position(_pos(qty=-1, current=95))
    assert sl == pytest.approx(95 * 1.01)


def test_break_even_moves_stop_to_at_least_entry():
    pm = _pm({"type": "fixed_percent_with_break_even", "sl": 2, "tp": 10, "break_even": 1, "trailing": 5})
    assert pm._evaluate_position(_pos(current=100.5, sl=98, tp=110)) == (False, None, None)
    _, sl, _ = pm._evaluate_position(_pos(current=102, sl=98, tp=110))
    assert sl == pytest.approx(100)  # max(102 - 5.1, entry)


def test_dynamic_trails_only_beyond_entry():
    pm = _pm({"type": "dynamic_percent", "sl": 2, "trailing": 3})
    assert pm._evaluate_position(_pos(current=102, sl=98)) == (False, None, None)   # 98.94 < entry
    _, sl, _ = pm._evaluate_position(_pos(current=110, sl=98))
    assert sl == pytest.approx(110 * 0.97)


def test_max_loss_amount_closes():
    pm = _pm({"type": "max_loss_amount", "max_loss_amount": 50})
    assert pm._evaluate_position(_pos(qty=2, current=70))[0] is True      # -30 * 2 = -60 <= -50
    assert pm._evaluate_position(_pos(qty=2, current=80))[0] is False     # -20 * 2 = -40
    assert pm._evaluate_position(_pos(qty=-2, current=130))[0] is True    # short: -30 * 2


def test_max_loss_stop_loss_derives_the_stop_from_size():
    pm = _pm({"type": "max_loss_stop_loss", "max_loss_amount": 50})
    assert pm._evaluate_position(_pos(qty=2)) == (False, pytest.approx(75), None)
    assert pm._evaluate_position(_pos(qty=-0.5)) == (False, pytest.approx(200), None)
    assert pm._evaluate_position(_pos(qty=2, sl=80)) == (False, None, None)


def test_wilder_atr_matches_reference():
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(size=60))
    high, low = close + rng.uniform(0.1, 1, 60), close - rng.uniform(0.1, 1, 60)
    period = 14
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    ref = tr[:period].mean()
    for x in tr[period:]:
        ref = (ref * (period - 1) + x) / period
    assert wilder_atr(high, low, close, period) == pytest.approx(ref)
    assert wilder_atr(high[:5], low[:5], close[:5], period) is None


def test_atr_manager_reads_closed_bars_from_the_buffer():
    buf = PriceBuffer(symbol="X", timeframe="5m", buffer_size=50, timeframe_minutes=5)
    idx = pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC", name="date")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1.0}, index=idx)
    buf.update(df, idx[-1] + pd.Timedelta(minutes=10))
    pm = _pm({"type": "fixed_atr", "sl": 2, "tp": 3, "atr_period": 14}, price_buffer=buf)
    assert pm.initial_levels(100.0, True) == (pytest.approx(96), pytest.approx(106))


async def test_manage_positions_routes_to_apply_and_close():
    pm = _pm({"type": "fixed_percent", "sl": 2, "tp": 4})
    applied, closed = [], []

    async def apply(p, sl, tp):
        applied.append((p["position_id"], sl, tp))
        return True

    async def close(p, reason):
        closed.append(p["position_id"])
        return True

    assert await pm.manage_positions([_pos()], apply, close) == set()
    assert applied == [("p1", pytest.approx(98), pytest.approx(104))]
    loss = _pm({"type": "max_loss_amount", "max_loss_amount": 1})
    assert await loss.manage_positions([_pos(current=50)], apply, close) == {"p1"}
