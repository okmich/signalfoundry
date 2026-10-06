"""Timeframe validation and client-order-id encoding."""
import re

import pytest

from okmich_quant_crypto.client_order_id import (
    ClientIdRule, client_order_prefix, is_ours, magic_of, make_client_order_id,
)
from okmich_quant_crypto.timeframe_utils import (
    bar_open_ms, last_closed_bar_open_ms, timeframe_to_minutes, validate_venue_timeframe,
)

OKX_RULE = ClientIdRule(max_length=32)
BYBIT_RULE = ClientIdRule(max_length=36, pattern=r"^[A-Za-z0-9_-]+$")


@pytest.mark.parametrize("tf,minutes", [("1m", 1), ("3m", 3), ("5m", 5), ("15m", 15), ("1h", 60), ("4h", 240),
                                        ("12h", 720), ("1d", 1440)])
def test_timeframes_up_to_one_day(tf, minutes):
    assert timeframe_to_minutes(tf) == minutes


@pytest.mark.parametrize("tf", ["1w", "1M", "3d", "30s", "7m", "", "5x", 5])
def test_timeframes_core_cannot_label_are_rejected(tf):
    with pytest.raises(ValueError):
        timeframe_to_minutes(tf)


def test_venue_must_offer_the_timeframe():
    offered = {"1m": "1", "5m": "5", "1w": "W"}
    assert validate_venue_timeframe("5m", offered) == 5
    with pytest.raises(ValueError, match=r"usable timeframes: \['1m', '5m'\]"):
        validate_venue_timeframe("15m", offered)


def test_bar_grid_helpers():
    tf = 300_000
    now = 10 * tf + 1234
    assert bar_open_ms(now, tf) == 10 * tf
    assert last_closed_bar_open_ms(now, tf) == 9 * tf
    assert last_closed_bar_open_ms(10 * tf, tf) == 9 * tf  # exactly on the boundary: the bar that just closed


def test_client_order_id_format_and_rules():
    cid = make_client_order_id(42, OKX_RULE)
    assert re.fullmatch(r"sf42x[0-9a-z]+", cid)
    assert len(cid) <= 32
    BYBIT_RULE.check(cid)  # strictest-common-denominator ids satisfy every certified venue
    assert magic_of(cid) == 42
    assert is_ours(cid, 42) and not is_ours(cid, 421) and not is_ours(cid, 4)


def test_prefix_cannot_collide_across_magics():
    assert not make_client_order_id(421, OKX_RULE).startswith(client_order_prefix(42))
    assert magic_of("sf421xabc") == 421


def test_ids_are_unique():
    ids = {make_client_order_id(7, OKX_RULE) for _ in range(2000)}
    assert len(ids) == 2000


def test_foreign_ids_are_not_ours():
    assert magic_of(None) is None
    assert magic_of("manual-order-1") is None
    assert not is_ours("web123", 42)


def test_rule_rejects_bad_ids():
    with pytest.raises(ValueError):
        OKX_RULE.check("sf42x-has-hyphen")
    with pytest.raises(ValueError):
        OKX_RULE.check("x" * 33)
    with pytest.raises(ValueError):
        make_client_order_id(10 ** 30, OKX_RULE)
