"""ACCOUNT_ADMIN_SPEC §7.2 clock (§14.8) and §6.7 stage-1 config validation (§14.13)."""
from __future__ import annotations

import copy
import random

import pytest

from okmich_quant_core.account_admin import (AdminClock, AdminConfigError, BlackoutWindow, ClockConfig, gap_max,
                                             longest_blackout_run, parse_admin_config)
from okmich_quant_core.account_admin._validate import Problems
from okmich_quant_core.account_admin.clock import HEARTBEAT_GAP_MAX_S, parse_clock

from .admin_fakes import admin_config, prop_guard_entry

FLEET = (BlackoutWindow(60, 0, 3), BlackoutWindow(300, 0, 8), BlackoutWindow(30, 0, 2))


def test_longest_blackout_run_merges_and_wraps():
    assert longest_blackout_run(FLEET) == 8
    assert longest_blackout_run((BlackoutWindow(60, 58, 4),)) == 4               # wraps the minute
    assert longest_blackout_run((BlackoutWindow(60, 0, 3), BlackoutWindow(60, 3, 2))) == 5   # contiguous windows merge
    with pytest.raises(ValueError):
        longest_blackout_run((BlackoutWindow(2, 0, 1), BlackoutWindow(2, 1, 1)))  # blocks every second


def test_property_no_start_in_a_blackout_and_every_gap_bounded():
    """§14.8: 10^5 simulated cycles, including overruns."""
    config = ClockConfig(cycle_s=20, jitter_s=5, valid_for_s=240, blackout=FLEET)
    bound = gap_max(config)
    b = longest_blackout_run(FLEET)
    rng = random.Random(7)
    clock = AdminClock(config, rng=random.Random(11))
    t = clock.release(1_700_000_000.0)
    for _ in range(100_000):
        assert not clock.is_blocked(t), f"cycle started at {t % 300:.3f}s into the 5-minute period"
        duration = rng.choice((0.05, 0.3, 2.0, 30.0)) if rng.random() < 0.01 else rng.uniform(0.01, 0.4)
        nxt = clock.next_start(t, t + duration)
        gap = nxt - t
        assert gap <= max(bound, duration + b + 1) + 1e-9
        if duration < config.cycle_s - config.jitter_s:
            assert gap <= bound + 1e-9
        t = nxt


def _clock_problems(**raw):
    base = {"cycle_s": 20, "jitter_s": 5, "valid_for_s": 240,
            "blackout": [{"every_s": 60, "offset_s": 0, "length_s": 3}]}
    base.update(raw)
    p = Problems("clock")
    return parse_clock(base, p), p.items


def test_clock_bounds_refuse_to_start():
    assert _clock_problems()[0] is not None
    assert any("jitter_s" in m for m in _clock_problems(jitter_s=10)[1])
    assert any("valid_for_s / 4" in m for m in _clock_problems(valid_for_s=80)[1])
    assert any(f"<= {HEARTBEAT_GAP_MAX_S}s" in m for m in _clock_problems(cycle_s=120, valid_for_s=2000)[1])
    assert any("shorter than every_s" in m for m in _clock_problems(blackout=[{"every_s": 5, "offset_s": 0, "length_s": 5}])[1])


def test_a_valid_config_builds_its_tasks():
    cfg = parse_admin_config(admin_config([prop_guard_entry(), {"kind": "pending_order_cleanup", "max_age_s": 3600}]))
    assert [t.kind.value for t in cfg.tasks] == ["prop_guard", "pending_order_cleanup"]
    assert cfg.governing_task is cfg.tasks[0]


def test_every_problem_is_reported_in_one_pass():
    raw = admin_config([prop_guard_entry(daily_loss_pct=150, day_tz="Mars/Olympus"),
                        {"kind": "pending_order_cleanup", "max_age_s": 5, "scope": "magics"},
                        {"kind": "coffee_maker"}])
    raw["strategies"] = [{}]
    raw["clock"]["jitter_s"] = 15
    with pytest.raises(AdminConfigError) as err:
        parse_admin_config(raw)
    text = "\n".join(err.value.problems)
    for expected in ("daily_loss_pct", "day_tz", "max_age_s", "tasks[1].magics", "coffee_maker", "strategies", "jitter_s"):
        assert expected in text, f"{expected} not reported:\n{text}"


def test_duplicate_kind_and_empty_tasks_are_refused():
    with pytest.raises(AdminConfigError, match="more than once"):
        parse_admin_config(admin_config([prop_guard_entry(), prop_guard_entry()]))
    with pytest.raises(AdminConfigError, match="non-empty list"):
        parse_admin_config(admin_config([]))


@pytest.mark.parametrize("mutate,needle", [
    (lambda p: p["policy"]["conditions"]["daily_warn"].update(directive="ALL_OPS"), "not allowed"),
    (lambda p: p["policy"]["conditions"].pop("max_loss"), "max_loss: missing"),
    (lambda p: p["policy"].update(calendar=[{"id": "w", "kind": "weekly", "tz": "America/New_York", "start": "Fri 16:30",
                                             "end": "Sun 17:05", "directive": "ALL_OPS"}]), "not allowed"),
    (lambda p: p["policy"].update(calendar=[{"id": "w", "kind": "weekly", "tz": "America/New_York", "start": "Friday",
                                             "end": "Sun 17:05", "directive": "NO_OPS"}]), "<Day HH:MM>"),
    (lambda p: p["policy"].update(calendar=[{"id": "d", "kind": "dated", "start_utc": "2026-10-13T12:45:00Z",
                                             "end_utc": "2026-10-13T12:25:00Z", "directive": "NO_OPS"}]), "before end_utc"),
    (lambda p: p["policy"].update(max_loss_mode="trailing_drawdown"), "unknown value"),
    (lambda p: p["policy"].update(daily_loss_pc=5), "unknown key"),
    (lambda p: p.update(max_override_s=10), "max_override_s"),
])
def test_prop_guard_refuses_what_it_may_not_take(mutate, needle):
    entry = copy.deepcopy(prop_guard_entry())
    mutate(entry)
    with pytest.raises(AdminConfigError) as err:
        parse_admin_config(admin_config([entry]))
    assert any(needle in m for m in err.value.problems), err.value.problems


def test_target_condition_may_be_omitted_without_a_target():
    entry = prop_guard_entry(profit_target_pct=None)
    del entry["policy"]["conditions"]["target"]
    assert parse_admin_config(admin_config([entry])).tasks
