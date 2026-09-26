"""``prop_guard`` (ACCOUNT_ADMIN_SPEC §6, §8.2, §11): arithmetic against hand calculations and multi-day replays (§14.7),
requests (§14.9), obedience and orphans (§14.11)."""
from __future__ import annotations

from datetime import timedelta

import pytest

from okmich_quant_core.account_admin import (AccountDirective, AdminEvent, AdminRequest, DailyBase, DealEntry, DealKind,
                                             LimitReference, MaxLossMode, PropGuardTask, RequestOutcome)
from okmich_quant_core.account_admin.owners import MagicOwner
from okmich_quant_core.account_admin.tasks.prop_guard_policy import compute_levels, day_boundary, parse_prop_guard

from .admin_fakes import FakeAccount, make_ctx, order, position, prop_guard_entry, t


def policy(**overrides):
    parsed, problems = parse_prop_guard(prop_guard_entry(**overrides), "t")
    assert not problems, problems
    return parsed


class Run:
    """Drive a prop_guard task cycle by cycle, carrying its slot like the host does."""

    def __init__(self, **overrides):
        self.task = PropGuardTask(prop_guard_entry(**overrides))
        self.acct = FakeAccount()
        self.slot: dict = {}
        self.last = None

    def at(self, when, equity=None, balance=None, **kw):
        if balance is not None:
            self.acct.balance = balance
        if equity is not None:
            self.acct.equity = equity
        self.last = self.task.on_cycle(make_ctx(self.acct, when, self.slot, **kw))
        self.slot = self.last.slot
        return self.last

    @property
    def directive(self):
        return AccountDirective(self.last.output["directive"])

    @property
    def causes(self):
        return self.last.output["causes"]

    def events(self, kind):
        return [e for e in self.last.events if e.event is kind]


# ------------------------------------------------------------------------------------------------ arithmetic


@pytest.mark.parametrize("base_mode", list(DailyBase))
@pytest.mark.parametrize("reference", list(LimitReference))
@pytest.mark.parametrize("mode", list(MaxLossMode))
def test_levels_match_hand_calculation(base_mode, reference, mode):
    p = policy(daily_base=base_mode.value, daily_limit_reference=reference.value, max_loss_mode=mode.value)
    daily_base, balance_high, equity_high = 102_000.0, 104_000.0, 106_000.0
    lv = compute_levels(p, daily_base, balance_high, equity_high)
    l_d = 0.05 * (100_000.0 if reference is LimitReference.INITIAL_CAPITAL else 102_000.0)
    assert lv.daily_floor == pytest.approx(102_000.0 - l_d)
    assert lv.daily_warn == pytest.approx(102_000.0 - l_d + 0.4 * l_d)
    anchor = {MaxLossMode.STATIC: 100_000.0, MaxLossMode.TRAILING_BALANCE: 104_000.0,
              MaxLossMode.TRAILING_EQUITY: 106_000.0}[mode]
    assert lv.max_floor == pytest.approx(anchor - 10_000.0)
    assert lv.max_warn == pytest.approx(anchor - 10_000.0 + 0.2 * 10_000.0)
    assert lv.target == pytest.approx(110_000.0)


def test_trailing_floor_locks_at_initial_capital():
    p = policy(max_loss_mode="trailing_equity", max_loss_trail_locks_at_initial=True)
    assert compute_levels(p, 100_000.0, 100_000.0, 105_000.0).max_floor == pytest.approx(95_000.0)
    assert compute_levels(p, 100_000.0, 100_000.0, 115_000.0).max_floor == pytest.approx(100_000.0)   # locked at C
    assert compute_levels(p, 100_000.0, 100_000.0, 99_000.0).max_floor == pytest.approx(90_000.0)    # anchor >= C


def test_day_boundary_follows_the_zone_through_dst():
    p = policy()   # 17:00 America/New_York
    assert day_boundary(p, t("2026-09-24 20:59:59")) == t("2026-09-23 21:00")   # EDT: 17:00 = 21:00 UTC
    assert day_boundary(p, t("2026-09-24 21:00:00")) == t("2026-09-24 21:00")
    assert day_boundary(p, t("2026-11-03 21:30:00")) == t("2026-11-02 22:00")   # EST after 1 Nov: 17:00 = 22:00 UTC


# ------------------------------------------------------------------------------------------------ replay


def test_daily_warn_loss_latch_and_day_roll():
    """One account through two trading days (boundary 21:00 UTC). C 100k; day 1: F_d 95k, W_d 97k; F_m 90k."""
    r = Run()
    r.at(t("2026-09-24 13:00"), equity=100_000)
    assert r.directive is AccountDirective.ALL_OPS and r.last.output["episode"] == 1
    assert r.last.output["levels"]["daily_floor"] == 95_000.0 and r.last.output["levels"]["daily_warn"] == 97_000.0

    r.at(t("2026-09-24 13:01"), equity=96_900)                      # warn (latch none)
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["daily_warn"]
    assert r.events(AdminEvent.DIRECTIVE_CHANGED) and r.last.output["episode"] == 2
    r.at(t("2026-09-24 13:02"), equity=97_500)                      # recovers: nothing latched
    assert r.directive is AccountDirective.ALL_OPS

    r.at(t("2026-09-24 14:00"), equity=94_900)                      # loss: NO_OPS, latched for the trading day
    assert r.directive is AccountDirective.NO_OPS and r.causes[0] == "daily_loss"
    assert r.events(AdminEvent.LATCH_TRIPPED)
    r.at(t("2026-09-24 15:00"), equity=98_000)                      # recovered, still latched
    assert r.directive is AccountDirective.NO_OPS and r.causes == ["daily_loss"]

    r.acct.add_deal(900, t("2026-09-24 20:00"), net=-2_000)         # day 1 closed at 98k balance
    r.at(t("2026-09-24 20:59:50"), equity=98_000, balance=98_000)
    r.at(t("2026-09-24 21:00:15"), equity=98_000, balance=98_000)   # boundary: latch clears, base 98k
    assert r.directive is AccountDirective.ALL_OPS
    assert r.last.output["trading_day"] == {"start_utc": "2026-09-24T21:00:00Z", "base": 98_000.0, "base_estimated": False}
    assert r.last.output["levels"]["daily_floor"] == 93_000.0
    assert [e.fields["condition"] for e in r.events(AdminEvent.LATCH_CLEARED)] == ["daily_loss"]


def test_manual_latch_survives_the_day_and_a_reset_retrips_while_it_holds():
    r = Run()
    r.at(t("2026-09-24 13:00"), equity=100_000)
    r.at(t("2026-09-24 13:01"), equity=89_500)                       # max_loss (manual)
    assert r.directive is AccountDirective.NO_OPS and "max_loss" in r.causes
    r.at(t("2026-09-25 13:00"), equity=99_000, balance=99_000)       # next day, recovered: still latched
    assert r.directive is AccountDirective.NO_OPS and r.causes == ["max_loss"]

    r.acct.equity = 89_000                                            # breached again: a reset cannot hold it open
    req = AdminRequest("r1", t("2026-09-25 13:05"), "ops", "prop_guard", "reset_latch", {"target": "max_loss"})
    r.at(t("2026-09-25 13:05"), requests=[req])
    assert r.last.request_outcomes["r1"][0] is RequestOutcome.APPLIED
    assert r.directive is AccountDirective.NO_OPS and "max_loss" in r.slot["latches"]
    assert "daily_loss" in r.slot["latches"]                          # 89k is also below today's 94k floor

    req = AdminRequest("r2", t("2026-09-25 13:06"), "ops", "prop_guard", "reset_latch", {"target": "max_loss"})
    r.at(t("2026-09-25 13:06"), equity=99_000, requests=[req])
    assert "max_loss" not in r.slot["latches"]
    assert r.directive is AccountDirective.NO_OPS and r.causes == ["daily_loss"]   # today's latch still stands
    req = AdminRequest("r3", t("2026-09-25 13:07"), "ops", "prop_guard", "reset_latch", {"target": "all"})
    r.at(t("2026-09-25 13:07"), equity=99_000, requests=[req])
    assert r.directive is AccountDirective.ALL_OPS and not r.slot["latches"]


def test_day_start_max_balance_equity_base_is_estimated_when_the_admin_missed_the_boundary():
    r = Run(daily_base="day_start_max_balance_equity")
    r.at(t("2026-09-24 22:00"), equity=101_000, balance=100_000)     # first ever cycle, after the boundary
    day = r.last.output["trading_day"]
    assert day["base"] == 100_000.0 and day["base_estimated"] is True
    assert any("ESTIMATED" in a.title for a in r.last.alerts)

    r.at(t("2026-09-25 20:59:55"), equity=101_500, balance=100_000)  # running across the next boundary
    r.at(t("2026-09-25 21:00:20"), equity=101_500, balance=100_000)
    day = r.last.output["trading_day"]
    assert day["base"] == 101_500.0 and day["base_estimated"] is False


def test_trailing_balance_anchor_rebuilt_from_history_and_advanced_by_new_deals():
    r = Run(max_loss_mode="trailing_balance")
    r.acct.add_deal(1, t("2026-09-10 10:00"), net=+6_000)            # balance peaked at 106k ...
    r.acct.add_deal(2, t("2026-09-12 10:00"), net=-4_000)            # ... now 102k
    r.at(t("2026-09-24 13:00"), equity=102_000, balance=102_000)
    assert r.last.output["levels"]["max_floor"] == 96_000.0          # 106k - 10k
    r.acct.add_deal(3, t("2026-09-24 13:00:30"), net=+5_000)
    r.acct.add_deal(4, t("2026-09-24 13:00:40"), net=-1_000)          # 107k then 106k between two cycles
    r.at(t("2026-09-24 13:01"), equity=106_000, balance=106_000)
    assert r.last.output["levels"]["max_floor"] == 97_000.0          # the 107k peak between cycles counts


def test_target_on_balance_latches_no_ops():
    r = Run()
    r.at(t("2026-09-24 13:00"), equity=111_000, balance=109_000)
    assert r.directive is AccountDirective.ALL_OPS                   # measured on balance
    r.at(t("2026-09-24 13:01"), equity=111_000, balance=110_000)
    assert r.directive is AccountDirective.NO_OPS and r.causes == ["target"]


def test_calendar_windows_weekly_wrap_and_dated():
    cal = [{"id": "weekend", "kind": "weekly", "tz": "America/New_York", "start": "Fri 16:30", "end": "Sun 17:05",
            "directive": "NO_OPS"},
           {"id": "cpi", "kind": "dated", "start_utc": "2026-10-13T12:25:00Z", "end_utc": "2026-10-13T12:45:00Z",
            "directive": "NO_ENTRY_OPS"}]
    r = Run(calendar=cal)
    r.at(t("2026-09-25 20:29"))                                      # Fri 16:29 EDT
    assert r.directive is AccountDirective.ALL_OPS
    r.at(t("2026-09-25 20:30"))                                      # Fri 16:30 EDT
    assert r.directive is AccountDirective.NO_OPS and r.causes == ["calendar:weekend"]
    r.at(t("2026-09-26 12:00"))                                      # Saturday
    assert r.directive is AccountDirective.NO_OPS
    r.at(t("2026-09-27 21:05"))                                      # Sun 17:05 EDT: end is exclusive
    assert r.directive is AccountDirective.ALL_OPS
    r.at(t("2026-10-13 12:30"))
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["calendar:cpi"]
    r.at(t("2026-10-13 12:45"))
    assert r.directive is AccountDirective.ALL_OPS


def test_override_only_tightens_is_bounded_and_lapses():
    r = Run()
    now = t("2026-09-24 13:00")
    reqs = [AdminRequest("a", now, "ops", "prop_guard", "override", {"directive": "ALL_OPS", "until_utc": "2026-09-24T15:00:00Z"}),
            AdminRequest("b", now, "ops", "prop_guard", "override", {"directive": "NO_OPS", "until_utc": "2026-09-27T15:00:00Z"}),
            AdminRequest("c", now, "ops", "prop_guard", "override", {"directive": "NO_OPS"}),
            AdminRequest("d", now, "ops", "prop_guard", "override", {"directive": "NO_ENTRY_OPS",
                                                                     "until_utc": "2026-09-24T13:10:00Z", "reason": "FOMC"})]
    r.at(now, requests=reqs)
    outcomes = {k: v[0] for k, v in r.last.request_outcomes.items()}
    assert outcomes == {"a": RequestOutcome.REJECTED, "b": RequestOutcome.REJECTED, "c": RequestOutcome.REJECTED,
                        "d": RequestOutcome.APPLIED}
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["override"]
    r.at(t("2026-09-24 13:09:59"))
    assert r.directive is AccountDirective.NO_ENTRY_OPS
    r.at(t("2026-09-24 13:10:00"))
    assert r.directive is AccountDirective.ALL_OPS and r.events(AdminEvent.OVERRIDE_EXPIRED)


def test_override_never_loosens_a_breached_limit():
    r = Run()
    r.at(t("2026-09-24 13:00"), equity=94_000)
    req = AdminRequest("o", t("2026-09-24 13:01"), "ops", "prop_guard", "override",
                       {"directive": "NO_ENTRY_OPS", "until_utc": "2026-09-24T14:00:00Z"})
    r.at(t("2026-09-24 13:01"), requests=[req])
    assert r.directive is AccountDirective.NO_OPS and r.causes[0] == "daily_loss" and "override" in r.causes


def test_malformed_and_unknown_requests_are_rejected():
    r = Run()
    reqs = [AdminRequest("x", t("2026-09-24 13:00"), "ops", "prop_guard", "flatten_everything", {}),
            AdminRequest("y", t("2026-09-24 13:00"), "ops", "prop_guard", "override",
                         {"directive": "NO_OPS", "until_utc": "tomorrow"}),
            AdminRequest("z", t("2026-09-24 13:00"), "ops", "prop_guard", "reset_latch", {"target": "daily_loss"})]
    r.at(t("2026-09-24 13:00"), requests=reqs)
    assert all(v[0] is RequestOutcome.REJECTED for v in r.last.request_outcomes.values())


def test_balance_operation_after_start_trips_a_manual_latch():
    r = Run()
    r.acct.add_deal(1, t("2026-08-31 09:00"), net=100_000, kind=DealKind.BALANCE, entry=DealEntry.NONE)   # before start
    r.at(t("2026-09-24 13:00"))
    assert r.directive is AccountDirective.ALL_OPS
    r.acct.add_deal(2, t("2026-09-24 13:00:30"), net=5_000, kind=DealKind.BALANCE, entry=DealEntry.NONE)
    r.at(t("2026-09-24 13:01"), balance=105_000, equity=105_000)
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["balance_operation"]


def test_state_lost_when_a_directive_existed():
    r = Run()
    prev = {"sequence": 500, "episode": 40, "directive": "ALL_OPS"}
    r.at(t("2026-09-24 13:00"), first_cycle=True, state_load="missing", previous_output=prev)
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["state_lost"]
    assert r.last.output["sequence"] == 501 and r.last.output["episode"] == 41


def test_first_ever_start_is_not_state_lost():
    r = Run()
    r.at(t("2026-09-24 13:00"), first_cycle=True, state_load="missing", previous_output=None)
    assert r.directive is AccountDirective.ALL_OPS and r.last.output["sequence"] == 1


def test_degraded_cycle_publishes_no_entry_and_alerts_entering_and_leaving():
    r = Run()
    r.at(t("2026-09-24 13:00"))
    r.at(t("2026-09-24 13:01"), degraded=True)
    assert r.directive is AccountDirective.NO_ENTRY_OPS and r.causes == ["admin_degraded"]
    assert r.last.output["metrics"]["equity"] == 100_000.0           # last known evidence kept
    assert r.events(AdminEvent.ADMIN_DEGRADED)[0].fields["state"] == "entering"
    r.at(t("2026-09-24 13:02"))
    assert r.directive is AccountDirective.ALL_OPS
    assert r.events(AdminEvent.ADMIN_DEGRADED)[0].fields["state"] == "leaving"


def test_obedience_breach_after_grace_only():
    r = Run()
    r.at(t("2026-09-24 13:00"))
    r.at(t("2026-09-24 13:01"), equity=96_900)                       # NO_ENTRY_OPS since 13:01:00, grace 30 s
    r.acct.add_deal(10, t("2026-09-24 13:01:20"), kind=DealKind.BUY, entry=DealEntry.IN, magic=0)    # within grace
    r.acct.add_deal(11, t("2026-09-24 13:01:40"), kind=DealKind.BUY, entry=DealEntry.IN, magic=42)   # breach
    r.acct.add_deal(12, t("2026-09-24 13:01:45"), kind=DealKind.SELL, entry=DealEntry.OUT, magic=42) # an exit is fine
    r.acct.orders = [order(20, t("2026-09-24 13:01:50"), magic=0)]                                    # new pending: breach
    owners = {42: MagicOwner(42, "icmarkets.demo/propfolio_breakout-multi", "propfolio_breakout-multi", True)}
    r.at(t("2026-09-24 13:02"), equity=96_900, owners=owners)
    breaches = {e.fields["ticket"]: e.fields for e in r.events(AdminEvent.OBEDIENCE_BREACH)}
    assert set(breaches) == {11, 20}
    assert breaches[11]["owner"] == "icmarkets.demo/propfolio_breakout-multi" and breaches[20]["owner"] == "manual"
    r.at(t("2026-09-24 13:03"), equity=96_900, owners=owners)
    assert not r.events(AdminEvent.OBEDIENCE_BREACH)                 # each reported once per episode


def test_orphans_under_no_ops_after_close_grace_and_episode_summary():
    r = Run()
    r.at(t("2026-09-24 13:00"))
    r.acct.open_positions = [position(1, magic=42)]
    r.at(t("2026-09-24 13:01"), equity=94_000)                       # NO_OPS since 13:01, close grace 120 s
    assert not r.events(AdminEvent.ORPHAN)
    r.at(t("2026-09-24 13:03:01"), equity=94_000)
    assert [e.fields["ticket"] for e in r.events(AdminEvent.ORPHAN)] == [1]
    assert r.events(AdminEvent.ORPHAN)[0].fields["owner"] == "not this account's"
    r.at(t("2026-09-24 13:04"), equity=94_000)
    assert not r.events(AdminEvent.ORPHAN)
    r.acct.open_positions = []
    r.at(t("2026-09-25 21:00:30"), equity=99_000, balance=99_000)    # next day: episode ends
    assert any(a.title == "NO_OPS EPISODE ENDED" for a in r.last.alerts)


def test_sequence_increments_every_cycle_and_episode_only_on_change():
    r = Run()
    for i in range(3):
        r.at(t("2026-09-24 13:00") + timedelta(minutes=i))
    assert (r.last.output["sequence"], r.last.output["episode"]) == (3, 1)
