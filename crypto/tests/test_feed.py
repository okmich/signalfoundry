"""Closed-bar feeds: REST reconciliation, exactly-once delivery, gap backfill, staleness, and STREAM == POLL."""
import pytest

from okmich_quant_core import BarOutcome, LogEventType
from okmich_quant_crypto.feed import BarReconciler, BarSequencer, PollBarSource, StreamBarSource
from okmich_quant_crypto.strategy import bars_frame

from .conftest import make_cfg, make_strategy
from .fakes import MIN, T0, FakeClock, FakeExchange

SYM, TFS, TF = "BTC/USDT:USDT", "5m", 5 * MIN


def _candle(open_ms, i, final=True):
    p = 100.0 + i
    # A non-final (stale) update differs from the final candle in close and volume.
    return [open_ms, p, p + 3, p - 3, p + (2 if final else 0.5), 50.0 + i if final else 1.0]


def _market(clock):
    """30 closed candles before the clock's current bar, plus the forming one."""
    ex = FakeExchange(clock)
    start = (clock() // TF) * TF - 30 * TF
    ex.add_candles(SYM, TFS, [_candle(start + i * TF, i) for i in range(30)])
    ex.set_forming(SYM, TFS, _candle(start + 30 * TF, 30, final=False))
    return ex, start


async def test_reconciler_never_takes_the_forming_candle(clock):
    ex, start = _market(clock)
    rec = BarReconciler(ex, SYM, TFS, ohlcv_limit=3, clock=clock)
    forming_open = start + 30 * TF
    assert await rec.fetch_closed(forming_open) is None              # its close is in the future
    assert (await rec.fetch_closed(forming_open - TF)).close == pytest.approx(_candle(0, 29)[4])
    assert await rec.fetch_closed(start - 5 * TF) is None             # venue has no such candle


async def test_reconciler_pages_with_explicit_limit(clock):
    ex, start = _market(clock)
    rec = BarReconciler(ex, SYM, TFS, ohlcv_limit=3, clock=clock)
    bars = await rec.fetch_recent(10)
    assert [b.open_ms for b in bars] == [start + i * TF for i in range(20, 30)]
    assert all(req[3] == 3 for req in ex.ohlcv_requests)               # the page size is always explicit


async def test_sequencer_exactly_once_gap_backfill_and_staleness(clock):
    ex, start = _market(clock)
    got = []

    async def on_bar(bar, live):
        got.append((bar.open_ms, live))

    seq = BarSequencer(BarReconciler(ex, SYM, TFS, 3, clock), on_bar, max_wait_ms=30_000, clock=clock)
    seq.last_open_ms = start + 25 * TF
    clock.now_ms = start + 30 * TF + 1_000                               # 1s after bar 29 closed
    assert await seq.offer(start + 29 * TF) is True
    assert got == [(start + 26 * TF, False), (start + 27 * TF, False), (start + 28 * TF, False),
                   (start + 29 * TF, True)]
    assert await seq.offer(start + 29 * TF) is True                      # duplicate: no second delivery
    assert await seq.offer(start + 27 * TF) is True                      # older: ignored
    assert len(got) == 4


async def test_late_bar_is_stale(clock):
    ex, start = _market(clock)
    got = []

    async def on_bar(bar, live):
        got.append(live)

    seq = BarSequencer(BarReconciler(ex, SYM, TFS, 3, clock), on_bar, max_wait_ms=30_000, clock=clock)
    clock.now_ms = start + 30 * TF + 31_000
    await seq.offer(start + 29 * TF)
    assert got == [False]


async def test_missed_bar_is_reported_then_backfilled(clock):
    ex, start = _market(clock)
    ex.set_forming(SYM, TFS, None)                      # the venue never publishes bar 30 in time
    missed, got = [], []
    src = PollBarSource(ex, SYM, TFS, _sink(got), ohlcv_limit=3, close_grace_seconds=3, close_max_wait_seconds=10,
                        clock=clock, sleep=clock.sleep, on_missed=missed.append)
    src.sequencer.last_open_ms = start + 29 * TF
    clock.now_ms = start + 31 * TF + 3_000
    assert await src.poll_once() is False
    assert missed == [start + 30 * TF] and got == []
    # Next boundary: both candles are there now. Bar 30 is backfilled buffer-only, bar 31 runs live.
    ex.add_candles(SYM, TFS, [_candle(start + 30 * TF, 30), _candle(start + 31 * TF, 31)])
    clock.now_ms = start + 32 * TF + 3_000
    assert await src.poll_once() is True
    assert got == [(start + 30 * TF, False), (start + 31 * TF, True)]


def _sink(store):
    async def on_bar(bar, live):
        store.append((bar.open_ms, live))
    return on_bar


async def test_watchdog_closes_a_bar_the_socket_never_did(clock):
    ex, start = _market(clock)
    got = []
    src = StreamBarSource(ex, SYM, TFS, _sink(got), ohlcv_limit=3, close_grace_seconds=3, close_max_wait_seconds=30,
                          clock=clock, sleep=clock.sleep)
    src.sequencer.last_open_ms = start + 28 * TF
    clock.now_ms = start + 30 * TF + 15_000
    await src.watchdog_tick()
    assert got == [(start + 29 * TF, True)]


# ---------------------------------------------------------------------- STREAM and POLL are indistinguishable

def _scripted_run_rows(start):
    """For bars 30..34: WS delivers a STALE update of the bar (its final update is 'missed'), then the next bar."""
    steps = []
    for i in range(30, 35):
        steps.append((start + i * TF, [_candle(start + i * TF, i, final=False)],
                      [_candle(start + (i + 1) * TF, i + 1, final=False)]))
    return steps


async def _run(mode: str):
    clock = FakeClock(T0 + 40 * TF + 7_000)
    clock.now_ms = (clock() // TF) * TF + 7_000
    ex, start = _market(clock)
    strategy, rec = make_strategy(make_cfg(bars_to_copy=10))
    strategy._clock = clock
    cls = StreamBarSource if mode == "stream" else PollBarSource
    src = cls(ex, SYM, TFS, strategy._on_bar_close, ohlcv_limit=3, close_grace_seconds=3, close_max_wait_seconds=30,
              clock=clock, sleep=clock.sleep)
    seed = await src.seed(11)
    strategy.price_buffer.update(bars_frame(seed), strategy._now_dt())
    for open_ms, during, rollover in _scripted_run_rows(start):
        i = (open_ms - start) // TF
        # The venue publishes the bar's FINAL candle at its close; the next one starts forming.
        ex.add_candles(SYM, TFS, [_candle(open_ms, i)])
        ex.set_forming(SYM, TFS, rollover[0])
        if mode == "stream":
            clock.now_ms = open_ms + TF - 2_000
            await src.handle_rows(during)          # still forming: nothing closes
            clock.now_ms = open_ms + TF + 1_500
            await src.handle_rows(rollover)        # rollover -> REST-confirmed close
        else:
            clock.now_ms = open_ms + TF + 3_000    # boundary + close_grace
            await src.poll_once()
    records = [(r.asof_bar_ts, r.outcome, getattr(r, "features", None), getattr(r, "bar_close", None))
               for r in rec.records if r.envelope.event is LogEventType.BAR]
    return strategy.cycles, records, strategy.price_buffer.get_data()


async def test_stream_and_poll_produce_identical_bars_and_decisions():
    s_cycles, s_records, s_buffer = await _run("stream")
    p_cycles, p_records, p_buffer = await _run("poll")
    assert len(s_cycles) == 5 and s_cycles == p_cycles
    assert s_records == p_records and all(r[1] is BarOutcome.OK for r in s_records)
    assert s_buffer.equals(p_buffer)
    # The bar the strategy saw is the REST-final candle, not the stale WS update.
    assert s_buffer["volume"].iloc[-1] > 1.0
