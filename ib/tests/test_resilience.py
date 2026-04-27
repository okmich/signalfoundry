"""Unit tests for resilience: error classification and retry decorator."""
import asyncio

import pytest

from okmich_quant_ib.resilience import (
    ErrorClass, IBConnectionError, IBPermanentError, IBTransientError,
    classify_ib_error, with_retry,
)


class TestClassifyIbError:
    def test_transient_codes(self):
        for code in (1100, 1101, 1102, 2110, 10225, 504, 162):
            assert classify_ib_error(code) == ErrorClass.TRANSIENT

    def test_permanent_codes(self):
        for code in (200, 201, 202, 203, 321, 10147, 10148):
            assert classify_ib_error(code) == ErrorClass.PERMANENT

    def test_warning_range(self):
        # 2110 is explicitly TRANSIENT (overrides the 2000-2999 WARNING bucket).
        assert classify_ib_error(2104) == ErrorClass.WARNING
        assert classify_ib_error(2150) == ErrorClass.WARNING

    def test_unknown(self):
        assert classify_ib_error(99999) == ErrorClass.UNKNOWN


@pytest.mark.asyncio
async def test_with_retry_succeeds_after_transient():
    calls = []

    @with_retry(max_retries=3, initial_delay=0.01, backoff_factor=1.0)
    async def flaky():
        calls.append(1)
        if len(calls) < 2:
            raise IBTransientError("temporarily flaky", 162)
        return "ok"

    assert await flaky() == "ok"
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_with_retry_does_not_retry_permanent():
    calls = []

    @with_retry(max_retries=3, initial_delay=0.01)
    async def boom():
        calls.append(1)
        raise IBPermanentError("rejected", 201)

    with pytest.raises(IBPermanentError):
        await boom()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_with_retry_exhausts_and_raises():
    @with_retry(max_retries=2, initial_delay=0.01, backoff_factor=1.0)
    async def always_flaky():
        raise IBTransientError("never recovers", 162)

    with pytest.raises(IBTransientError):
        await always_flaky()


@pytest.mark.asyncio
async def test_with_retry_recovers_from_connection_error():
    calls = []

    @with_retry(max_retries=2, initial_delay=0.01, backoff_factor=1.0)
    async def reconnecting():
        calls.append(1)
        if len(calls) == 1:
            raise IBConnectionError("dropped")
        return "back"

    assert await reconnecting() == "back"
    assert len(calls) == 2
