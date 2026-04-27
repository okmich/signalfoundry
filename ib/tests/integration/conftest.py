"""Integration tests are gated by IB_INTEGRATION=1 (and require a paper Gateway).

All fixtures here are async and skip cleanly if the env var is not set.
"""
import os

import pytest
import pytest_asyncio
from ib_async import IB


@pytest_asyncio.fixture(scope="session")
async def ib_paper():
    if not os.environ.get("IB_INTEGRATION"):
        pytest.skip("set IB_INTEGRATION=1 to run integration tests")
    ib = IB()
    await ib.connectAsync("127.0.0.1", 4002, clientId=999, timeout=10.0)
    yield ib
    ib.disconnect()
