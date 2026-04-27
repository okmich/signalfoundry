"""Paper-account connectivity smoke test. Runs only when IB_INTEGRATION=1."""
import pytest


@pytest.mark.asyncio
async def test_connected_to_paper(ib_paper):
    assert ib_paper.isConnected()
