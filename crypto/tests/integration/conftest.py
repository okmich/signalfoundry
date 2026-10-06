"""Integration tests run against a REAL venue copy (testnet or demo) and are skipped unless configured:

    CRYPTO_IT_EXCHANGE=bybit      # or binance
    CRYPTO_IT_ENV=demo            # testnet: bybit only. LIVE is refused
    CRYPTO_IT_API_KEY=...
    CRYPTO_IT_API_SECRET=...

They place real (paper) orders: a far-from-market limit on spot and perp, and a minimum-size perp round trip.
"""
import os

import pytest
import pytest_asyncio

from okmich_quant_crypto import VenueEnvironment
from okmich_quant_crypto.config import CryptoVenueConfig
from okmich_quant_crypto.functions.crypto import connect_exchange
from okmich_quant_crypto.models import Credentials
from okmich_quant_crypto.venue.registry import resolve_profile


def _settings():
    exchange_id = os.environ.get("CRYPTO_IT_EXCHANGE")
    env = os.environ.get("CRYPTO_IT_ENV")
    key, secret = os.environ.get("CRYPTO_IT_API_KEY"), os.environ.get("CRYPTO_IT_API_SECRET")
    if not (exchange_id and env and key and secret):
        pytest.skip("set CRYPTO_IT_EXCHANGE, CRYPTO_IT_ENV, CRYPTO_IT_API_KEY, CRYPTO_IT_API_SECRET to run")
    environment = VenueEnvironment(env.lower())
    if environment is VenueEnvironment.LIVE:
        pytest.fail("integration tests never run against LIVE")
    return exchange_id, environment, Credentials(api_key=key, secret=secret)


@pytest.fixture
def it_settings(tmp_path):
    exchange_id, environment, creds = _settings()
    venue = CryptoVenueConfig(exchange_id=exchange_id, environment=environment, state_dir=str(tmp_path / "state"))
    return venue, creds


@pytest_asyncio.fixture
async def it_exchange(it_settings):
    venue, creds = it_settings
    profile = resolve_profile(venue.exchange_id)
    exchange = await connect_exchange(profile, venue, creds)
    yield exchange, profile, venue
    await exchange.close()
