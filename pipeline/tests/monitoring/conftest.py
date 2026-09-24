"""Shared monitoring-test fixtures.

The inference logger resolves ``<log_base>/<account>`` and takes the account from ``OKMICH_QUANT_ACCOUNT``
only (LOGGING_CONTRACT §10), so tests that write logs through the real ``JsonlEventLogger`` must set it.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ops_account(monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_ACCOUNT", "test.demo")
