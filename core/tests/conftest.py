"""Shared core-test fixtures.

The fail-closed default inference logger (LOGGING_CONTRACT §5) is constructed in
``BaseStrategy.__init__`` and writes under ``OKMICH_QUANT_LOG_BASE``. This autouse fixture
redirects that base to a per-test temp dir so any test that builds a real strategy never writes
to the operator's configured production ops log root.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ops_log_base(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path_factory.mktemp("quant_logs")))
