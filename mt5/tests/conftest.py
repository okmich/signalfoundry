"""Shared MT5-test fixtures.

``BaseStrategy.__init__`` builds a fail-closed default inference logger (LOGGING_CONTRACT §5) whose
root is resolved from ``OKMICH_QUANT_LOG_BASE`` — with no hardcoded production fallback (OPS §7), so
a test that constructs a real strategy must supply it. Redirect it to a per-test temp dir so building
an MT5 strategy never writes to (or requires) the operator's production ops log root.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ops_log_base(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("OKMICH_QUANT_LOG_BASE", str(tmp_path_factory.mktemp("quant_logs")))
