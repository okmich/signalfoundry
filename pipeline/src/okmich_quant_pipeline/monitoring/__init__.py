"""Streaming drift-monitor pipeline.

Cron-driven orchestration that reads inference JSONL logs written by live
HMM traders, loads the cached OOS calibration baselines from each deployed
model's ``metadata.json``, re-derives the feature baseline from the OOS
window, and evaluates the three streaming gates from
``okmich_quant_ml.posterior_inference``. Persists per-symbol violation
counter state and dispatches alerts via :class:`okmich_quant_core.BaseNotifier`
when consecutive failures exceed a configured threshold.

Entry point: ``monitor-streaming-gates`` console script (see pipeline/pyproject.toml).
"""
from .config import MonitorConfig
from .counter_state import ViolationCounterState
from .run_monitor import run_monitor_for_symbol

__all__ = ["MonitorConfig", "ViolationCounterState", "run_monitor_for_symbol"]
