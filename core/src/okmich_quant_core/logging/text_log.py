"""Per-process text-log setup for runner scripts (``systems/run_*.py``).

The structured channels (inference JSONL, ``status.json``) are framework-owned and resolve
``OKMICH_QUANT_LOG_BASE``. The human-readable stdlib *text* log a runner emits (debug output, crash
forensics) had no shared home — each runner hand-rolled it next to its ``config.json``, which on a
live box is the read-only *artefact* tree (``LIVE_BASE``), not a log root. This module gives the text
log one resolution rule that mirrors the structured layout:

* ``OKMICH_QUANT_LOG_BASE`` set  -> ``<log_base>/<runner_strategy_root>/<prefix>_<ts>.log``
  (beside ``status.json``; one text log per process, single or multi);
* unset                          -> the config directory (dev / standalone fallback).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .identity import _path_safe, runner_strategy_root


def _runner_strategy_root_from_config(config_path: Path) -> str:
    """Derive the runner-root strategy folder from a system ``config.json``: a non-empty
    ``strategies[]`` marks a multi-trader (statutory ``-multi`` suffix); otherwise the singular
    ``strategy.name``. Mirrors the Supervisor's discovery classification so live and log roots agree."""
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    strategies = data.get("strategies") or []
    if strategies:
        return runner_strategy_root(str(strategies[0].get("name", "")), multi=True)
    strategy = data.get("strategy") or {}
    return runner_strategy_root(str(strategy.get("name", "")), multi=False)


def text_log_dir(config_path: str | Path) -> Path:
    """The directory this runner's text log belongs in: ``<OKMICH_QUANT_LOG_BASE>/<runner_strategy_root>``
    when the env root is set, else the config directory (dev / standalone fallback)."""
    cp = Path(config_path)
    raw = os.environ.get("OKMICH_QUANT_LOG_BASE")
    if raw and raw.strip():
        base = Path(os.path.expanduser(os.path.expandvars(raw.strip())))
        return base / _path_safe(_runner_strategy_root_from_config(cp))
    return cp.parent


def setup_text_logger(config_path: str | Path, *, level: int = logging.INFO,
                      prefix: str = "z_system_log") -> Path:
    """Configure the stdlib root logger with a timestamped file handler (+ stdout) under
    :func:`text_log_dir`. Returns the log file path. Drop-in replacement for the per-runner
    ``_build_logger`` boilerplate."""
    log_dir = text_log_dir(config_path)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_{datetime.now().strftime('%y%m%d%H%M%S')}.log"

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(fh)
    root.addHandler(sh)
    return log_file
