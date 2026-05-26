"""Per-symbol violation-counter state, JSON-persisted across monitor invocations.

Each monitor cycle either passes (resets counters) or fails (increments per-gate
counters). When any counter reaches ``threshold`` consecutive failing cycles, the
gate is considered "tripped" and an alert is dispatched. The tripped state is also
recorded so the alert fires once per breach (re-arming only after the next pass).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from okmich_quant_ml.posterior_inference import MonitoringCycleReport


@dataclass
class ViolationCounterState:
    """Mutable per-symbol counter state.

    Loaded from ``<output_dir>/<symbol>/counter_state.json`` (or initialised to
    zeros on first run). ``advance(report, threshold)`` updates counters and
    returns the list of gate names whose counter newly crossed the threshold
    this cycle — these are the alerts the orchestrator should dispatch.
    """
    posterior_consecutive_failures: int = 0
    feature_consecutive_failures: int = 0
    loglik_consecutive_failures: int = 0
    posterior_alert_armed: bool = True
    feature_alert_armed: bool = True
    loglik_alert_armed: bool = True
    last_cycle_utc: str | None = None
    last_overall_ok: bool | None = None

    @classmethod
    def load_or_init(cls, path: str | Path) -> ViolationCounterState:
        p = Path(path)
        if not p.is_file():
            return cls()
        payload = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: payload[k] for k in payload if k in cls.__dataclass_fields__})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def advance(self, report: MonitoringCycleReport, threshold: int) -> list[str]:
        """Update counters from ``report``, return list of newly-tripped gate names.

        A gate trips when its counter *reaches* ``threshold`` (>=) for the first
        time after being armed. After tripping, the alert is disarmed; it re-arms
        when the gate passes again. This ensures one alert per breach episode
        rather than one per failing cycle.
        """
        if threshold < 1:
            raise ValueError(f"ViolationCounterState.advance: threshold must be >= 1, got {threshold}")
        newly_tripped: list[str] = []

        for gate_name, ok, counter_attr, armed_attr in (
            ("posterior", report.posterior.overall_ok, "posterior_consecutive_failures", "posterior_alert_armed"),
            ("feature", report.feature.overall_ok, "feature_consecutive_failures", "feature_alert_armed"),
            ("loglik", report.loglik.overall_ok, "loglik_consecutive_failures", "loglik_alert_armed"),
        ):
            if ok:
                setattr(self, counter_attr, 0)
                setattr(self, armed_attr, True)  # re-arm after recovery
            else:
                new_count = getattr(self, counter_attr) + 1
                setattr(self, counter_attr, new_count)
                if new_count >= threshold and getattr(self, armed_attr):
                    newly_tripped.append(gate_name)
                    setattr(self, armed_attr, False)

        self.last_cycle_utc = str(report.log_window_end_ts)
        self.last_overall_ok = bool(report.overall_ok)
        return newly_tripped

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
