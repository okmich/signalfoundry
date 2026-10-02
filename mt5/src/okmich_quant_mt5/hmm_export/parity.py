"""Bind the MQL5 implementation to the Python reference on real bars.

Workflow:

1. Attach ``HMM_Filtering`` or ``HMM_CausalViterbi`` to a chart (or run it in the
   Strategy Tester) with ``InpDumpCsv = true``. It writes
   ``MQL5/Files/<InpDumpFile>``.
2. Pull the same symbol/timeframe bars into Python.
3. Call :func:`check_parity`.

What this catches that nothing else does: a feature computed slightly
differently on the two sides. That failure never raises - it shifts the
observation distribution, the fitted emissions are evaluated off-distribution,
and the state sequence becomes confident nonsense. A tolerance check on the
feature columns is the only thing standing in front of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features
from .kernel import LABEL_NONE, Mql5HmmKernel, Mql5HmmParams, Mql5InferenceMode


@dataclass
class ParityReport:
    n_compared: int
    per_feature_abs_diff: dict[str, float] = field(default_factory=dict)
    label_mismatches: int = 0
    max_confidence_abs_diff: float = 0.0
    feature_tol: float = 1e-9
    confidence_tol: float = 1e-9

    @property
    def max_feature_abs_diff(self) -> float:
        return max(self.per_feature_abs_diff.values(), default=0.0)

    @property
    def ok(self) -> bool:
        return (
            self.n_compared > 0
            and self.max_feature_abs_diff <= self.feature_tol
            and self.label_mismatches == 0
            and self.max_confidence_abs_diff <= self.confidence_tol
        )

    def __str__(self) -> str:
        lines = [
            f"parity: {'PASS' if self.ok else 'FAIL'}  ({self.n_compared} bars compared)",
            f"  labels mismatched      : {self.label_mismatches}",
            f"  max |d confidence|     : {self.max_confidence_abs_diff:.3e}  (tol {self.confidence_tol:.1e})",
        ]
        for name, diff in self.per_feature_abs_diff.items():
            lines.append(f"  max |d {name:<14}| : {diff:.3e}  (tol {self.feature_tol:.1e})")
        return "\n".join(lines)


def load_dump(path: str | Path) -> pd.DataFrame:
    """Read the CSV written by ``CHmmIndicator::DumpRow``, indexed by bar time (UTC)."""
    frame = pd.read_csv(path)
    if "time" not in frame.columns:
        raise ValueError(f"{path} has no 'time' column; is this an HMM indicator dump?")

    frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
    return frame.set_index("time").sort_index()


def check_parity(dump_path: str | Path, times, high, low, close, params: Mql5HmmParams,
                 mode: Mql5InferenceMode, *, burn_in: int = 300, feature_tol: float = 1e-9,
                 confidence_tol: float = 1e-9, raise_on_fail: bool = False) -> ParityReport:
    """Compare an MQL5 dump against the Python reference on identical bars.

    ``times`` must be timezone-aware UTC timestamps aligned with the OHLC arrays
    (MT5 bar open times). Only bars present on both sides are compared, so a
    dump covering a subset of the pulled history is fine.
    """
    mode = Mql5InferenceMode(mode)
    dump = load_dump(dump_path)

    features = build_features(high, low, close, params.spec)
    labels, confidence = Mql5HmmKernel(params, mode).run(features, burn_in=burn_in)

    reference = pd.DataFrame(features, columns=list(params.spec.names), index=pd.DatetimeIndex(times, tz="UTC"))
    reference["label"] = labels
    reference["confidence"] = confidence
    reference = reference[reference["label"] != LABEL_NONE]

    joined = dump.join(reference, how="inner", rsuffix="_py")
    if joined.empty:
        raise ValueError(
            "no overlapping bars between the dump and the supplied history. Check that the symbol, "
            "timeframe and time zone match, and that the dump is not stale."
        )

    report = ParityReport(n_compared=len(joined), feature_tol=feature_tol, confidence_tol=confidence_tol)
    for name in params.spec.names:
        diff = np.abs(joined[str(name)].to_numpy() - joined[f"{name}_py"].to_numpy())
        report.per_feature_abs_diff[str(name)] = float(np.nanmax(diff))

    report.label_mismatches = int((joined["label"].to_numpy() != joined["label_py"].to_numpy()).sum())
    conf_diff = np.abs(joined["confidence"].to_numpy() - joined["confidence_py"].to_numpy())
    report.max_confidence_abs_diff = float(np.nanmax(conf_diff))

    if raise_on_fail and not report.ok:
        raise AssertionError(f"MQL5 / Python parity check failed:\n{report}")

    return report
