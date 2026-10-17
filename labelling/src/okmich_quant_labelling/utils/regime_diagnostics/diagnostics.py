"""regime_diagnostics(): the one call. Every pass on one causal regime label, optionally conditioned on other regime
dimensions, delivered as one result object (and one HTML report when ``out_dir`` is given).

    diag = regime_diagnostics(close, label, first_test_start="2024-07-01", condition=vol_state, cost=cost,
                              out_dir="results/diagnostics/EURUSD_path")
    print(diag.story)          # plain-language read, numbers with bands
    diag.expectations          # what to expect next per label state x condition cell
    diag.forward / diag.descriptive_oos / diag.conditional   # the parts
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from okmich_quant_labelling.utils.label_eval_util import label_path_structure_statistics

from .charts import build_charts
from .conditional import ConditionalResult, conditional_analysis
from .core import ForwardSeparabilityResult, forward_separability
from .report import write_report
from .story import tell_story


@dataclass
class RegimeDiagnostics:
    name: str
    forward: ForwardSeparabilityResult
    descriptive_train: pd.DataFrame
    descriptive_oos: pd.DataFrame
    conditional: ConditionalResult | None
    condition_name: str
    expectations: pd.DataFrame  # per label state (x condition cell) x target x layer: mean and quantiles with bands
    story: str
    report_path: Path | None

    def __repr__(self) -> str:
        return f"RegimeDiagnostics({self.name}: {self.forward!r})"


def regime_diagnostics(close: pd.Series, label: pd.Series, *, first_test_start, condition=None, name: str = "label",
                       cost=None, out_dir=None, **forward_kwargs) -> RegimeDiagnostics:
    """Descriptive pass, forward pass, tails, conditioning, expectations, story and report for ``label``.

    ``condition`` is None, a Series, or a {name: Series} of other regime dimensions (each causal, on ``close``'s
    index). Without it, the expectation table is per label state; with it, per label state x condition cell, and
    the within-condition test asks whether the label still separates inside each condition stratum.
    ``forward_kwargs`` go to ``forward_separability`` (horizons, lookbacks, label_fit_end, n_null, ...).
    """
    forward = forward_separability(close, label, first_test_start=first_test_start, cost=cost, **forward_kwargs)
    oos_start = forward.values.index[0]
    frame = pd.DataFrame({"close": close, "returns": np.log(close).diff(), "state": label})
    descriptive_train = _descriptive(frame[frame.index < oos_start])
    descriptive_oos = _descriptive(frame[frame.index >= oos_start])

    n_null = forward_kwargs.get("n_null", 999)
    conditional = conditional_analysis(forward, condition, n_null=n_null) if condition is not None else None
    marginal = conditional if conditional is not None else conditional_analysis(forward, None, within_test=False)
    expectations = marginal.expectations
    condition_name = " & ".join(condition) if isinstance(condition, dict) else getattr(condition, "name", "") or ""

    story = tell_story(name, forward, descriptive_oos, conditional)
    diag = RegimeDiagnostics(name, forward, descriptive_train, descriptive_oos, conditional, condition_name,
                             expectations, story, None)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tables = {"summary": forward.summary, "per_state": forward.per_state, "per_fold": forward.per_fold,
                  "quantiles": forward.quantiles, "tails": forward.tails, "expectations": expectations,
                  "descriptive_train": descriptive_train, "descriptive_oos": descriptive_oos}
        if conditional is not None:
            tables.update({"within_condition": conditional.within,
                           "within_condition_states": conditional.within_states})
        for table_name, table in tables.items():
            table.to_csv(out_dir / f"{table_name}.csv", index=False)
        (out_dir / "story.txt").write_text(story, encoding="utf-8")
        diag.report_path = out_dir / "report.html"
        write_report(diag, build_charts(forward, close, label), diag.report_path)
    return diag


def _descriptive(frame: pd.DataFrame) -> pd.DataFrame:
    try:
        return label_path_structure_statistics(frame, state_col="state", returns_col="returns", price_col="close")
    except ValueError:
        return pd.DataFrame()
