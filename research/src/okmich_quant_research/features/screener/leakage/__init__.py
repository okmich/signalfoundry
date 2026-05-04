"""
Leakage Diagnostics
===================
Post-hoc tooling for detecting interaction-mediated leakage in a ScreenerResult.

This subpackage is the consumer of the persistence fields plumbed onto ScreenerResult (stage1_scores, cluster_assignments,
cluster_representatives, boruta_groups). It does not modify the screener pipeline.

    >>> from okmich_quant_research.features.screener.leakage import SuspectRegistry, resolve,
    >>>
    >>> suspects = SuspectRegistry(prefixes=("tm_vol_",), rationale="vol family")
    >>> resolved = resolve(suspects, screener_result)
    >>> resolved.cluster_lineage   # which clusters absorbed a suspect
"""
from ._diagnostics import LeakageDiagnostics
from ._report import LeakageReport, Severity, classify
from ._sampling import SamplingTask, stratified_row_sample
from ._suspects import ResolvedSuspectSet, SuspectRegistry, resolve

__all__ = [
    "SuspectRegistry",
    "ResolvedSuspectSet",
    "resolve",
    "LeakageDiagnostics",
    "LeakageReport",
    "Severity",
    "classify",
    "SamplingTask",
    "stratified_row_sample",
]
