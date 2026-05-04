"""Offline ruptures labelling for diagnostic / research use only.

Module path: ``okmich_quant_labelling.diagnostics.ruptures``.

The single sanctioned use case is producing hindsight teacher boundaries against which causal BOCPD posterior features are evaluated.
Per the posteriors-first paradigm, every label produced here is paired with the BOCPD run-length posterior trajectory observed
up to that bar; downstream consumers receive ``(label, posterior)`` pairs, never bare hard labels.

NOT for live signaling, ensembling, regime naming, or "confidence" outputs — use ``okmich_quant_ml.bocpd`` or ``okmich_quant_ml.cumsum`` for those.

Public surface
--------------
- ``label_with_posteriors`` — produces a ``LabeledPosteriors`` artefact.
- ``LabeledPosteriors`` — frozen pairing of hindsight ruptures labels and the causal BOCPD posterior trajectory.
- ``UnivariateCost`` — sanctioned ruptures cost functions for univariate input.

Posterior summaries (spec §6, applied to ``LabeledPosteriors.posterior``):
- ``cp_prob``, ``map_run_length``, ``expected_run_length``, ``entropy``,
  ``mass_below_k``, ``posterior_js_innovation``.

Targets (spec §1 catalogue):
- ``is_boundary`` (recommended primary target), ``cp_distance`` (right-censored survival), ``within_segment_position``,
  ``censor_fold_edge_segments``, ``CpDistanceTarget``.
"""

from okmich_quant_labelling.diagnostics.ruptures.artefacts import LabeledPosteriors
from okmich_quant_labelling.diagnostics.ruptures.enums import UnivariateCost
from okmich_quant_labelling.diagnostics.ruptures.labeler import label_with_posteriors
from okmich_quant_labelling.diagnostics.ruptures.posterior_features import (
    cp_prob,
    entropy,
    expected_run_length,
    map_run_length,
    mass_below_k,
    posterior_js_innovation,
)
from okmich_quant_labelling.diagnostics.ruptures.targets import (
    CpDistanceTarget,
    censor_fold_edge_segments,
    cp_distance,
    is_boundary,
    within_segment_position,
)

__all__ = [
    "CpDistanceTarget",
    "LabeledPosteriors",
    "UnivariateCost",
    "censor_fold_edge_segments",
    "cp_distance",
    "cp_prob",
    "entropy",
    "expected_run_length",
    "is_boundary",
    "label_with_posteriors",
    "map_run_length",
    "mass_below_k",
    "posterior_js_innovation",
    "within_segment_position",
]
