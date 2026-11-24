"""General hierarchical HMM engine: any state tree, exact inference by flattening, EM on the tree's parameters.

The engine knows nothing about markets. Observation builders (zigzag legs, bar features) and paper presets
("flavours", e.g. Tayal 2009 or the embedded HHMM of Adam et al. 2019) live beside it in
``okmich_quant_ml.hmm.hierarchical``.
"""
from .emissions import CategoricalEmission, CovarianceType, GaussianEmission, LeafEmission, StudentTEmission
from .model import HHMM, FitReport
from .schedule import ExitMode, ExitSchedule
from .topology import StateNode, Topology

__all__ = [
    "HHMM",
    "FitReport",
    "StateNode",
    "Topology",
    "ExitMode",
    "ExitSchedule",
    "LeafEmission",
    "CategoricalEmission",
    "GaussianEmission",
    "StudentTEmission",
    "CovarianceType",
]
