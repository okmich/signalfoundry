"""Configuration types for HmmFeatureScreener."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from okmich_quant_ml.hmm import DistType, PomegranateHMM, PomegranateMixtureHMM

from ..registry import SIGNAL_TYPES


class ScreenStrategy(StrEnum):
    """Subset enumeration strategy for the screener."""
    ABLATION = "ablation"
    EXHAUSTIVE = "exhaustive"


# Algo registry mirroring okmich_quant_research.backtesting.cluster_comparison_pipeline.
# (algo_key -> (DistType, is_mixture)). Kept local to avoid coupling to that module's private internals.
HMM_ALGO_REGISTRY: dict[str, tuple[DistType, bool]] = {
    "hmm_pmgnt":      (DistType.NORMAL,   False),
    "hmm_lambda":     (DistType.LAMDA,    False),
    "hmm_student":    (DistType.STUDENTT, False),
    "hmm_mm_pmgnt":   (DistType.NORMAL,   True),
    "hmm_mm_lambda":  (DistType.LAMDA,    True),
    "hmm_mm_student": (DistType.STUDENTT, True),
}


def build_hmm(algo: str, n_states: int, mm_n_components: int, random_state: int | None = None):
    """Build a fresh HMM model from an algo key."""
    if algo not in HMM_ALGO_REGISTRY:
        raise ValueError(f"Unknown algo={algo!r}. Known: {sorted(HMM_ALGO_REGISTRY)}")
    dist_type, is_mixture = HMM_ALGO_REGISTRY[algo]
    if is_mixture:
        return PomegranateMixtureHMM(distribution_type=dist_type, n_states=n_states,
                                     n_components=mm_n_components, random_state=random_state)
    return PomegranateHMM(distribution_type=dist_type, n_states=n_states, random_state=random_state)


@dataclass(frozen=True)
class HmmScreenerConfig:
    """Configuration for an HmmFeatureScreener run.

    The screener fits one HMM per candidate feature subset using ``(algo, n_states, mm_n_components)`` and
    computes axis-matched diagnostics against the ``signal_type``'s evaluator.

    Strict-default off-axis check: ``allowed_signal_types`` defaults to ``frozenset({signal_type})``. To screen a model
    that legitimately combines multiple axes, pass a wider set explicitly.
    """
    signal_type: str
    algo: str
    n_states: int
    mm_n_components: int = 3
    data_size: int = 80_000
    horizons: tuple[int, ...] = (12, 60)
    honesty_threshold: float = 0.99
    honesty_trap_rate: float = 0.40
    # Phase-A structural quality gate (run before Pareto). Subsets failing either
    # check are classified FRAGILE and excluded from the Pareto frontier.
    min_significant_states: int = 2
    max_balance_ratio: float = 10.0
    allowed_signal_types: frozenset[str] | None = None
    raise_on_off_axis: bool = False
    random_state: int | None = None

    def __post_init__(self):
        if self.signal_type not in SIGNAL_TYPES:
            raise ValueError(f"signal_type={self.signal_type!r} not in registry SIGNAL_TYPES "
                             f"({sorted(SIGNAL_TYPES)})")
        if self.algo not in HMM_ALGO_REGISTRY:
            raise ValueError(f"algo={self.algo!r} not in HMM_ALGO_REGISTRY "
                             f"({sorted(HMM_ALGO_REGISTRY)})")
        if self.n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {self.n_states}")
        if self.allowed_signal_types is not None:
            unknown = set(self.allowed_signal_types) - set(SIGNAL_TYPES)
            if unknown:
                raise ValueError(f"allowed_signal_types contains unknown values: {unknown}")
        if not 0.0 <= self.honesty_threshold <= 1.0:
            raise ValueError(f"honesty_threshold must be in [0, 1], got {self.honesty_threshold}")
        if not 0.0 <= self.honesty_trap_rate <= 1.0:
            raise ValueError(f"honesty_trap_rate must be in [0, 1], got {self.honesty_trap_rate}")
        if self.min_significant_states < 1:
            raise ValueError(f"min_significant_states must be >= 1, got {self.min_significant_states}")
        if self.min_significant_states > self.n_states:
            raise ValueError(
                f"min_significant_states ({self.min_significant_states}) cannot exceed n_states ({self.n_states})"
            )
        if self.max_balance_ratio < 1.0:
            raise ValueError(f"max_balance_ratio must be >= 1.0, got {self.max_balance_ratio}")

    @property
    def effective_allowed_signal_types(self) -> frozenset[str]:
        """Resolve the strict-default sentinel: ``None`` -> ``{signal_type}``."""
        return self.allowed_signal_types if self.allowed_signal_types is not None else frozenset({self.signal_type})
