"""Configuration types for HmmFeatureScreener."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from okmich_quant_ml.hmm import DistType, PomegranateHMM, PomegranateMixtureHMM

from ..registry import SIGNAL_TYPES


class ScreenStrategy(StrEnum):
    """Subset enumeration strategy for the screener.

    ``ABLATION`` is anchored: every add-one subset is ``baseline + candidate``, so no subset ever
    pairs two non-baseline features and the search is a 1-neighbourhood of a hand-chosen fixed
    point. Cheap (``1 + n`` fits) but its winner is a restatement of the anchor as much as a
    finding -- read ``HmmScreenerResult.base_frac`` beside any ablation verdict.

    ``GREEDY_FORWARD`` drops the anchor: it beam-searches up from the empty set, so the first
    feature is chosen by the data rather than by hand. Costs roughly
    ``n + beam_width * (n-1 + ... + n-depth+1)`` fits (~464 at n=38, B=3, depth 5) -- an order of
    magnitude more than ablation, but the only affordable unanchored option: ``EXHAUSTIVE`` is
    ``2**n - 1`` and is out of reach above n ~ 16.

    ``EXHAUSTIVE`` is kept for small pools and for tests; it is not tractable on a full recipe.
    """
    ABLATION = "ablation"
    GREEDY_FORWARD = "greedy_forward"
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
    # When True, forward-looking axis evaluators (momentum / direction / volatility /
    # liquidity) clip the forward-return / forward-vol / forward-volume window at
    # session (calendar-date) boundaries: a bar whose forward window would cross the
    # overnight gap is dropped from scoring instead of carrying an overnight-gap-
    # contaminated label. Default False preserves the legacy (leaky) behaviour so
    # existing screens are unchanged until they opt in. Requires a DatetimeIndex on
    # ``raw_data``; silently a no-op otherwise.
    respect_session_boundaries: bool = False
    # Stage-0b marginal-persistence floor: max(|acf1|) over x, |x|, x^2. HMM-specific (a tree screener
    # may legitimately use white-noise features, hence not in the shared stage-0).
    # Default 0.0 = DIAGNOSTIC ONLY (scored and reported, nothing removed), which also preserves existing
    # screens unchanged until they opt in -- matching the convention used by respect_session_boundaries.
    # Removal is opt-in because the test is per-feature and MARGINAL while the emission is joint and
    # shape-aware: it is provably blind to covariance regimes (marginals white, correlation switches) and
    # to tail-shape regimes (equal variance, different kurtosis) that the model can use. 0.15 is the
    # FXPIG-M5 calibration only -- not validated cross-instrument or out-of-sample. See _persistence.py.
    min_persistence: float = 0.0
    # Stage-0c collinearity ceiling: VIF = 1/(1-R^2) of each feature on the others. HMM-specific -- a
    # near-duplicate pair ill-conditions the joint emission covariance and makes the multi-restart max-LL
    # pick a coin flip. Default inf = DIAGNOSTIC ONLY (scored and reported, nothing removed), preserving
    # existing screens until they opt in. Removal is opt-in and the threshold must stay HIGH: VIF sees
    # only static linear redundancy and is blind to regime-switching covariance the emission can use, so
    # this is a near-duplicate remover, NOT an orthogonaliser. 10 is the FXPIG-M5 setting. See _collinearity.py.
    max_vif: float = float("inf")
    honesty_threshold: float = 0.99
    honesty_trap_rate: float = 0.40
    # Phase-A structural quality gate (run before Pareto). Subsets failing either
    # check are classified FRAGILE and excluded from the Pareto frontier.
    min_significant_states: int = 2
    max_balance_ratio: float = 10.0
    allowed_signal_types: frozenset[str] | None = None
    raise_on_off_axis: bool = False
    random_state: int | None = None
    # --- GREEDY_FORWARD beam search -------------------------------------------------------
    # Beam width. B=1 is plain greedy, which merely moves the fixed point one step later: the
    # step-1 winner then anchors everything after it, and step 1 ranks n single-feature HMMs on an
    # in-sample criterion where rank 1 vs rank 3 is easily inside noise. B>=3 keeps the top paths
    # alive so path-dependence is measurable instead of assumed -- it is the structural replacement
    # for re-running an anchored screen under a second hand-picked baseline.
    beam_width: int = 3
    # Depth cap, in features. Overridden by ``screen(max_subset_size=...)`` when that is passed.
    greedy_max_depth: int = 5
    # Early stop: end the search when the best step gain falls below this FRACTION of the best
    # separation so far. Relative rather than absolute because axis_separation is in the axis
    # target unit and differs by orders of magnitude across axes (log-returns vs tick volume).
    greedy_min_relative_gain: float = 0.02

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
        if not 0.0 <= self.min_persistence <= 1.0:
            raise ValueError(f"min_persistence must be in [0, 1], got {self.min_persistence}")
        if self.max_vif != self.max_vif:  # NaN
            raise ValueError("max_vif must be a number (or inf), got NaN")
        if self.max_vif < 1.0:
            raise ValueError(f"max_vif must be >= 1.0 (VIF's floor for an orthogonal feature), got {self.max_vif}")
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
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")
        if self.greedy_max_depth < 1:
            raise ValueError(f"greedy_max_depth must be >= 1, got {self.greedy_max_depth}")
        if self.greedy_min_relative_gain < 0.0:
            raise ValueError(
                f"greedy_min_relative_gain must be >= 0, got {self.greedy_min_relative_gain}")

    @property
    def effective_allowed_signal_types(self) -> frozenset[str]:
        """Resolve the strict-default sentinel: ``None`` -> ``{signal_type}``."""
        return self.allowed_signal_types if self.allowed_signal_types is not None else frozenset({self.signal_type})
