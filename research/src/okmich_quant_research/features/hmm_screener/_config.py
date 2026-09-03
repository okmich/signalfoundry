"""Configuration types for HmmFeatureScreener."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from okmich_quant_ml.hmm import DistType, PomegranateHMM, PomegranateMixtureHMM

from ..registry import AXIS_PRIMARY_HORIZON, AXIS_SIGNAL_TYPES, Axis, SIGNAL_TYPES


class OneSidedPolicy(StrEnum):
    """What a DIRECTIONAL screen does with a lone ONE_SIDED feature.

    A one-sided feature (``momentum.minus_di``, ``timothymasters.trend.aroon_up``, ...) maps onto its
    CONJUGATE under reflection rather than onto its own negation. Its high state means "strong move THIS
    way"; its low state pools "the other way" WITH "no move at all". A K=2 split on one alone is
    therefore NOT an up/down partition -- and ``momentum.minus_di`` is the feature that won the trend
    axis on 11 of 14 FX symbols, so this is a measured defect in shipped results, not a hypothetical.

    ``WARN`` (default) records it on the subset so it reaches ``SubsetEvaluation.warnings`` and the
    results frame. ``EXCLUDE`` additionally marks the subset FRAGILE, keeping it off the Pareto
    frontier. ``RAISE`` refuses outright.

    There is deliberately no SUBSTITUTE policy. Silently swapping a feature for its spread mid-search
    would corrupt the ``seen``/``evaluated`` bookkeeping GREEDY_FORWARD's beam relies on, and would make
    the reported subset differ from the subset actually fitted. Substitute by putting the spread
    (``momentum.di_spread``, ``timothymasters.trend.aroon_diff``) in the candidate pool instead.
    """
    WARN = "warn"
    EXCLUDE = "exclude"
    RAISE = "raise"


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
    computes axis-matched diagnostics against the ``axis``'s evaluator.

    ``axis`` is a ``registry.Axis``, NOT a ``signal_type``. The two were one field until the axis layer
    landed, and conflating them is what let a feature's namespace stand in for what it measures.
    ``allowed_signal_types`` is still about feature TAGS and defaults to ``AXIS_SIGNAL_TYPES[axis]``;
    pass a narrower or wider set explicitly to change what counts as off-axis contamination.

    A signal_type such as ``axis="trend"`` is rejected: it is a feature TAG, not an axis. Both
    ``trend`` and ``momentum`` are ``Axis.DIRECTIONAL`` -- there is one directional axis.
    """
    axis: Axis
    algo: str
    n_states: int
    mm_n_components: int = 3
    data_size: int = 80_000
    horizons: tuple[int, ...] = (12, 60)
    # Primary forward horizon for the axis evaluator, in bars. ``None`` resolves to
    # AXIS_PRIMARY_HORIZON[axis] -- 18 for DIRECTIONAL, 12 elsewhere. It is a per-AXIS quantity, not a
    # global one: measured on the persisted labels, the directional label's unconditional separation
    # peaks at H=18 (nsep 0.091) against H=12 (0.080), and per symbol the best horizon ranges 9..36.
    # The whole existing corpus was screened at 12 only because ``horizons[0]`` was hard-coded as the
    # primary. Set explicitly to override the per-axis default.
    primary_horizon: int | None = None
    # When True, the forward-looking axis evaluators (DIRECTIONAL / VOLATILITY / LIQUIDITY --
    # PATH_STRUCTURE has no forward window) clip the forward-return / forward-vol /
    # forward-volume window at session (calendar-date) boundaries: a bar whose forward
    # window would cross the overnight gap is dropped from scoring instead of carrying an
    # overnight-gap-contaminated label. Default False leaves the window session-blind, which
    # is leaky, so existing screens are unchanged until they opt in. Requires a DatetimeIndex on
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
    # Data-capability precondition for Axis.LIQUIDITY. The library has no feed metadata -- the volume
    # column is resolved by NAME only -- so whether a feed offers real volume is something the caller
    # must assert, not something the data can be asked. On FXPIG (and MT5 generally) ``tick_volume`` is
    # a tick COUNT: screening liquidity on it measures quote activity, not liquidity. Default False
    # refuses the axis rather than silently measuring the wrong thing.
    has_real_volume: bool = False
    # What a DIRECTIONAL screen does with a lone one-sided feature. See OneSidedPolicy.
    one_sided_policy: OneSidedPolicy = OneSidedPolicy.WARN
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
        # Coerce a bare string so axis="directional" works, while a signal_type such as "trend" or
        # "momentum" fails here with the valid axes named. Frozen dataclass -> setattr bypass.
        raw_axis = self.axis
        try:
            object.__setattr__(self, "axis", Axis(raw_axis))
        except (ValueError, TypeError):
            # TypeError as well as ValueError: an unhashable argument (a list, say) fails inside the
            # enum's own value lookup, and letting that escape as "unhashable type: 'list'" would tell
            # the caller nothing about what the field actually wants.
            hint = ""
            if isinstance(raw_axis, str) and raw_axis in SIGNAL_TYPES:
                hint = (f" {raw_axis!r} is a signal_type (a feature TAG), not an axis."
                        f" 'trend' and 'momentum' are both Axis.DIRECTIONAL.")
            raise ValueError(f"axis={raw_axis!r} is not a valid Axis "
                             f"({sorted(a.value for a in Axis)}).{hint}") from None
        if self.primary_horizon is not None and self.primary_horizon < 1:
            raise ValueError(f"primary_horizon must be >= 1, got {self.primary_horizon}")
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
        """Resolve the sentinel: ``None`` -> the feature tags this axis draws on.

        Was ``frozenset({signal_type})``, which only made sense while axis and tag were the same string.
        An axis legitimately draws on several tags -- DIRECTIONAL on trend, momentum and regime -- so the
        old default flagged as "off-axis" exactly the cross-namespace features the measurement showed
        were on-axis all along (only 9 of the 23 trend-pool candidates were tagged ``trend``).
        """
        if self.allowed_signal_types is not None:
            return self.allowed_signal_types
        return AXIS_SIGNAL_TYPES[self.axis]

    @property
    def effective_primary_horizon(self) -> int:
        """Explicit ``primary_horizon`` if set, else the measured per-axis default."""
        if self.primary_horizon is not None:
            return self.primary_horizon
        return AXIS_PRIMARY_HORIZON[self.axis]
