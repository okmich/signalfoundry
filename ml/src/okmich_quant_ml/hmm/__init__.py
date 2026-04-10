from .duration import BaseDuration, GammaDuration, LogNormalDuration, NegBinDuration, NonparametricDuration, PoissonDuration
from .evaluator import RegimeQualityEvaluator
from .factorial import FactorialHMM
from .inference_cache import InferenceCache
from .pomegranate import PomegranateHMM, DistType
from .pomegranate_mm import PomegranateMixtureHMM
from .util import DurationType, InferenceMode


def _build_duration_model(duration_type: DurationType, n_states: int, max_duration: int) -> BaseDuration:
    """Construct a duration model from the enum type."""
    match duration_type:
        case DurationType.POISSON:
            return PoissonDuration(n_states, max_duration)
        case DurationType.NONPARAMETRIC:
            return NonparametricDuration(n_states, max_duration)
        case DurationType.NEGBIN:
            return NegBinDuration(n_states, max_duration)
        case DurationType.GAMMA:
            return GammaDuration(n_states, max_duration)
        case DurationType.LOGNORMAL:
            return LogNormalDuration(n_states, max_duration)
        case _:
            raise ValueError(f"Unknown duration type: {duration_type}")


def _infer_duration_type(duration_model: BaseDuration) -> DurationType:
    """Infer DurationType enum from a duration model instance."""
    if isinstance(duration_model, PoissonDuration):
        return DurationType.POISSON
    if isinstance(duration_model, NonparametricDuration):
        return DurationType.NONPARAMETRIC
    if isinstance(duration_model, NegBinDuration):
        return DurationType.NEGBIN
    if isinstance(duration_model, GammaDuration):
        return DurationType.GAMMA
    if isinstance(duration_model, LogNormalDuration):
        return DurationType.LOGNORMAL
    raise ValueError(
        f"Unsupported duration model type: {type(duration_model).__name__}. "
        "Pass duration_types explicitly to train()."
    )


# ------------------------------------------------------------------
# Factory function for Pomegranate HMM instances
# ------------------------------------------------------------------
def create_simple_hmm_instance(dist_type: DistType, n_states: int = 3, *, n_components: int | None = None,
                              is_mixture_model: bool = False, random_state: int = 100, max_iter: int = 100,
                              inference_mode: InferenceMode = InferenceMode.FILTERING,
                              covariance_type: str = "full", min_cov: float | None = None,
                              duration_type: DurationType | None = None, max_duration: int = 100,
                              **dist_kwargs) -> PomegranateHMM | PomegranateMixtureHMM:
    # Build duration model if requested
    duration_model = None
    if duration_type is not None:
        duration_model = _build_duration_model(duration_type, n_states, max_duration)

    # Add covariance parameters for distributions that support them
    distributions_with_covariance = {DistType.NORMAL, DistType.STUDENTT, DistType.LOGNORMAL}
    if dist_type in distributions_with_covariance:
        dist_kwargs['covariance_type'] = covariance_type
        if min_cov is not None:
            dist_kwargs['min_cov'] = min_cov

    # Create mixture model if requested
    if is_mixture_model:
        if n_components is None:
            raise ValueError("n_components must be specified when is_mixture_model=True")

        # Validate n_components value
        if n_components < 2:
            raise ValueError(f"Number of mixture components must be >= 2, got {n_components}")

        # Check if distribution type is supported for mixture models
        mixture_supported = {
            DistType.NORMAL, DistType.STUDENTT, DistType.LOGNORMAL,
            DistType.GAMMA, DistType.EXPONENTIAL, DistType.LAMDA
        }
        if dist_type not in mixture_supported:
            raise ValueError(
                f"Distribution type {dist_type.name} is not supported for mixture models. "
                f"Supported types: {[d.name for d in mixture_supported]}")

        return PomegranateMixtureHMM(distribution_type=dist_type, n_states=n_states, n_components=n_components,
                                     random_state=random_state, max_iter=max_iter,
                                     inference_mode=inference_mode, duration_model=duration_model, **dist_kwargs)
    else:
        return PomegranateHMM(distribution_type=dist_type, n_states=n_states, random_state=random_state,
                              max_iter=max_iter, inference_mode=inference_mode,
                              duration_model=duration_model, **dist_kwargs)
