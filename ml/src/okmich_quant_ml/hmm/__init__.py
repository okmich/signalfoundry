from .evaluator import RegimeQualityEvaluator
from .factorial import FactorialHMM
from .inference_cache import InferenceCache
from .pomegranate import PomegranateHMM, DistType
from .pomegranate_mm import PomegranateMixtureHMM
from .util import InferenceMode


_REMOVED_HSMM_KWARGS: frozenset = frozenset({"duration_model", "duration_type", "max_duration"})


# ------------------------------------------------------------------
# Factory function for Pomegranate HMM instances
# ------------------------------------------------------------------
def create_simple_hmm_instance(dist_type: DistType, n_states: int = 3, *, n_components: int | None = None,
                              is_mixture_model: bool = False, random_state: int = 100, max_iter: int = 100,
                              inference_mode: InferenceMode = InferenceMode.FILTERING,
                              covariance_type: str = "full", min_cov: float | None = None,
                              **dist_kwargs) -> PomegranateHMM | PomegranateMixtureHMM:
    bad = _REMOVED_HSMM_KWARGS & dist_kwargs.keys()
    if bad:
        raise TypeError(
            f"HSMM support has been removed. The following kwargs are no longer accepted: {sorted(bad)}. "
            "Remove them from the call."
        )
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
                                     inference_mode=inference_mode, **dist_kwargs)
    else:
        return PomegranateHMM(distribution_type=dist_type, n_states=n_states, random_state=random_state,
                              max_iter=max_iter, inference_mode=inference_mode, **dist_kwargs)