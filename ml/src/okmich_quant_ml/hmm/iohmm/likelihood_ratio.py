"""Did the input help? The nested likelihood-ratio test of an IOHMM against the same HMM without inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import chi2

from ..util import DistType
from .model import PomegranateIOHMM


@dataclass(frozen=True)
class InputLikelihoodRatio:
    """Result of :func:`input_likelihood_ratio_test`. ``statistic = 2 (loglik_full - loglik_null)`` ~ chi2(df)."""

    null_model: PomegranateIOHMM
    full_model: PomegranateIOHMM
    null_log_likelihood: float
    full_log_likelihood: float
    statistic: float
    df: int
    p_value: float


def input_likelihood_ratio_test(X: np.ndarray, U: np.ndarray, lengths: Optional[Sequence[int]] = None, *,
                                distribution_type: DistType, n_states: int = 2, **model_kwargs) -> InputLikelihoodRatio:
    """
    Fit the HMM without inputs, then the IOHMM started from it, and test ``W = 0`` by the likelihood ratio.

    Starting the full model at the null's optimum makes the pair properly nested: EM can only raise the likelihood
    from there, so the statistic is never negative because of a different local optimum. ``df = K (K - 1) d``.

    Caveats. The chi-square reference is asymptotic (it holds for HMM maximum likelihood under the usual regularity
    conditions, serial dependence included). The L2 penalty shrinks ``W`` slightly, which makes the test a little
    conservative. It says nothing about forward usefulness, and screening many candidate inputs needs a
    multiple-testing correction.
    """
    null = PomegranateIOHMM(distribution_type, n_states, **model_kwargs).fit(X, lengths)
    full = PomegranateIOHMM(distribution_type, n_states, **model_kwargs).fit(X, lengths, U=U, init_model=null)
    ll_null = null.selection_scores(X, lengths, with_entropy=False).log_likelihood
    ll_full = full.selection_scores(X, lengths, U=U, with_entropy=False).log_likelihood
    statistic = max(0.0, 2.0 * (ll_full - ll_null))
    df = n_states * (n_states - 1) * full.n_inputs
    return InputLikelihoodRatio(null_model=null, full_model=full, null_log_likelihood=ll_null,
                                full_log_likelihood=ll_full, statistic=statistic, df=df,
                                p_value=float(chi2.sf(statistic, df)))
