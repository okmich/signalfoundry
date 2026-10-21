"""
Input-Output HMM (Bengio & Frasconi, 1995): observed inputs drive the transitions, pomegranate emissions stay shared.

``PomegranateIOHMM`` duck-types ``PomegranateHMM``: every public method takes the same positional parameters, and the
inputs are the extra keyword-only argument ``U``. Fitted without ``U`` it is an ordinary HMM. ``InputTiming`` (in
``okmich_quant_ml.hmm``) declares when each input becomes known, so inference stays causal.
"""

from .likelihood_ratio import InputLikelihoodRatio, input_likelihood_ratio_test
from .model import PomegranateIOHMM
from .transitions import InputDrivenTransitions

__all__ = ["InputDrivenTransitions", "InputLikelihoodRatio", "PomegranateIOHMM", "input_likelihood_ratio_test"]
