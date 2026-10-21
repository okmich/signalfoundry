"""
Coupled HMM (Brand et al., 1997): several chains, each chain's transitions conditioned on its parent chains' previous
states, inferred exactly on the joint state space.

The factorial HMM is the special case with no coupling; clamping a chain to observed states gives the input-output
HMM's one-way coupling with a one-hot input. Emission blocks (``GaussianEmission``, ``LocationScaleEmission``,
``ClampedEmission``) each read one column of X and depend on one or two chains.
"""

from .emissions import ClampedEmission, Emission, GaussianEmission, LocationScaleEmission
from .model import CoupledHMM, FitReport

__all__ = ["ClampedEmission", "CoupledHMM", "Emission", "FitReport", "GaussianEmission", "LocationScaleEmission"]
