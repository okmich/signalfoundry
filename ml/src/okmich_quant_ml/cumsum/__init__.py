"""Univariate sequential CUSUM detection. See ``signalfoundry-lab/docs/cusum.md``."""

from .calibration import CalibrationMethod, calibrate_from_window, target_arl_threshold
from .detector import CusumDetector
from .projection import accumulation_start, collapse_to_binary, first_crossings, soft_alarm_projection
from .protocols import ReferenceModel
from .reference_models import EwmaReferenceModel, GaussianReferenceModel, Sided, SignCusumReferenceModel

__all__ = [
    "CalibrationMethod",
    "CusumDetector",
    "EwmaReferenceModel",
    "GaussianReferenceModel",
    "ReferenceModel",
    "Sided",
    "SignCusumReferenceModel",
    "accumulation_start",
    "calibrate_from_window",
    "collapse_to_binary",
    "first_crossings",
    "soft_alarm_projection",
    "target_arl_threshold",
]
