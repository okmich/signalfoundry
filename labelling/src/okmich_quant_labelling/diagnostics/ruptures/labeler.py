"""Offline ruptures labelling paired with the causal BOCPD posterior trajectory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import ruptures as rpt
from numpy.typing import NDArray

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector
from okmich_quant_ml.bocpd.protocols import ObservationModel

from okmich_quant_labelling.diagnostics.ruptures.artefacts import LabeledPosteriors
from okmich_quant_labelling.diagnostics.ruptures.enums import UnivariateCost


def _coerce_1d(arr: pd.Series | NDArray, name: str) -> NDArray:
    if isinstance(arr, pd.Series):
        out = arr.to_numpy(dtype=np.float64, copy=False)
    else:
        out = np.asarray(arr, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {out.shape}")
    if out.size > 0 and not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains NaN or Inf values.")
    return out


def label_with_posteriors(signal: pd.Series | NDArray, bocpd_observation_model: ObservationModel,
                          hazard_rate: float, r_max: int, cost_model: UnivariateCost = UnivariateCost.L2,
                          penalty: float = 10.0, min_size: int = 5, warm_up_signal: pd.Series | NDArray | None = None) -> LabeledPosteriors:
    """Produce hindsight ruptures labels alongside the causal BOCPD posterior.

    The two outputs are computed on the same labelled window ``signal`` but with strictly different information sets:
    ``segment_ids`` use the full ``signal`` (offline, hindsight), ``posterior`` rows are causal and use only past
    observations within ``warm_up_signal`` followed by ``signal``. Pairing them is the contract of this function.

    Fold-local generation: when running per-fold, pass the labelable window as ``signal`` and **all prior observations**
    as ``warm_up_signal``. PELT segments only ``signal``; BOCPD consumes ``warm_up_signal`` first to advance its state,
    so ``posterior[0]`` reflects what a live detector would have believed at the fold edge rather than the BOCPD prior.
    Without the warm-up the eval-window posterior is not live-equivalent.

    Parameters
    ----------
    signal : pd.Series | NDArray
        Univariate time series, 1-D. Multi-dimensional input is rejected — use a future multivariate variant for multivariate costs.
    bocpd_observation_model : ObservationModel
        Conjugate observation model for BOCPD (e.g. NormalInverseGammaModel).
    hazard_rate : float
        BOCPD hazard rate, must satisfy ``0 < hazard_rate < 1``.
    r_max : int
        Maximum run-length tracked by BOCPD, must be ``>= 2``.
    cost_model : UnivariateCost
        Ruptures cost function (univariate-compatible only). Default ``L2``.
    penalty : float
        PELT penalty, must be ``> 0``.
    min_size : int
        Minimum segment size for PELT, must be ``>= 2`` and ``<= len(signal)``.
    warm_up_signal : pd.Series | NDArray | None
        Optional prior observations to advance BOCPD before ``signal``. Must be 1-D and finite if provided. Empty arrays
        are accepted and treated as ``None``. Not used by PELT and not included in ``breakpoints``, ``segment_ids``, or
        returned ``posterior`` rows.

    Returns
    -------
    LabeledPosteriors

    Raises
    ------
    ValueError
        If ``signal`` or ``warm_up_signal`` is not 1-D, contains non-finite values, has fewer than ``min_size``
        observations (signal only), or any scalar precondition is violated.
    TypeError
        If ``cost_model`` is not a member of ``UnivariateCost``.
    """
    if not isinstance(cost_model, UnivariateCost):
        raise TypeError(f"cost_model must be a UnivariateCost member, got {type(cost_model).__name__}")
    if not (0.0 < float(hazard_rate) < 1.0):
        raise ValueError(f"hazard_rate must be in (0, 1), got {hazard_rate}")
    if int(r_max) < 2:
        raise ValueError(f"r_max must be >= 2, got {r_max}")
    if not (float(penalty) > 0.0):
        raise ValueError(f"penalty must be > 0, got {penalty}")
    if int(min_size) < 2:
        raise ValueError(f"min_size must be >= 2, got {min_size}")

    values = _coerce_1d(signal, "signal")
    if values.size < int(min_size):
        raise ValueError(f"signal length {values.size} below min_size={min_size}")

    warm_up = _coerce_1d(warm_up_signal, "warm_up_signal") if warm_up_signal is not None else None

    algo = rpt.Pelt(model=cost_model.value, min_size=int(min_size)).fit(values.reshape(-1, 1))
    breakpoints = np.asarray(algo.predict(pen=float(penalty)), dtype=np.int64)

    segment_ids = np.empty(len(values), dtype=np.int64)
    start = 0
    for seg_id, end in enumerate(breakpoints):
        segment_ids[start:end] = seg_id
        start = end

    detector = BayesianOnlineChangepointDetector(
        observation_model=bocpd_observation_model, hazard_rate=float(hazard_rate), r_max=int(r_max),
    )
    if warm_up is not None and warm_up.size > 0:
        detector.batch(warm_up)
    posterior = detector.batch(values)

    return LabeledPosteriors(
        breakpoints=breakpoints, segment_ids=segment_ids, posterior=posterior,
        cost_model=cost_model, penalty=float(penalty),
        hazard_rate=float(hazard_rate), r_max=int(r_max), min_size=int(min_size),
        warm_up_length=int(warm_up.size) if warm_up is not None else 0,
        observation_model_class=type(bocpd_observation_model).__name__,
    )
