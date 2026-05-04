"""Frozen artefact pairing hindsight ruptures labels with the causal BOCPD posterior."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from okmich_quant_labelling.diagnostics.ruptures.enums import UnivariateCost


@dataclass(frozen=True)
class LabeledPosteriors:
    """Hindsight ruptures labels paired with the causal BOCPD posterior trajectory.

    Stores every scalar setting that ``label_with_posteriors`` consumes from PELT and BOCPD, plus lightweight self-identification fields
    (``warm_up_length``, ``observation_model_class``) that distinguish two runs whose other settings happen to coincide. **Full reproducibility
    requires the caller to record externally**: (a) the contents of ``warm_up_signal`` and ``signal``; (b) the BOCPD observation-model
    hyperparameters (e.g. NIG ``mu_0``/``kappa_0``/``alpha_0``/``beta_0``).
    The dataclass cannot store the full observation-model state without coupling itself to every model class, and that coupling is deliberately
    out of scope.

    Attributes
    ----------
    breakpoints : NDArray
        Sorted breakpoints exactly as returned by ``rpt.Pelt.predict``: the last element equals ``len(signal)`` and is
        the trailing endpoint, not a change point. Interior breakpoints are ``breakpoints[:-1]``. A value ``b``
        (other than the trailing endpoint) means the segment occupying slice ``[prev_b:b]`` is closed and a new segment begins at index ``b``.
    segment_ids : NDArray
        Per-bar integer segment id (offline, hindsight). Shape ``(T,)``.
    posterior : NDArray
        Per-bar BOCPD run-length posterior. Shape ``(T, r_max)``. Row ``t`` depends only on ``warm_up_signal`` followed by ``signal[:t+1]`` —
        strictly causal. When ``warm_up_signal`` is provided, the posterior at ``t = 0`` reflects the warmed BOCPD state, not the prior.
    cost_model : UnivariateCost
        Ruptures cost function used.
    penalty : float
        Penalty parameter passed to ``rpt.Pelt.predict``.
    hazard_rate : float
        BOCPD hazard rate used to generate ``posterior``. Required for ``posterior_js_innovation``.
    r_max : int
        Maximum BOCPD run-length tracked. Equals ``posterior.shape[1]``.
    min_size : int
        PELT minimum segment size used to generate ``breakpoints``.
    warm_up_length : int
        Number of observations consumed by BOCPD before ``signal``. Equals
        0 when ``warm_up_signal`` is None or empty.
    observation_model_class : str
        ``type(bocpd_observation_model).__name__``. Distinguishes ``"NormalInverseGammaModel"`` from ``"GaussianKnownVarianceModel"``
        from ``"GammaExponentialModel"`` etc.; does **not** capture the model's hyperparameters — the caller must record those externally.
    """

    breakpoints: NDArray
    segment_ids: NDArray
    posterior: NDArray
    cost_model: UnivariateCost
    penalty: float
    hazard_rate: float
    r_max: int
    min_size: int
    warm_up_length: int
    observation_model_class: str
