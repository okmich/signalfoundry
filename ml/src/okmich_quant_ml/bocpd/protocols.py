from __future__ import annotations

from typing import Protocol

from numpy.typing import NDArray


class ObservationModel(Protocol):
    """Conjugate observation model for Bayesian online changepoint detection.

    Required methods:
        - ``log_pred_probs(x)``: posterior-predictive log-probability for each run-length slot.
        - ``update(x)``: advance the sufficient-statistic bank by one observation.
        - ``reset(r_max)``: initialise the sufficient-statistic bank with capacity ``r_max``.

    Optional fast-path (duck-typed at runtime by the detector):
        - ``batch_update_posterior(xs, posterior, log_hazard, log_growth) -> NDArray``

    When implemented, ``batch_update_posterior`` must run the full Adams–MacKay
    recursion over ``xs`` and return the per-step run-length posterior matrix of
    shape ``(len(xs), r_max)``. It **must also advance the observation model's
    sufficient-statistic bank to the post-batch state** — equivalent to calling
    ``update(x)`` for every ``x`` in ``xs``. Without that self-update, the
    detector's state becomes silently inconsistent after a batch call.
    """

    def log_pred_probs(self, x: float) -> NDArray:
        """Return log predictive probability of ``x`` for each run-length slot."""
        ...

    def update(self, x: float) -> None:
        """Absorb ``x`` into the sufficient-statistic bank."""
        ...

    def reset(self, r_max: int) -> None:
        """Reset the sufficient-statistic bank with capacity ``r_max``."""
        ...
