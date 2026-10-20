from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ReferenceModel(Protocol):
    """In-control reference distribution for CUSUM detection.

    See ``signalfoundry-lab/docs/cusum.md`` §3.1 for the full protocol contract,
including the score-then-update purity rule for adaptive references and the
    sampler dispatch used by ARL Monte Carlo (§6).
    """

    def score(self, x: float) -> NDArray:
        """Return per-direction score for scalar observation ``x``. Shape ``(K,)``."""
        ...

    def update(self, x: float) -> None:
        """Optionally advance the reference's sufficient statistics."""
        ...

    def reset(self) -> None:
        """Reinitialise the reference to its prior state."""
        ...

    @property
    def n_directions(self) -> int:
        """K, the number of hypothesis directions emitted by ``score``."""
        ...

    @property
    def requires_external_sampler(self) -> bool:
        """If True, ARL Monte Carlo requires an ``in_control_sampler`` from the caller."""
        ...

    def sample_in_control(self, n: int, rng: np.random.Generator) -> NDArray:
        """Draw ``n`` H0 observations. Required only when ``requires_external_sampler`` is False."""
        ...
