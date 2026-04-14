from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def margin(probs: NDArray) -> NDArray:
    """Return top-minus-second probability margin along the last axis."""
    sorted_probs = np.sort(probs, axis=-1)
    return sorted_probs[..., -1] - sorted_probs[..., -2]


def entropy(probs: NDArray) -> NDArray:
    """Return Shannon entropy in nats along the last axis."""
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=-1)
