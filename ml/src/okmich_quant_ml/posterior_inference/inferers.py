from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ArgmaxInferer:
    """Bridge inferer that collapses posterior probabilities to hard labels."""

    def infer(self, probs: NDArray) -> NDArray:
        return np.argmax(probs, axis=-1)

    def get_metadata(self) -> dict:
        return {}
