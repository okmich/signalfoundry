from __future__ import annotations

from typing import Any, Protocol

from numpy.typing import NDArray


class PosteriorTransformer(Protocol):
    """Transform posterior probabilities while preserving simplex structure."""

    def transform(self, probs: NDArray) -> NDArray:
        """Return transformed probabilities with shape ``(T, K)``."""
        ...


class PosteriorInferer(Protocol):
    """Consume posterior probabilities and produce consumer-specific output."""

    def infer(self, probs: NDArray) -> Any:
        """Return inferred output from posterior probabilities."""
        ...

    def get_metadata(self) -> dict:
        """Return inferer metadata for diagnostics or reason codes."""
        ...
