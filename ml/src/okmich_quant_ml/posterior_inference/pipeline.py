from __future__ import annotations

from typing import Any

from numpy.typing import NDArray

from .protocols import PosteriorInferer, PosteriorTransformer


class PosteriorPipeline:
    """Sequential chain of posterior transformers and an optional inferer."""

    def __init__(self, transformers: list[PosteriorTransformer], inferer: PosteriorInferer | None = None) -> None:
        self.transformers = transformers
        self.inferer = inferer

    def run(self, probs: NDArray) -> Any:
        transformed = probs
        for transformer in self.transformers:
            transformed = transformer.transform(transformed)

        if self.inferer is None:
            return transformed

        return self.inferer.infer(transformed)
