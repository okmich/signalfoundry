from collections import deque
from typing import Sequence
import numpy as np


class InferenceCache:

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._cache = deque(maxlen=capacity)

    def append(self, probs: Sequence[float] | np.ndarray) -> None:
        """
        Append a single probability vector to the cache.

        Parameters
        ----------
        probs : Sequence[float] or np.ndarray
            Filtered state belief vector at time `t`.
            Should represent a probability distribution (typically sums to 1.0).

        Raises
        ------
        TypeError
            If probs is not a sequence or numpy array.
        ValueError
            If probs is empty.
        """
        if not isinstance(probs, (Sequence, np.ndarray)):
            raise TypeError(
                f"probs must be a Sequence or ndarray, got {type(probs).__name__}"
            )

        if len(probs) == 0:
            raise ValueError("probs cannot be empty")

        # Convert to immutable tuple to enforce immutability
        immutable_probs = tuple(probs)
        self._cache.append(immutable_probs)

    def append_all(self, probs_sequence: Sequence[Sequence[float]] | np.ndarray) -> None:
        """
        Append multiple probability vectors at once (bulk initialization).

        Use this to populate the cache with historical inferences when the cache is empty. For ongoing live trading,
        use append() to add only the latest inference one at a time.

        Parameters
        ----------
        probs_sequence : Sequence[Sequence[float]] or np.ndarray
            Multiple probability vectors to append in order.
            Each inner sequence/row represents a single time step's inference.
            Accepts both nested sequences and 2D numpy arrays.

        Raises
        ------
        TypeError
            If probs_sequence is not a sequence/array or contains non-sequences.
        ValueError
            If probs_sequence is empty or contains empty sequences.

        Examples
        --------
        >>> cache = InferenceCache(capacity=100)
        >>> historical_inferences = [
        ...     [0.7, 0.3],
        ...     [0.6, 0.4],
        ...     [0.5, 0.5]
        ... ]
        >>> cache.append_all(historical_inferences)
        >>> len(cache)
        3

        >>> import numpy as np
        >>> cache2 = InferenceCache(capacity=100)
        >>> probs_array = np.array([[0.7, 0.3], [0.6, 0.4]])
        >>> cache2.append_all(probs_array)
        >>> len(cache2)
        2
        """
        # Accept both Sequence and numpy arrays
        if not isinstance(probs_sequence, (Sequence, np.ndarray)):
            raise TypeError(f"probs_sequence must be a Sequence or ndarray, got {type(probs_sequence).__name__}")

        if len(probs_sequence) == 0:
            raise ValueError("probs_sequence cannot be empty")

        for probs in probs_sequence:
            self.append(probs)

    def latest(self) -> tuple[float, ...]:
        if len(self._cache) == 0:
            raise IndexError("Cannot retrieve latest from empty cache")

        return self._cache[-1]

    def window(self, k: int) -> list[tuple[float, ...]]:
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        if k == 0:
            return []

        # Return up to last k elements
        cache_size = len(self._cache)
        start_idx = max(0, cache_size - k)
        return list(self._cache)[start_idx:]

    def format_contents(self, max_items: int | None = None, precision: int = 4) -> str:
        if self.is_empty:
            return "InferenceCache is empty"

        lines = []
        lines.append(repr(self))
        lines.append("-" * 60)

        # Get items to display (most recent first)
        cache_list = list(self._cache)
        n_items = len(cache_list)

        if max_items is not None:
            display_count = min(max_items, n_items)
        else:
            display_count = n_items

        for i in range(display_count):
            idx = n_items - 1 - i
            probs = cache_list[idx]

            probs_str = ", ".join(f"{p:.{precision}f}" for p in probs)
            if i == 0:
                label = f"[t-{i}] (most recent)"
            else:
                label = f"[t-{i}]"

            lines.append(f"{label:20s}: [{probs_str}]")

        if max_items is not None and n_items > max_items:
            lines.append(f"... and {n_items - max_items} more items")

        lines.append("-" * 60)
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._cache)

    def __repr__(self) -> str:
        return f"InferenceCache(capacity={self._capacity}, size={len(self._cache)})"

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_empty(self) -> bool:
        return len(self._cache) == 0

    @property
    def is_full(self) -> bool:
        return len(self._cache) == self._capacity
