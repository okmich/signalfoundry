import pytest
from okmich_quant_ml.hmm.inference_cache import InferenceCache


class TestInferenceCache:
    """Test suite for InferenceCache."""

    def test_initialization(self):
        """Test cache initialization with valid capacity."""
        cache = InferenceCache(capacity=100)
        assert cache.capacity == 100
        assert len(cache) == 0
        assert cache.is_empty
        assert not cache.is_full

    def test_invalid_capacity(self):
        """Test that invalid capacity raises ValueError."""
        with pytest.raises(ValueError):
            InferenceCache(capacity=0)

        with pytest.raises(ValueError):
            InferenceCache(capacity=-5)

    def test_append_and_latest(self):
        """Test appending probabilities and retrieving the latest."""
        cache = InferenceCache(capacity=10)

        probs1 = [0.7, 0.2, 0.1]
        cache.append(probs1)
        assert cache.latest() == (0.7, 0.2, 0.1)
        assert len(cache) == 1
        assert not cache.is_empty

        probs2 = [0.6, 0.3, 0.1]
        cache.append(probs2)
        assert cache.latest() == (0.6, 0.3, 0.1)
        assert len(cache) == 2

    def test_immutability(self):
        """Test that stored probabilities are immutable."""
        cache = InferenceCache(capacity=10)

        probs = [0.7, 0.2, 0.1]
        cache.append(probs)

        # Modify original list
        probs[0] = 0.999

        # Cached value should be unchanged
        assert cache.latest() == (0.7, 0.2, 0.1)

    def test_latest_empty_cache(self):
        """Test that latest() raises IndexError on empty cache."""
        cache = InferenceCache(capacity=10)

        with pytest.raises(IndexError):
            cache.latest()

    def test_window_basic(self):
        """Test window retrieval."""
        cache = InferenceCache(capacity=10)

        cache.append([0.7, 0.3])
        cache.append([0.6, 0.4])
        cache.append([0.5, 0.5])

        # Get last 2 entries
        window = cache.window(2)
        assert len(window) == 2
        assert window[0] == (0.6, 0.4)  # Older
        assert window[1] == (0.5, 0.5)  # Newer

    def test_window_exceeds_cache_size(self):
        """Test window when k is larger than cache size."""
        cache = InferenceCache(capacity=10)

        cache.append([0.7, 0.3])
        cache.append([0.6, 0.4])

        # Request more than available
        window = cache.window(5)
        assert len(window) == 2
        assert window == [(0.7, 0.3), (0.6, 0.4)]

    def test_window_zero(self):
        """Test window with k=0."""
        cache = InferenceCache(capacity=10)
        cache.append([0.7, 0.3])

        window = cache.window(0)
        assert window == []

    def test_window_negative(self):
        """Test that negative k raises ValueError."""
        cache = InferenceCache(capacity=10)

        with pytest.raises(ValueError):
            cache.window(-1)

    def test_capacity_overflow(self):
        """Test FIFO behavior when capacity is exceeded."""
        cache = InferenceCache(capacity=3)

        cache.append([0.1, 0.9])  # Entry 1
        cache.append([0.2, 0.8])  # Entry 2
        cache.append([0.3, 0.7])  # Entry 3
        assert len(cache) == 3
        assert cache.is_full

        # This should expire the first entry
        cache.append([0.4, 0.6])  # Entry 4
        assert len(cache) == 3  # Still at capacity
        assert cache.is_full

        # First entry should be gone
        window = cache.window(10)
        assert window[0] == (0.2, 0.8)  # Entry 2 is now oldest
        assert window[1] == (0.3, 0.7)  # Entry 3
        assert window[2] == (0.4, 0.6)  # Entry 4 is newest

    def test_empty_probs_validation(self):
        """Test that empty probs sequence raises ValueError."""
        cache = InferenceCache(capacity=10)

        with pytest.raises(ValueError):
            cache.append([])

    def test_invalid_probs_type(self):
        """Test that non-sequence probs raises TypeError."""
        cache = InferenceCache(capacity=10)

        with pytest.raises(TypeError):
            cache.append(0.5)  # Not a sequence

        with pytest.raises(TypeError):
            cache.append(None)

    def test_repr(self):
        """Test string representation."""
        cache = InferenceCache(capacity=100)
        cache.append([0.7, 0.3])

        repr_str = repr(cache)
        assert "capacity=100" in repr_str
        assert "size=1" in repr_str


def example_live_trading_loop():
    """
    Example: Canonical usage pattern for live/backtest inference.

    This demonstrates how to use InferenceCache in a live trading loop
    to maintain immutable, causal inference history.
    """
    # Initialize cache with capacity for 1000 time steps
    cache = InferenceCache(capacity=1000)

    # Simulated live trading loop
    n_states = 3
    observations = [
        [1.2, 0.5, -0.3],
        [0.8, 0.2, 0.1],
        [-0.5, -0.8, 0.2],
        [1.5, 0.9, 0.3],
    ]

    for t, x_t in enumerate(observations):
        # Step 1: Run causal (filter) inference on new observation
        # In real usage, this would be: probs_t = hmm.predict_proba(x_t)
        # For demo purposes, using dummy probabilities
        probs_t = [0.5 + 0.1 * t, 0.3, 0.2 - 0.1 * t]
        probs_t = [max(0.01, min(0.99, p)) for p in probs_t]  # Clip to valid range
        probs_t = [p / sum(probs_t) for p in probs_t]  # Normalize

        # Step 2: Append to immutable cache
        cache.append(probs_t)

        # Step 3: Use current belief for trading decision
        current_belief = cache.latest()
        most_likely_state = current_belief.index(max(current_belief))

        print(
            f"Time {t}: Observation={x_t}, Belief={current_belief}, State={most_likely_state}"
        )

        # Optional: Look at recent history for pattern detection
        if len(cache) >= 3:
            recent_history = cache.window(3)
            print(f"  Recent 3-step history: {recent_history}")

    # After loop: All past inferences remain unchanged
    print(f"\nFinal cache size: {len(cache)}")
    all_inferences = cache.window(len(cache))
    print(f"First inference: {all_inferences[0]}")
    print(f"Last inference: {cache.latest()}")


def example_comparison_with_smoothing():
    """
    Example: Demonstrating the difference between filter and smoother.

    This shows why InferenceCache stores filter beliefs, not smoothed results.
    """
    print("=" * 70)
    print("Filter (Causal) vs Smoother (Non-Causal) Comparison")
    print("=" * 70)

    # Cache stores ONLY filter beliefs (causal)
    filter_cache = InferenceCache(capacity=100)

    # Simulate filtering inference
    observations = [[1.2], [0.8], [-0.5], [1.5]]

    print("\nFilter (Causal) - What InferenceCache stores:")
    print("-" * 70)

    for t, x_t in enumerate(observations):
        # Filter belief at time t given data up to t
        filter_probs = [0.6 - 0.05 * t, 0.3, 0.1 + 0.05 * t]
        filter_probs = [p / sum(filter_probs) for p in filter_probs]

        filter_cache.append(filter_probs)
        print(f"t={t}: P(S_t | x_0:t) = {tuple(round(p, 3) for p in filter_probs)}")

    print("\nSmoother (Non-Causal) - What we DO NOT store:")
    print("-" * 70)
    print("After seeing all data, backward pass would update beliefs retroactively:")
    print("t=0: P(S_0 | x_0:3) = (0.52, 0.31, 0.17) <- Different from filter!")
    print("t=1: P(S_1 | x_0:3) = (0.54, 0.30, 0.16) <- Different from filter!")
    print("t=2: P(S_2 | x_0:3) = (0.56, 0.29, 0.15) <- Different from filter!")
    print("t=3: P(S_3 | x_0:3) = (0.55, 0.30, 0.15) <- Same as filter")

    print("\n" + "=" * 70)
    print("InferenceCache maintains causal integrity:")
    print("=" * 70)
    all_filter_beliefs = filter_cache.window(4)
    for t, belief in enumerate(all_filter_beliefs):
        print(f"t={t}: {tuple(round(p, 3) for p in belief)} (IMMUTABLE)")

    print("\nKey insight: Filter beliefs remain unchanged, enabling:")
    print("  1. Exact backtest <-> live parity")
    print("  2. Reproducible trading decisions")
    print("  3. Causal performance attribution")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("InferenceCache Usage Examples")
    print("=" * 70 + "\n")

    example_live_trading_loop()
    print("\n")
    example_comparison_with_smoothing()
