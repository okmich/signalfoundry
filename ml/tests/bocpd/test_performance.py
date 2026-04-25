import time

import numpy as np

from okmich_quant_ml.bocpd import BayesianOnlineChangepointDetector, NormalInverseGammaModel


def test_batch_throughput_meets_spec_target() -> None:
    """§10.6: ``batch(10_000, r_max=500)`` under NIG completes within the perf budget.

    Spec target is < 500 ms (≥ 20k bars/sec) on a commodity laptop. The assertion
    here is intentionally relaxed to < 2.0s to absorb CI variance while still
    catching any regression that introduces O(T²) behaviour, breaks the numba
    fast-path, or otherwise blows up per-bar cost.

    A short warmup batch is run first so JIT compilation does not contaminate the
    measured wall-clock — the timed call reflects steady-state kernel performance.
    """
    rng = np.random.default_rng(seed=2026)
    xs = rng.standard_normal(10_000).astype(np.float64)

    warmup = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=1 / 100, r_max=500)
    warmup.batch(xs[:50])

    detector = BayesianOnlineChangepointDetector(NormalInverseGammaModel(), hazard_rate=1 / 100, r_max=500)
    start = time.perf_counter()
    detector.batch(xs)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, (
        f"batch(10_000, r_max=500) took {elapsed:.3f}s; spec target is < 0.5s "
        "(test bound is CI-relaxed to 2.0s)."
    )
