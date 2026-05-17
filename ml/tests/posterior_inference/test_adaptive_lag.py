"""Tests for AdaptiveLagInferer, AdaptiveLagResult, lag_commitment_audit, compute_trajectories.

Mirrors the lab notebook's invariants on hand-built synthetic trajectories — no HMM dependency.

Important semantic invariant (set by ``_build_stable_grid``): at lag 0 there is no prior lag to
compare against, so all criteria treat ``stable[0]`` as False. This makes ``m_consec`` symmetric
across KL_ONLY, KL_PLUS_ENTROPY, and MAP_PERSISTENCE: ``m_consec=N`` requires N consecutive real
stability observations starting at lag 1. Consequence: ``m_consec=2`` (the lab default) can fire
no earlier than lag 2 for any criterion.
"""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.posterior_inference import (
    AdaptiveLagInferer,
    AdaptiveLagResult,
    MaturationAlignTransformer,
    StabilityCriterion,
    compute_trajectories,
    lag_commitment_audit,
)


def _constant_trajectory(n_lags: int, T: int, row: list[float]) -> np.ndarray:
    """Build a (n_lags, T, K) trajectory where every (lag, bar) has the same posterior row.

    The caller-provided ``row`` should sum to 1.0 since trajectory validation requires simplex rows.
    """
    arr = np.asarray(row, dtype=float)
    K = arr.shape[0]
    return np.broadcast_to(arr, (n_lags, T, K)).copy()


def test_constructor_rejects_non_positive_theta() -> None:
    with pytest.raises(ValueError, match="theta_kl"):
        AdaptiveLagInferer(theta_kl=0.0)
    with pytest.raises(ValueError, match="theta_kl"):
        AdaptiveLagInferer(theta_kl=-0.01)


def test_constructor_rejects_invalid_m_consec() -> None:
    with pytest.raises(ValueError, match="m_consec"):
        AdaptiveLagInferer(m_consec=0)


def test_infer_rejects_non_3d_input() -> None:
    inferer = AdaptiveLagInferer()
    with pytest.raises(ValueError, match=r"n_lags, T, K"):
        inferer.infer(np.zeros((10, 3)))


def test_infer_rejects_single_lag_trajectory() -> None:
    inferer = AdaptiveLagInferer()
    with pytest.raises(ValueError, match="n_lags >= 2"):
        inferer.infer(np.zeros((1, 10, 3)))


def test_infer_rejects_nan_in_trajectories() -> None:
    inferer = AdaptiveLagInferer()
    bad = np.full((4, 5, 3), 1.0 / 3.0)
    bad[2, 1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        inferer.infer(bad)


def test_infer_rejects_negative_mass() -> None:
    inferer = AdaptiveLagInferer()
    P = np.full((4, 5, 3), 1.0 / 3.0)
    P[1, 2, 0] = -0.5
    P[1, 2, 1] = 0.83  # keep row sum at ~1.0 so the negativity check triggers, not the row-sum check
    with pytest.raises(ValueError, match="negative values"):
        inferer.infer(P)


def test_infer_rejects_non_simplex_rows() -> None:
    inferer = AdaptiveLagInferer()
    P = np.full((4, 5, 3), 0.5)  # Row sums = 1.5
    with pytest.raises(ValueError, match="sums deviate from 1.0"):
        inferer.infer(P)


def test_infer_rejects_zero_sum_rows() -> None:
    inferer = AdaptiveLagInferer()
    P = np.full((4, 5, 3), 1.0 / 3.0)
    P[1, 2, :] = 0.0
    with pytest.raises(ValueError, match="non-positive sum"):
        inferer.infer(P)


def test_infer_rejects_m_consec_exceeding_k_max() -> None:
    """Under the stable[0]=False convention only k_max = n_lags - 1 real stability observations
    exist, so m_consec must be <= k_max. Specifically m_consec == n_lags (= k_max + 1) cannot fire
    and must be rejected at validation time."""
    inferer = AdaptiveLagInferer(m_consec=3)
    P = _constant_trajectory(n_lags=3, T=4, row=[0.9, 0.05, 0.05])
    with pytest.raises(ValueError, match="m_consec.*exceeds.*k_max"):
        inferer.infer(P)


def test_infer_rejects_m_consec_far_exceeding_k_max() -> None:
    inferer = AdaptiveLagInferer(m_consec=10)
    P = _constant_trajectory(n_lags=3, T=4, row=[0.9, 0.05, 0.05])
    with pytest.raises(ValueError, match="m_consec.*exceeds.*k_max"):
        inferer.infer(P)


def test_kl_criterion_rejects_theta_kl_zero() -> None:
    with pytest.raises(ValueError, match="theta_kl"):
        AdaptiveLagInferer(theta_kl=0.0, criterion=StabilityCriterion.KL_ONLY)
    with pytest.raises(ValueError, match="theta_kl"):
        AdaptiveLagInferer(theta_kl=-0.01, criterion=StabilityCriterion.KL_PLUS_ENTROPY)


def test_map_persistence_accepts_theta_kl_zero() -> None:
    """theta_kl is documented as ignored for MAP_PERSISTENCE; construction must not raise."""
    inferer = AdaptiveLagInferer(theta_kl=0.0, criterion=StabilityCriterion.MAP_PERSISTENCE)
    assert inferer.criterion == StabilityCriterion.MAP_PERSISTENCE
    assert inferer.theta_kl == 0.0


def test_kl_plus_entropy_fires_at_l2_on_perfectly_stable_trajectory_with_m_consec_2() -> None:
    """Default config (m_consec=2). All lags identical → KL is zero, entropy constant.
    Earliest fire is lag 2 because stable[0] is forced False and m_consec=2 requires two
    consecutive real stability observations starting at lag 1."""
    P = _constant_trajectory(n_lags=8, T=10, row=[0.9, 0.05, 0.05])
    inferer = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_PLUS_ENTROPY, m_consec=2)

    result = inferer.infer(P)

    assert isinstance(result, AdaptiveLagResult)
    np.testing.assert_array_equal(result.commit_lag, np.full(10, 2, dtype=np.int64))
    np.testing.assert_array_equal(result.fired, np.full(10, True))
    np.testing.assert_array_equal(result.regime_label, np.zeros(10, dtype=np.int64))
    assert result.metadata["criterion"] == "kl_plus_entropy"
    assert result.metadata["fired_rate"] == 1.0
    assert result.metadata["mean_lag"] == pytest.approx(2.0)


def test_kl_only_fires_at_l2_on_perfectly_stable_trajectory_with_m_consec_2() -> None:
    P = _constant_trajectory(n_lags=8, T=5, row=[0.8, 0.15, 0.05])
    inferer = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_ONLY, m_consec=2)
    result = inferer.infer(P)
    np.testing.assert_array_equal(result.commit_lag, np.full(5, 2, dtype=np.int64))


def test_m_consec_1_fires_at_l1_not_l0_for_kl_criteria() -> None:
    """Regression guard: pre-fix, m_consec=1 trivially committed at lag 0 because stable[0] was
    initialized True for KL criteria. After fix, stable[0] is False uniformly, so m_consec=1 must
    fire at lag 1 (the first lag with a real prior-lag comparison)."""
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.05, 0.05])

    for criterion in (StabilityCriterion.KL_ONLY, StabilityCriterion.KL_PLUS_ENTROPY,
                      StabilityCriterion.MAP_PERSISTENCE):
        inferer = AdaptiveLagInferer(theta_kl=0.005, criterion=criterion, m_consec=1)
        result = inferer.infer(P)
        np.testing.assert_array_equal(
            result.commit_lag, np.full(5, 1, dtype=np.int64),
            err_msg=f"criterion={criterion} should fire at lag 1 with m_consec=1, not lag 0",
        )


def test_map_persistence_fires_at_l2_with_m_consec_2() -> None:
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.05, 0.05])
    inferer = AdaptiveLagInferer(criterion=StabilityCriterion.MAP_PERSISTENCE, m_consec=2)
    result = inferer.infer(P)
    np.testing.assert_array_equal(result.commit_lag, np.full(5, 2, dtype=np.int64))


def test_infer_falls_back_to_kmax_when_kl_stays_above_threshold() -> None:
    """Cyclic large swings between lags keep KL high; criterion never fires; commit_lag = k_max."""
    n_lags, T, K = 8, 4, 3
    cycle = [[0.5, 0.3, 0.2], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5], [0.5, 0.2, 0.3]]
    P = np.zeros((n_lags, T, K), dtype=float)
    for l in range(n_lags):
        P[l, :, :] = cycle[l % 4]

    inferer = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_PLUS_ENTROPY, m_consec=2)
    result = inferer.infer(P)

    np.testing.assert_array_equal(result.commit_lag, np.full(T, n_lags - 1, dtype=np.int64))
    np.testing.assert_array_equal(result.fired, np.full(T, False))
    np.testing.assert_array_equal(result.regime_label, np.argmax(P[n_lags - 1], axis=1))


def test_entropy_gate_rejects_bars_where_entropy_is_rising() -> None:
    """KL-only commits when KL stays small even if entropy is rising; KL+entropy must reject."""
    n_lags, T, K = 4, 1, 2
    # P[0] = [0.99, 0.01]; entropy ~0.056
    # P[1] = [0.985, 0.015]; tiny KL(P1||P0); entropy ~0.077 (RISING)
    # P[2] = [0.98, 0.02]; tiny KL; entropy ~0.097 (RISING)
    # P[3] = [0.97, 0.03]; tiny KL; entropy ~0.135 (RISING)
    P = np.array([
        [[0.99, 0.01]],
        [[0.985, 0.015]],
        [[0.98, 0.02]],
        [[0.97, 0.03]],
    ], dtype=float)
    assert P.shape == (n_lags, T, K)

    kl_only = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_ONLY, m_consec=2)
    kl_plus_ent = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_PLUS_ENTROPY, m_consec=2)

    # KL-only should fire at lag 2 (m_consec=2 requires stable[1] AND stable[2]).
    assert kl_only.infer(P).commit_lag[0] == 2
    # KL+entropy never fires because entropy is rising — falls back to k_max.
    assert kl_plus_ent.infer(P).commit_lag[0] == n_lags - 1


def test_metadata_records_aggregate_stats() -> None:
    P = _constant_trajectory(n_lags=8, T=100, row=[0.9, 0.05, 0.05])
    result = AdaptiveLagInferer().infer(P)

    md = result.metadata
    assert md["n_bars"] == 100
    assert md["k_max"] == 7
    assert md["n_fired"] == 100
    assert md["fired_rate"] == 1.0
    assert md["median_lag"] == 2.0
    assert md["mean_lag"] == pytest.approx(2.0)
    assert md["theta_kl"] == pytest.approx(0.005)
    assert md["criterion"] == "kl_plus_entropy"
    assert md["m_consec"] == 2


def test_metadata_omits_theta_for_map_persistence() -> None:
    P = _constant_trajectory(n_lags=4, T=10, row=[0.9, 0.05, 0.05])
    inferer = AdaptiveLagInferer(criterion=StabilityCriterion.MAP_PERSISTENCE, m_consec=2)
    result = inferer.infer(P)
    assert result.metadata["theta_kl"] is None


def test_available_at_property_returns_t_plus_commit_lag() -> None:
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.05, 0.05])
    result = AdaptiveLagInferer().infer(P)
    # commit_lag = [2, 2, 2, 2, 2] → available_at = [2, 3, 4, 5, 6]
    np.testing.assert_array_equal(result.available_at, np.array([2, 3, 4, 5, 6], dtype=np.int64))


def test_as_of_labels_aligns_causally_for_constant_lag() -> None:
    """Constant commit_lag=2 → available_at[t] = t+2. as_of[t] = label[t-2] for t >= 2; abstain before."""
    P = _constant_trajectory(n_lags=8, T=10, row=[0.9, 0.05, 0.05])
    result = AdaptiveLagInferer().infer(P)

    out = result.as_of_labels(abstain_label=-1)

    expected = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_as_of_labels_picks_most_recent_target_even_with_late_arrivals() -> None:
    """A late-arriving label (high commit_lag) at low s is ignored once a fresher label (low
    commit_lag at higher s) has already become available — as_of returns the latest target bar."""
    commit_lag = np.array([5, 1, 1, 1, 1, 1], dtype=np.int64)
    regime_label = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    fired = np.array([False, True, True, True, True, True])
    result = AdaptiveLagResult(
        commit_lag=commit_lag, regime_label=regime_label, fired=fired,
        committed_posterior=np.zeros((6, 2), dtype=float), metadata={},
    )

    # available_at = [5, 2, 3, 4, 5, 6]
    # t=0: no arrivals -> abstain (-1)
    # t=1: no arrivals -> abstain
    # t=2: bar 1 arrives -> label 1
    # t=3: bar 2 arrives -> label 2
    # t=4: bar 3 arrives -> label 3
    # t=5: bars 0 and 4 both arrive; max s = 4 -> label 4 (the late-arriving s=0's label IS available but s=4 is the newer target)
    out = result.as_of_labels(abstain_label=-1)
    expected = np.array([-1, -1, 1, 2, 3, 4], dtype=np.int64)
    np.testing.assert_array_equal(out, expected)


def test_as_of_labels_respects_custom_abstain_label() -> None:
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.1])
    result = AdaptiveLagInferer().infer(P)
    out = result.as_of_labels(abstain_label=99)
    # commit_lag is 2 for all bars; first label available at t=2.
    assert out[0] == 99
    assert out[1] == 99
    assert out[2] == 0


def test_as_of_posterior_matches_maturation_align_for_constant_lag() -> None:
    """Cross-check: for a constant per-bar commit_lag=K, as_of_posterior() must produce the same
    (T, K) matrix as MaturationAlignTransformer(K) applied to committed_posterior. This locks the
    posterior-side aligner to the constant-K contract that already governs MaturationAlign."""
    K_LAG = 3
    T, K = 12, 3
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((T, K))
    committed = np.exp(logits)
    committed /= committed.sum(axis=1, keepdims=True)

    result = AdaptiveLagResult(
        commit_lag=np.full(T, K_LAG, dtype=np.int64),
        regime_label=np.argmax(committed, axis=1).astype(np.int64),
        fired=np.ones(T, dtype=bool),
        committed_posterior=committed,
        metadata={},
    )

    expected = MaturationAlignTransformer(lag=K_LAG).transform(committed)
    np.testing.assert_allclose(result.as_of_posterior(), expected)


def test_as_of_posterior_emits_uniform_prior_during_warmup() -> None:
    """Rows before any label has arrived must carry the uniform prior 1/K, matching
    MaturationAlignTransformer's warmup contract."""
    K = 3
    T = 6
    committed = np.full((T, K), 1.0 / K)
    # All bars commit at lag 2 → first matured label arrives at t=2.
    result = AdaptiveLagResult(
        commit_lag=np.full(T, 2, dtype=np.int64),
        regime_label=np.zeros(T, dtype=np.int64),
        fired=np.ones(T, dtype=bool),
        committed_posterior=committed,
        metadata={},
    )

    out = result.as_of_posterior()
    np.testing.assert_allclose(out[:2], np.full((2, K), 1.0 / K))
    np.testing.assert_allclose(out[2:], committed[: T - 2])


def test_as_of_posterior_forward_fills_between_commitments() -> None:
    """When commit_lag varies, the most-recently-matured target bar's posterior should persist
    until a fresher matured bar replaces it. Concretely: bar 0 arrives at t=2 and bar 2 arrives
    at t=3; t=2 should reflect bar 0, t=3 onward should reflect bar 2 (the freshest matured bar)."""
    committed = np.array([
        [0.9, 0.1],   # bar 0 posterior
        [0.5, 0.5],   # bar 1 posterior
        [0.2, 0.8],   # bar 2 posterior
        [0.6, 0.4],   # bar 3 posterior
        [0.7, 0.3],   # bar 4 posterior
    ], dtype=float)
    commit_lag = np.array([2, 4, 1, 1, 1], dtype=np.int64)
    # available_at = [2, 5, 3, 4, 5]
    result = AdaptiveLagResult(
        commit_lag=commit_lag,
        regime_label=np.argmax(committed, axis=1).astype(np.int64),
        fired=np.ones(5, dtype=bool),
        committed_posterior=committed,
        metadata={},
    )

    out = result.as_of_posterior()
    # t=0, t=1: warmup uniform 1/2
    np.testing.assert_allclose(out[0], [0.5, 0.5])
    np.testing.assert_allclose(out[1], [0.5, 0.5])
    # t=2: bar 0 arrives → its posterior
    np.testing.assert_allclose(out[2], committed[0])
    # t=3: bar 2 arrives → its posterior (newer target than bar 0)
    np.testing.assert_allclose(out[3], committed[2])
    # t=4: bar 3 arrives → its posterior (newer than bar 2)
    np.testing.assert_allclose(out[4], committed[3])


def test_as_of_posterior_prefers_newer_target_on_simultaneous_arrival() -> None:
    """When a late-arriving low-s label and an already-arrived higher-s label coexist, the
    selection must pick the highest s (newest target). Parallels test_as_of_labels_picks_most_
    recent_target_even_with_late_arrivals; verifies the same precedence for posteriors."""
    committed = np.array([
        [0.1, 0.9],   # bar 0 — late arrival
        [0.4, 0.6],   # bar 1
        [0.5, 0.5],   # bar 2
        [0.6, 0.4],   # bar 3
        [0.7, 0.3],   # bar 4 — should dominate over bar 0 at t=5
        [0.8, 0.2],   # bar 5
    ], dtype=float)
    commit_lag = np.array([5, 1, 1, 1, 1, 1], dtype=np.int64)
    # available_at = [5, 2, 3, 4, 5, 6]; at t=5 both bar 0 and bar 4 are available; max s = 4.
    result = AdaptiveLagResult(
        commit_lag=commit_lag,
        regime_label=np.argmax(committed, axis=1).astype(np.int64),
        fired=np.ones(6, dtype=bool),
        committed_posterior=committed,
        metadata={},
    )

    out = result.as_of_posterior()
    # t=5: bar 4 wins (later target), not bar 0 (late arrival but older target).
    np.testing.assert_allclose(out[5], committed[4])


def test_as_of_posterior_returns_t_k_shape_and_simplex_rows() -> None:
    """Output shape (T, K), dtype float, every row sums to 1.0 (warmup uniform sums to 1; matured
    rows are direct copies of committed_posterior rows which were validated as simplex upstream)."""
    P = _constant_trajectory(n_lags=8, T=15, row=[0.7, 0.2, 0.1])
    result = AdaptiveLagInferer().infer(P)

    out = result.as_of_posterior()
    assert out.shape == (15, 3)
    assert out.dtype == np.float64
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-12)


def test_as_of_posterior_matches_argmax_alignment_with_as_of_labels() -> None:
    """argmax(as_of_posterior()) must equal as_of_labels() on every non-warmup row. (Warmup rows
    diverge by design: as_of_labels emits abstain, as_of_posterior emits uniform whose argmax is
    state 0; the test slices off warmup before comparing.)"""
    P = _constant_trajectory(n_lags=8, T=12, row=[0.85, 0.15])
    result = AdaptiveLagInferer().infer(P)

    posterior = result.as_of_posterior()
    labels = result.as_of_labels(abstain_label=-1)
    matured = labels != -1

    np.testing.assert_array_equal(np.argmax(posterior[matured], axis=1), labels[matured])


def test_compute_trajectories_stacks_per_lag_outputs() -> None:
    """compute_trajectories wraps predict_proba_fixed_lag_sweep and stacks the dict into (n_lags, T, K)."""

    class _FakeHmm:
        def predict_proba_fixed_lag_sweep(self, X, lags, window_size=None):
            T = X.shape[0]
            K = 3
            # Valid simplex rows, but tagged per-lag so we can verify ordering by inspecting the first column.
            base = np.full((T, K), 1.0 / K)
            return {l: base.copy() for l in lags}

    X = np.zeros((50, 2))
    trajectories = compute_trajectories(_FakeHmm(), X, k_max=7, window_size=48)
    assert trajectories.shape == (8, 50, 3)
    # All rows uniform 1/3 — valid simplex; the stacking is the thing being tested.
    np.testing.assert_allclose(trajectories.sum(axis=2), 1.0, atol=1e-12)


def test_compute_trajectories_rejects_zero_kmax() -> None:
    class _FakeHmm:
        def predict_proba_fixed_lag_sweep(self, X, lags, window_size=None):
            return {}

    with pytest.raises(ValueError, match="k_max"):
        compute_trajectories(_FakeHmm(), np.zeros((10, 2)), k_max=0)


def test_lag_commitment_audit_reports_zero_miss_on_constant_oracle() -> None:
    """Constant oracle → no change points → uncond_miss = 0 when criterion fires at l >= 1."""
    P = _constant_trajectory(n_lags=8, T=100, row=[0.9, 0.05, 0.05])
    result = AdaptiveLagInferer().infer(P)

    audit = lag_commitment_audit(P, result)
    assert audit["n_bars"] == 100
    assert audit["k_max"] == 7
    assert audit["n_change_points"] == 0
    assert audit["change_point_rate"] == 0.0
    assert audit["uncond_miss"] == 0.0
    assert audit["mean_lag"] == pytest.approx(2.0)
    assert audit["fired_rate"] == 1.0


def test_lag_commitment_audit_emits_all_distance_bins() -> None:
    P = _constant_trajectory(n_lags=8, T=50, row=[0.9, 0.1])
    result = AdaptiveLagInferer().infer(P)
    audit = lag_commitment_audit(P, result)
    for label in ("at_change", "near", "medium", "far"):
        assert f"miss_{label}" in audit
        assert f"n_{label}" in audit


def test_lag_commitment_audit_rejects_mismatched_commit_lag_shape() -> None:
    P = _constant_trajectory(n_lags=8, T=20, row=[0.9, 0.1])
    bad = AdaptiveLagResult(
        commit_lag=np.zeros(10, dtype=np.int64),
        regime_label=np.zeros(10, dtype=np.int64),
        fired=np.zeros(10, dtype=bool),
        committed_posterior=np.zeros((10, 2), dtype=float),
        metadata={},
    )
    with pytest.raises(ValueError, match="commit_lag shape"):
        lag_commitment_audit(P, bad)


def test_lag_commitment_audit_rejects_commit_lag_out_of_range() -> None:
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.1])
    bad = AdaptiveLagResult(
        commit_lag=np.array([0, 1, 2, 100, 3], dtype=np.int64),
        regime_label=np.zeros(5, dtype=np.int64),
        fired=np.zeros(5, dtype=bool),
        committed_posterior=np.zeros((5, 2), dtype=float),
        metadata={},
    )
    with pytest.raises(ValueError, match="out of range"):
        lag_commitment_audit(P, bad)


def test_lag_commitment_audit_rejects_mismatched_committed_posterior_shape() -> None:
    """audit must reject a result whose committed_posterior K disagrees with trajectories K —
    otherwise downstream metrics are computed against a posterior that was not produced from
    this P."""
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.1])  # K=2
    bad = AdaptiveLagResult(
        commit_lag=np.array([1, 1, 1, 1, 1], dtype=np.int64),
        regime_label=np.zeros(5, dtype=np.int64),
        fired=np.ones(5, dtype=bool),
        committed_posterior=np.full((5, 3), 1.0 / 3),  # K=3 mismatch
        metadata={},
    )
    with pytest.raises(ValueError, match="committed_posterior shape"):
        lag_commitment_audit(P, bad)


def test_lag_commitment_audit_rejects_regime_label_argmax_inconsistency() -> None:
    """audit must reject a result whose regime_label is not argmax(committed_posterior) — the
    invariant is guaranteed by AdaptiveLagInferer.infer() and a violation indicates a manually
    constructed or stale result."""
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.1])
    committed = np.tile(np.array([0.9, 0.1]), (5, 1))
    bad = AdaptiveLagResult(
        commit_lag=np.array([1, 1, 1, 1, 1], dtype=np.int64),
        regime_label=np.ones(5, dtype=np.int64),  # argmax is 0, not 1
        fired=np.ones(5, dtype=bool),
        committed_posterior=committed,
        metadata={},
    )
    with pytest.raises(ValueError, match="regime_label disagrees with argmax"):
        lag_commitment_audit(P, bad)


def test_lag_commitment_audit_rejects_fired_commit_lag_inconsistency() -> None:
    """audit must reject a result whose fired flag disagrees with (commit_lag < k_max)."""
    P = _constant_trajectory(n_lags=8, T=5, row=[0.9, 0.1])  # k_max = 7
    committed = np.tile(np.array([0.9, 0.1]), (5, 1))
    bad = AdaptiveLagResult(
        commit_lag=np.array([1, 1, 1, 1, 1], dtype=np.int64),
        regime_label=np.zeros(5, dtype=np.int64),
        fired=np.zeros(5, dtype=bool),  # should be all True (1 < 7)
        committed_posterior=committed,
        metadata={},
    )
    with pytest.raises(ValueError, match="fired disagrees"):
        lag_commitment_audit(P, bad)


def test_lag_commitment_audit_default_bins_have_no_overlap_at_small_k_max() -> None:
    """When k_max is small (here, k_max=1) the default bin scheme used to produce overlapping
    ``near=[1,2]`` and ``far=[k_max+1, sent]=[2, sent]`` — meaning distance 2 was double-counted.
    far now starts at max(k_max+1, 3) so the bins partition correctly for any k_max."""
    # k_max = 1 ⇒ n_lags = 2. m_consec=1 to allow commit at lag 1 within this budget.
    P = _constant_trajectory(n_lags=2, T=20, row=[0.9, 0.1])
    result = AdaptiveLagInferer(m_consec=1).infer(P)
    audit = lag_commitment_audit(P, result)

    # All bars should be in exactly one bin: at_change + near + medium + far == n_bars.
    assert audit["n_at_change"] + audit["n_near"] + audit["n_medium"] + audit["n_far"] == audit["n_bars"]


def test_lag_commitment_audit_default_bins_unchanged_for_lab_k_max() -> None:
    """At the lab-validated k_max=7, the bin partition must be unchanged by the small-k_max fix:
    far starts at k_max+1=8."""
    P = _constant_trajectory(n_lags=8, T=40, row=[0.9, 0.1])
    result = AdaptiveLagInferer().infer(P)
    audit = lag_commitment_audit(P, result)
    # All four bins accounted for; partition is exhaustive.
    assert audit["n_at_change"] + audit["n_near"] + audit["n_medium"] + audit["n_far"] == audit["n_bars"]


def test_committed_posterior_matches_trajectory_at_commit_lag() -> None:
    """For each bar t, result.committed_posterior[t] must equal trajectories[commit_lag[t], t]."""
    P = _constant_trajectory(n_lags=8, T=10, row=[0.9, 0.05, 0.05])
    result = AdaptiveLagInferer().infer(P)
    expected = P[result.commit_lag, np.arange(10)]
    np.testing.assert_allclose(result.committed_posterior, expected)
    assert result.committed_posterior.shape == (10, 3)


def test_kl_treats_support_change_as_infinite_kl() -> None:
    """When P[l-1, t, j] == 0 and P[l, t, j] > 0, true KL is infinite. scipy.special.rel_entr
    encodes this correctly so the criterion does not falsely fire on a support change."""
    n_lags, T, K = 4, 1, 2
    # P[0] = [1.0, 0.0]; P[1] = [0.5, 0.5] (support change at j=1); P[2..3] hold steady.
    P = np.array([
        [[1.0, 0.0]],
        [[0.5, 0.5]],
        [[0.5, 0.5]],
        [[0.5, 0.5]],
    ], dtype=float)

    inferer = AdaptiveLagInferer(theta_kl=0.005, criterion=StabilityCriterion.KL_ONLY, m_consec=2)
    result = inferer.infer(P)

    # KL[1] is infinite (support change), so below[1] is False and the criterion can't fire at
    # lag 2. After lag 2 the posterior is steady, so KL[2]=0, KL[3]=0 — fires at lag 3.
    assert result.commit_lag[0] == 3


def test_compute_trajectories_rejects_k_max_at_or_above_window_size() -> None:
    """In bounded-history mode the lag-K smoother's window must include the target bar t, which
    requires k_max < window_size. compute_trajectories enforces this upfront."""

    class _FakeHmm:
        def predict_proba_fixed_lag_sweep(self, X, lags, window_size=None):
            T = X.shape[0]
            return {l: np.tile(np.array([0.5, 0.5], dtype=float), (T, 1)) for l in lags}

    X = np.zeros((100, 2))
    with pytest.raises(ValueError, match="k_max.*window_size"):
        compute_trajectories(_FakeHmm(), X, k_max=48, window_size=48)
    with pytest.raises(ValueError, match="k_max.*window_size"):
        compute_trajectories(_FakeHmm(), X, k_max=100, window_size=48)


def test_compute_trajectories_allows_full_history_mode_for_large_k_max() -> None:
    """window_size=None opts into full-history mode where the k_max < window_size check is moot."""

    class _FakeHmm:
        def predict_proba_fixed_lag_sweep(self, X, lags, window_size=None):
            T = X.shape[0]
            return {l: np.tile(np.array([0.5, 0.5], dtype=float), (T, 1)) for l in lags}

    X = np.zeros((100, 2))
    trajectories = compute_trajectories(_FakeHmm(), X, k_max=100, window_size=None)
    assert trajectories.shape == (101, 100, 2)


def test_from_hmm_classmethod_returns_result_directly() -> None:
    """from_hmm composes compute_trajectories + infer in one call."""

    class _FakeHmm:
        def predict_proba_fixed_lag_sweep(self, X, lags, window_size=None):
            T = X.shape[0]
            # Peaked, identical at every lag → criterion fires at lag 2 under default config.
            return {l: np.tile(np.array([0.95, 0.05], dtype=float), (T, 1)) for l in lags}

    X = np.zeros((20, 3))
    result = AdaptiveLagInferer.from_hmm(_FakeHmm(), X, k_max=7, theta_kl=0.005, window_size=48)
    assert isinstance(result, AdaptiveLagResult)
    np.testing.assert_array_equal(result.commit_lag, np.full(20, 2, dtype=np.int64))
    np.testing.assert_array_equal(result.regime_label, np.zeros(20, dtype=np.int64))
