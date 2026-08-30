"""Core HierarchicalHMM tests: topology, recovery, posteriors, modes, labels, variants, sessions."""
from datetime import time

import numpy as np
import pytest

from okmich_quant_ml.hmm.hierarchical import (
    ALPHABET_SIZE,
    HHMMLevel,
    HierarchicalHMM,
    MacroRegime,
    PosteriorMode,
    SessionPolicy,
    ZigzagDirection,
    build_transition_mask,
    state_block,
    state_direction,
)
from okmich_quant_ml.hmm.hierarchical import from_param_dict, to_param_dict
from okmich_quant_ml.hmm.hierarchical.config import symbols_for_direction
from okmich_quant_ml.hmm.hierarchical.hhmm import _BLOCK_STATES
from okmich_quant_ml.hmm.util import DistType

from .conftest import TRUE_EDGES, TRUE_EMISSIONS, _generate_stream


class TestTopology:
    def test_mask_forbids_same_direction_and_self_loops(self):
        mask = build_transition_mask()
        assert mask.shape == (4, 4)
        for s in range(4):
            assert mask[s, s] == 0.0  # no self-loops
            for t in range(4):
                # allowed iff the transition flips direction
                assert bool(mask[s, t]) == (s % 2 != t % 2)

    def test_state_helpers(self):
        assert [state_block(s) for s in range(4)] == [0, 0, 1, 1]
        assert state_direction(0) is ZigzagDirection.UP
        assert state_direction(1) is ZigzagDirection.DOWN

    def test_default_distribution_is_categorical(self):
        # The module exists to implement the canonical 18-symbol model, and param-dict persistence
        # only supports categorical, so the no-arg default must be CATEGORICAL.
        model = HierarchicalHMM()
        assert model.distribution_type is DistType.CATEGORICAL
        assert model.is_categorical


class TestStructuralInvariants:
    def test_transition_structural_zeros_preserved(self, fitted_hhmm):
        learned = fitted_hhmm.transition_prob()
        mask = build_transition_mask()
        assert learned[mask == 0].max() < 1e-6

    def test_emission_direction_pinning_preserved(self, fitted_hhmm):
        for s in range(4):
            probs = fitted_hhmm._emission_probs(s)
            wrong = list(symbols_for_direction(
                ZigzagDirection.DOWN if state_direction(s) is ZigzagDirection.UP else ZigzagDirection.UP))
            assert probs[wrong].max() < 1e-6


class TestPosteriors:
    def test_shapes_and_normalisation(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"]
        prod = fitted_hhmm.predict_proba(X, HHMMLevel.PRODUCTION, PosteriorMode.FILTER)
        macro = fitted_hhmm.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.FILTER)
        assert prod.shape == (len(X), 4)
        assert macro.shape == (len(X), 2)
        assert np.allclose(prod.sum(1), 1.0, atol=1e-6)
        assert np.allclose(macro.sum(1), 1.0, atol=1e-6)

    def test_macro_is_block_sum_of_production(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"]
        prod = fitted_hhmm.predict_proba(X, HHMMLevel.PRODUCTION, PosteriorMode.FILTER)
        macro = fitted_hhmm.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.FILTER)
        run_block = fitted_hhmm.macro_block(MacroRegime.RUN)
        rev_block = fitted_hhmm.macro_block(MacroRegime.REVERSAL)
        assert np.allclose(macro[:, 0], prod[:, list(_BLOCK_STATES[run_block])].sum(1), atol=1e-9)
        assert np.allclose(macro[:, 1], prod[:, list(_BLOCK_STATES[rev_block])].sum(1), atol=1e-9)

    @pytest.mark.parametrize("mode", list(PosteriorMode))
    def test_all_modes_valid(self, fitted_hhmm, synthetic_stream, mode):
        X = synthetic_stream["symbols"]
        lag = 5 if mode is PosteriorMode.FIXED_LAG else 0
        macro = fitted_hhmm.predict_proba(X, HHMMLevel.MACRO, mode, lag=lag)
        assert macro.shape == (len(X), 2)
        assert np.allclose(macro.sum(1), 1.0, atol=1e-6)

    def test_predict_before_fit_raises(self, synthetic_stream):
        model = HierarchicalHMM()
        with pytest.raises(RuntimeError):
            model.predict_proba(synthetic_stream["symbols"])

    def test_bad_symbols_rejected(self, fitted_hhmm):
        with pytest.raises(ValueError):
            fitted_hhmm.predict_proba(np.array([0, 5, ALPHABET_SIZE + 3]))

    def test_empty_input_raises_clear_error(self, fitted_hhmm):
        # An empty sequence must raise a clear ValueError, not a bare NumPy reduction error.
        with pytest.raises(ValueError, match="empty"):
            fitted_hhmm.predict_proba(np.array([], dtype=np.int64))


class TestCausality:
    """FILTER must be causal (prefix-stable); SMOOTHER is look-ahead. This is the live-eligibility
    guarantee for the whole causal-first design, so it is asserted directly."""

    def test_filter_is_prefix_stable(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"]
        t = len(X) * 2 // 3
        for level in (HHMMLevel.MACRO, HHMMLevel.PRODUCTION):
            full = fitted_hhmm.predict_proba(X, level, PosteriorMode.FILTER)
            prefix = fitted_hhmm.predict_proba(X[:t], level, PosteriorMode.FILTER)
            # Appending future zigzags must not change any already-emitted causal posterior.
            assert np.allclose(full[:t], prefix, atol=1e-9), f"FILTER leaked future info at level {level}"

    def test_fixed_lag_is_prefix_stable_up_to_lag(self, fitted_hhmm, synthetic_stream):
        # Fixed-lag freezes row t once t+lag observations exist, so rows <= t-lag are prefix-stable.
        X = synthetic_stream["symbols"]
        t, lag = 600, 5
        full = fitted_hhmm.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.FIXED_LAG, lag=lag)
        prefix = fitted_hhmm.predict_proba(X[:t], HHMMLevel.MACRO, PosteriorMode.FIXED_LAG, lag=lag)
        assert np.allclose(full[: t - lag], prefix[: t - lag], atol=1e-9)

    def test_smoother_uses_future(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"]
        t = len(X) * 2 // 3
        full = fitted_hhmm.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.SMOOTHER)
        prefix = fitted_hhmm.predict_proba(X[:t], HHMMLevel.MACRO, PosteriorMode.SMOOTHER)
        # Smoothing is non-causal: earlier rows shift once future data is available.
        assert not np.allclose(full[:t], prefix, atol=1e-6)


class TestParameterRecovery:
    def test_run_reversal_separation(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        true_run = synthetic_stream["block"] == 0
        run_p = macro[:, 0]
        sep = run_p[true_run].mean() - run_p[~true_run].mean()
        assert sep > 0.5  # strong Run/Reversal separation

    def test_run_block_has_higher_magnitude(self, fitted_hhmm, synthetic_stream):
        run_block = fitted_hhmm.macro_block(MacroRegime.RUN)
        rev_block = fitted_hhmm.macro_block(MacroRegime.REVERSAL)
        smoothed = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.PRODUCTION, PosteriorMode.SMOOTHER)
        states = smoothed.argmax(1)
        mags = synthetic_stream["magnitudes"]
        run_mag = mags[np.isin(states, _BLOCK_STATES[run_block])].mean()
        rev_mag = mags[np.isin(states, _BLOCK_STATES[rev_block])].mean()
        assert run_mag > rev_mag


@pytest.mark.slow
class TestRegimeIdentityStability:
    """The structural mask + emission pinning give stable Run/Reversal identity across noise."""

    def test_identity_stable_across_seeds(self):
        seps = []
        for seed in (21, 22, 23):
            stream = _generate_stream(1000, seed=seed)
            model = HierarchicalHMM(DistType.CATEGORICAL, n_init=1, tol=1e-3, random_state=seed, max_iter=35)
            model.fit(stream["symbols"], magnitudes=stream["magnitudes"])
            # direction identity is pinned every time (state 0 always emits up-symbols)
            assert model._emission_probs(0)[list(symbols_for_direction(ZigzagDirection.DOWN))].max() < 1e-6
            macro = model.predict_proba(stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
            true_run = stream["block"] == 0
            seps.append(macro[:, 0][true_run].mean() - macro[:, 0][~true_run].mean())
        # separation is consistently strong — no regime-identity flips across realizations
        assert min(seps) > 0.4


@pytest.mark.slow
class TestMacroLabelPersistence:
    def test_hungarian_match_keeps_identity(self, synthetic_stream):
        s1 = _generate_stream(1200, seed=31)
        s2 = _generate_stream(1200, seed=32)
        m1 = HierarchicalHMM(DistType.CATEGORICAL, n_init=1, tol=1e-3, random_state=1, max_iter=35).fit(s1["symbols"], magnitudes=s1["magnitudes"])
        # Refit on new data, matched to m1: Run block emissions should stay closest to m1's Run.
        m2 = HierarchicalHMM(DistType.CATEGORICAL, n_init=1, tol=1e-3, random_state=99, max_iter=35).fit(s2["symbols"], magnitudes=s2["magnitudes"], prev=m1)
        run_emis_1 = np.concatenate([m1._emission_probs(s) for s in _BLOCK_STATES[m1.macro_block(MacroRegime.RUN)]])
        run_emis_2 = np.concatenate([m2._emission_probs(s) for s in _BLOCK_STATES[m2.macro_block(MacroRegime.RUN)]])
        rev_emis_2 = np.concatenate([m2._emission_probs(s) for s in _BLOCK_STATES[m2.macro_block(MacroRegime.REVERSAL)]])
        # matched Run must be closer to m1's Run than the matched Reversal is
        assert np.abs(run_emis_1 - run_emis_2).sum() < np.abs(run_emis_1 - rev_emis_2).sum()

    def test_continuous_prev_refit_matches_via_means(self):
        # Regression: prev-matching used categorical `.probs`; a continuous refit must not crash and
        # must produce a valid Run/Reversal labelling (matched on emission means, not probs).
        def gen(seed):
            r = np.random.default_rng(seed)
            n = 1000
            st = np.empty(n, int)
            st[0] = 0
            for t in range(1, n):
                st[t] = r.choice(4, p=TRUE_EDGES[st[t - 1]])
            means = np.array([[2.0, 2.0], [2.0, -2.0], [0.3, 0.3], [0.3, -0.3]])
            return means[st] + r.normal(0, 0.5, (n, 2)), np.where(st // 2 == 0, 0.004, 0.001)

        X1, mag1 = gen(41)
        X2, mag2 = gen(42)
        a = HierarchicalHMM(DistType.NORMAL, covariance_type="diag", n_init=1, random_state=1, max_iter=40).fit(X1, magnitudes=mag1)
        b = HierarchicalHMM(DistType.NORMAL, covariance_type="diag", n_init=1, random_state=2, max_iter=40).fit(X2, magnitudes=mag2, prev=a)
        assert set(b.macro_labels_.values()) == {MacroRegime.RUN, MacroRegime.REVERSAL}


class TestEmissionVariants:
    def test_categorical_mixture_rejected(self):
        # A mixture of categoricals over the same alphabet is non-identifiable -> rejected at build.
        with pytest.raises(ValueError):
            HierarchicalHMM(DistType.CATEGORICAL, n_components=2)

    @pytest.mark.slow
    def test_continuous_mixture_fits(self):
        # Mixtures are meaningful for continuous emissions (multi-modal within a regime).
        rng = np.random.default_rng(8)
        n = 1500
        states = np.empty(n, int)
        states[0] = 0
        for t in range(1, n):
            states[t] = rng.choice(4, p=TRUE_EDGES[states[t - 1]])
        means = np.array([[2.0, 2.0], [2.0, -2.0], [0.3, 0.3], [0.3, -0.3]])
        X = means[states] + rng.normal(0, 0.5, size=(n, 2))
        mags = np.where(states // 2 == 0, 0.004, 0.001)
        model = HierarchicalHMM(DistType.NORMAL, n_components=2, covariance_type="diag", random_state=6, max_iter=50)
        model.fit(X, magnitudes=mags)
        macro = model.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.FILTER)
        assert macro.shape == (n, 2)
        assert np.allclose(macro.sum(1), 1.0, atol=1e-6)
        assert model.transition_prob()[build_transition_mask() == 0].max() < 1e-6

    @pytest.mark.slow
    def test_continuous_normal_emissions(self):
        # 2D continuous per-zigzag features; Run states (block 0) have larger mean.
        rng = np.random.default_rng(3)
        n = 1500
        states = np.empty(n, int)
        states[0] = 0
        for t in range(1, n):
            states[t] = rng.choice(4, p=TRUE_EDGES[states[t - 1]])
        means = np.array([[2.0, 2.0], [2.0, -2.0], [0.3, 0.3], [0.3, -0.3]])
        X = means[states] + rng.normal(0, 0.6, size=(n, 2))
        mags = np.where(states // 2 == 0, 0.004, 0.001)
        model = HierarchicalHMM(DistType.NORMAL, covariance_type="diag", random_state=5, max_iter=60)
        model.fit(X, magnitudes=mags)
        macro = model.predict_proba(X, HHMMLevel.MACRO, PosteriorMode.FILTER)
        assert macro.shape == (n, 2)
        assert np.allclose(macro.sum(1), 1.0, atol=1e-6)
        # structural transition zeros hold for the continuous variant too
        assert model.transition_prob()[build_transition_mask() == 0].max() < 1e-6

    def test_continuous_rejects_zigzag_observations(self, observations):
        model = HierarchicalHMM(DistType.NORMAL)
        with pytest.raises(TypeError):
            model.fit(observations)

    def test_mean_norm_handles_mixture_emissions(self):
        # Macro labeling fallback (no magnitudes) must not treat a mixture emission as mean-less;
        # _mean_norm averages the component mean norms rather than returning None.
        import torch
        from pomegranate.distributions import Normal
        from pomegranate.gmm import GeneralMixtureModel
        mix = GeneralMixtureModel([
            Normal(means=[3.0, 4.0], covs=[1.0, 1.0], covariance_type="diag", dtype=torch.float64),
            Normal(means=[0.0, 0.0], covs=[1.0, 1.0], covariance_type="diag", dtype=torch.float64),
        ])
        assert HierarchicalHMM._mean_norm(mix) == pytest.approx((5.0 + 0.0) / 2)


class TestSessionPolicy:
    def test_soft_downweight_lowers_confidence(self, fitted_hhmm):
        # Two zigzags: one inside a 22:00-06:00 low-liquidity window, one outside.
        from okmich_quant_ml.hmm.hierarchical.observations import ZigzagObservations, Zigzag
        import pandas as pd
        times = np.array([np.datetime64("2026-01-05T02:00"), np.datetime64("2026-01-05T12:00")],
                         dtype="datetime64[ns]")
        symbols = np.array([15, 6])  # strong up / strong down
        zz = [Zigzag(0, ZigzagDirection.UP, 0, 1, 100.0, 101.0, 0.01, 1, pd.Timestamp(times[0]), pd.Timestamp(times[0])),
              Zigzag(1, ZigzagDirection.DOWN, 1, 2, 101.0, 100.0, 0.01, 2, pd.Timestamp(times[1]), pd.Timestamp(times[1]))]
        obs = ZigzagObservations(zz, symbols, np.array([1, 0]), np.array([2, 2]), np.array([1, 1]),
                                 np.array([0.01, 0.01]), times)
        # Work on an independent reload so the session-scoped fixture is never mutated.
        model = from_param_dict(to_param_dict(fitted_hhmm))
        model.attach_session_policy(SessionPolicy.SOFT, low_liquidity_windows=((time(22, 0), time(6, 0)),),
                                    soft_downweight=0.3)
        gated = model.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=True)
        plain = model.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=False)
        # the flagged (first) obs should have its confidence pulled toward 0.5
        assert abs(gated[0].max() - 0.5) < abs(plain[0].max() - 0.5)
        # the unflagged (second) obs is unchanged
        assert np.allclose(gated[1], plain[1], atol=1e-9)

    def test_hard_segmentation_changes_filter(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"][:200]
        base = fitted_hhmm._segment_posterior(X, PosteriorMode.FILTER, 0)
        seg = fitted_hhmm.predict_proba(X, HHMMLevel.PRODUCTION, PosteriorMode.FILTER, session_breaks=[100])
        # segmentation restarts the forward pass at 100, so the row there differs from continuous filtering
        assert not np.allclose(base[100], seg[100], atol=1e-6)
        assert np.allclose(base[:100], seg[:100], atol=1e-9)  # first segment unchanged


class TestModelSelectionMetrics:
    def test_aic_bic_finite(self, fitted_hhmm, synthetic_stream):
        aic, bic = fitted_hhmm.get_aic_bic(synthetic_stream["symbols"])
        assert np.isfinite(aic) and np.isfinite(bic)
        assert bic > aic  # BIC penalises harder for T >> params
