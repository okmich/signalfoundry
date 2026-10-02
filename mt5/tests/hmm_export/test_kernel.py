"""The MQL5 kernel renormalises every bar; these tests prove that is neutral.

``HmmCore.mqh`` subtracts ``logsumexp`` (FILTERING) or the row max
(CAUSAL_VITERBI) from the frontier on every step so the scores stay bounded
over a long history. Both shifts are claimed to be exactly neutral. The claim
is checked here against the batch recursions the repo already trusts:
``BasePomegranateHMM._forward_pass`` and ``_max_product_forward_pass``.

Prefix consistency is checked too, because it is the property that makes
CAUSAL_VITERBI tradable at all.
"""

import numpy as np
import pytest
from okmich_quant_ml.hmm.base_pomegranate import BasePomegranateHMM

from okmich_quant_mt5.hmm_export import LABEL_NONE, Mql5HmmKernel, Mql5InferenceMode

MODES = [Mql5InferenceMode.FILTERING, Mql5InferenceMode.CAUSAL_VITERBI]


def batch_log_b(params, observations):
    return np.stack([params.log_emissions(params.standardise(x)) for x in observations])


def streaming_scores(params, mode, observations):
    """Frontier after each step, exactly as the MQL5 kernel holds it."""
    kernel = Mql5HmmKernel(params, mode)
    out = []
    for x in observations:
        assert kernel.step(x)
        out.append(kernel.score)
    return np.stack(out)


class TestStreamingMatchesBatch:
    def test_filtering_reproduces_forward_pass_posteriors(self, params, observations):
        log_b = batch_log_b(params, observations)
        log_alpha = BasePomegranateHMM._forward_pass(params.log_pi, params.log_a, log_b)
        expected = np.exp(log_alpha - np.logaddexp.reduce(log_alpha, axis=1, keepdims=True))

        actual = np.exp(streaming_scores(params, Mql5InferenceMode.FILTERING, observations))

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=1e-12)

    def test_causal_viterbi_reproduces_max_product_frontier(self, params, observations):
        log_b = batch_log_b(params, observations)
        log_delta = BasePomegranateHMM._max_product_forward_pass(params.log_pi, params.log_a, log_b)
        expected = log_delta - log_delta.max(axis=1, keepdims=True)

        actual = streaming_scores(params, Mql5InferenceMode.CAUSAL_VITERBI, observations)

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_causal_viterbi_labels_are_identical(self, params, observations):
        log_b = batch_log_b(params, observations)
        log_delta = BasePomegranateHMM._max_product_forward_pass(params.log_pi, params.log_a, log_b)

        labels, _ = Mql5HmmKernel(params, Mql5InferenceMode.CAUSAL_VITERBI).run(observations)

        np.testing.assert_array_equal(labels, np.argmax(log_delta, axis=1))

    def test_per_bar_predictive_loglik_is_the_discarded_constant(self, params, observations):
        """The shift removed each step IS log P(o_t | o_0..o_{t-1})."""
        log_b = batch_log_b(params, observations)
        log_alpha = BasePomegranateHMM._forward_pass(params.log_pi, params.log_a, log_b)
        cumulative = np.logaddexp.reduce(log_alpha, axis=1)
        expected = np.concatenate([[cumulative[0]], np.diff(cumulative)])

        kernel = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING)
        actual = []
        for x in observations:
            kernel.step(x)
            actual.append(kernel.last_predictive_loglik)

        np.testing.assert_allclose(actual, expected, rtol=1e-10)
        assert np.isclose(np.sum(actual), cumulative[-1], rtol=1e-12), "shifts must sum to the joint log-likelihood"


class TestCausality:
    @pytest.mark.parametrize("mode", MODES)
    def test_prefix_consistency(self, params, observations, mode):
        """run(X[:t+1])[-1] == run(X)[t] for every t - no label is ever rewritten."""
        full, _ = Mql5HmmKernel(params, mode).run(observations)

        for t in (0, 1, 7, 33, 150, 399):
            prefix, _ = Mql5HmmKernel(params, mode).run(observations[: t + 1])
            assert prefix[-1] == full[t], f"{mode} label at bar {t} changed when later bars were revealed"

    @pytest.mark.parametrize("mode", MODES)
    def test_extending_history_leaves_emitted_labels_untouched(self, params, observations, mode):
        short, _ = Mql5HmmKernel(params, mode).run(observations[:200])
        long, _ = Mql5HmmKernel(params, mode).run(observations)

        np.testing.assert_array_equal(short, long[:200])

    @pytest.mark.parametrize("mode", MODES)
    def test_future_bars_cannot_change_the_present(self, params, observations, mode):
        """Perturbing bar t+1 onward must not touch any label at or before t."""
        rng = np.random.default_rng(4)
        tampered = observations.copy()
        tampered[250:] += rng.normal(0.0, 5.0, tampered[250:].shape)

        base, _ = Mql5HmmKernel(params, mode).run(observations)
        other, _ = Mql5HmmKernel(params, mode).run(tampered)

        np.testing.assert_array_equal(base[:250], other[:250])


class TestDecoderBehaviour:
    def test_causal_viterbi_dwells_at_least_as_long_as_filtering(self, params, observations):
        """Taking max over prefixes makes a challenger pay log(a_ii / a_ij) to take over."""
        filtered, _ = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(observations)
        viterbi, _ = Mql5HmmKernel(params, Mql5InferenceMode.CAUSAL_VITERBI).run(observations)

        assert np.sum(np.diff(viterbi) != 0) <= np.sum(np.diff(filtered) != 0)

    def test_filtering_confidence_is_the_max_posterior(self, params, observations):
        kernel = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING)
        for x in observations[:50]:
            kernel.step(x)
            posterior = kernel.posterior()
            assert kernel.confidence() == pytest.approx(posterior.max())
            assert 1.0 / params.n_states - 1e-12 <= kernel.confidence() <= 1.0 + 1e-12

    def test_viterbi_confidence_is_a_non_negative_margin(self, params, observations):
        kernel = Mql5HmmKernel(params, Mql5InferenceMode.CAUSAL_VITERBI)
        for x in observations[:50]:
            kernel.step(x)
            ordered = np.sort(kernel.score)[::-1]
            assert kernel.confidence() == pytest.approx(ordered[0] - ordered[1])
            assert kernel.confidence() >= 0.0

    def test_posterior_is_refused_for_viterbi(self, params, observations):
        kernel = Mql5HmmKernel(params, Mql5InferenceMode.CAUSAL_VITERBI)
        kernel.step(observations[0])
        with pytest.raises(ValueError, match="scores paths, not states"):
            kernel.posterior()

    def test_recovers_the_generating_states(self, params, observations):
        """Sanity: on data drawn from the model, the decoder should mostly be right."""
        labels, _ = Mql5HmmKernel(params, Mql5InferenceMode.CAUSAL_VITERBI).run(observations)
        nearest = np.argmin(np.linalg.norm(observations[:, None, :] - params.mu[None, :, :], axis=2), axis=1)
        assert np.mean(labels == nearest) > 0.7


class TestRunSemantics:
    def test_burn_in_suppresses_labels_without_stalling_the_recursion(self, params, observations):
        labels, conf = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(observations, burn_in=100)

        assert np.all(labels[:100] == LABEL_NONE)
        assert np.all(np.isnan(conf[:100]))
        assert np.all(labels[100:] != LABEL_NONE)

        unburned, _ = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(observations)
        np.testing.assert_array_equal(labels[100:], unburned[100:]), "burn-in must hide labels, not change them"

    def test_non_finite_rows_are_skipped_without_advancing(self, params, observations):
        gapped = observations.copy()
        gapped[10] = np.nan

        labels, _ = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(gapped)
        assert labels[10] == LABEL_NONE

        without = np.delete(observations, 10, axis=0)
        reference, _ = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(without)
        np.testing.assert_array_equal(labels[11:], reference[10:])

    def test_rejects_wrong_feature_count(self, params):
        with pytest.raises(ValueError, match=r"features must be \(T, 2\)"):
            Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(np.zeros((10, 3)))

    def test_infeasible_path_is_reported_not_silently_labelled(self, params, observations):
        """An observation far enough out drives every emission to -inf; argmax would tie on 0."""
        broken = observations.copy()
        broken[5] = 1e200

        with pytest.raises(ValueError, match="no feasible state path at bar 5"):
            Mql5HmmKernel(params, Mql5InferenceMode.FILTERING).run(broken)

    def test_reset_returns_the_kernel_to_its_initial_state(self, params, observations):
        kernel = Mql5HmmKernel(params, Mql5InferenceMode.FILTERING)
        first, _ = kernel.run(observations)
        second, _ = kernel.run(observations)
        np.testing.assert_array_equal(first, second)
