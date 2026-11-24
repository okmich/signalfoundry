"""BasePomegranateHMM: sequence ``lengths`` validation, unequal-length fitting, and numpy ``state_posteriors``.

Regressions from the 2026-11-24 review: lengths that did not cover X silently dropped rows; a one-observation
sequence crashed pomegranate with a reshape error; restarts were scored on a ragged batch (unequal lengths crashed);
``state_posteriors`` returned a torch Tensor.
"""
import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, PomegranateHMM


def _X(n: int = 110) -> np.ndarray:
    return np.random.default_rng(0).normal(size=(n, 2))


def test_unequal_sequence_lengths_fit():
    model = PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=10).fit(_X(), lengths=[50, 60])
    assert model._fitted_log_prob(model._model, [_X()[:50], _X()[50:]]) == pytest.approx(
        sum(model._fitted_log_prob(model._model, [x]) for x in (_X()[:50], _X()[50:])))


def test_lengths_must_cover_the_data():
    with pytest.raises(ValueError, match="sum to the 110 rows"):
        PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=5).fit(_X(), lengths=[50])
    with pytest.raises(ValueError, match="positive"):
        PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=5).fit(_X(), lengths=[0, 110])


def test_fitting_sequences_need_two_observations():
    with pytest.raises(ValueError, match="fewer than 2 observations"):
        PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=5).fit(_X(), lengths=[109, 1])


def test_state_posteriors_is_a_numpy_array():
    model = PomegranateHMM(DistType.NORMAL, n_states=2, max_iter=5).fit(_X())
    post = model.state_posteriors(_X(20))
    assert isinstance(post, np.ndarray) and post.shape == (20, 2)
    np.testing.assert_allclose(post.sum(axis=1), 1.0, atol=1e-6)
