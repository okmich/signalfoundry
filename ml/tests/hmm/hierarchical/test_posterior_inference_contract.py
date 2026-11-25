"""HHMM posteriors flow into the posterior_inference stack unchanged: leaf posteriors (the duck-typed surface) and level
posteriors are (T, K) simplices that validators, inferers, transformers and the pipeline accept."""
import numpy as np
import pytest

from okmich_quant_ml.hmm.hierarchical import HHMM, GaussianEmission, StateNode
from okmich_quant_ml.posterior_inference import (ArgmaxInferer, EmaPosteriorTransformer, MarginGateInferer,
                                                 PosteriorPipeline, ViterbiInferer, validate_posterior_matrix)


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    regime = lambda n: StateNode(n, [StateNode.leaf(f"{n}.a"), StateNode.leaf(f"{n}.b")])
    model = HHMM(StateNode("root", [regime("bull"), regime("bear")]), [GaussianEmission([0])], n_init=1,
                 max_iter=20).fit(rng.normal(size=(400, 1)))
    return model, rng.normal(size=(200, 1))


def test_leaf_and_level_posteriors_validate(fitted):
    model, X = fitted
    validate_posterior_matrix(model.predict_proba(X), "hhmm-leaves")
    validate_posterior_matrix(model.predict_proba_level(X, 1), "hhmm-level-1")
    validate_posterior_matrix(model.predict_proba_fixed_lag(X, 3), "hhmm-fixed-lag")


def test_inferers_transformers_and_pipeline(fitted):
    model, X = fitted
    level = model.predict_proba_level(X, 1)
    labels = MarginGateInferer(theta_top=0.6, theta_margin=0.2).infer(level)
    assert labels.shape == (len(X),) and set(np.unique(labels)) <= {0, 1}
    smoothed = EmaPosteriorTransformer(alpha=0.3).transform(model.predict_proba(X))
    assert np.allclose(smoothed.sum(axis=1), 1.0, atol=1e-6)
    result = PosteriorPipeline(transformers=[EmaPosteriorTransformer(alpha=0.3)], inferer=ArgmaxInferer()).run(level)
    assert result.shape == (len(X),)


def test_viterbi_inferer_from_the_leaf_transition_matrix(fitted):
    model, X = fitted
    path = ViterbiInferer.from_transition_probabilities(model.transition_prob()).infer(model.predict_proba(X))
    assert path.shape == (len(X),) and set(np.unique(path)) <= set(range(model.n_states))
