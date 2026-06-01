"""Tests for posterior-column emission in ClusteringComparisonPipeline (Option B).

These cover the logic added so the pipeline emits ``post_{variant}_s{k}`` columns
alongside the ``lbl_{variant}`` hard labels: the soft-assignment selector
(``_safe_predict_proba``, including the Viterbi guard) and the column writer.
"""

import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm import InferenceMode
from okmich_quant_research.backtesting.cluster_comparison_pipeline import (
    _safe_predict_proba, _train_single_model, _write_posterior_columns)


class _FakeModel:
    def __init__(self, inference_mode=None, proba=None, labels=None, raise_proba=False):
        self.inference_mode = inference_mode
        self._proba = proba
        self._labels = labels
        self._raise_proba = raise_proba

    def predict_proba(self, X):
        if self._raise_proba:
            raise RuntimeError("boom")
        return self._proba

    def fit_predict(self, X):
        return self._labels


# ---------------------------------------------------------------------------
# _safe_predict_proba
# ---------------------------------------------------------------------------

def test_safe_predict_proba_gmm_uses_dataframe_directly():
    X = pd.DataFrame({"a": [0.0, 1.0]})
    proba = np.array([[0.6, 0.4], [0.1, 0.9]])
    out = _safe_predict_proba("gmm", _FakeModel(proba=proba), X)
    np.testing.assert_array_equal(out, proba)


def test_safe_predict_proba_hmm_smoothing_uses_values():
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    proba = np.full((3, 2), 0.5)
    model = _FakeModel(inference_mode=InferenceMode.SMOOTHING, proba=proba)
    out = _safe_predict_proba("hmm_lambda", model, X)
    np.testing.assert_array_equal(out, proba)


def test_safe_predict_proba_hmm_viterbi_returns_none():
    # Viterbi is a MAP path, not a marginal posterior -> no soft assignment
    model = _FakeModel(inference_mode=InferenceMode.VITERBI, proba=np.ones((3, 2)))
    assert _safe_predict_proba("hmm_lambda", model, pd.DataFrame({"a": [0, 1, 2]})) is None


def test_safe_predict_proba_hard_clusterer_returns_none():
    assert _safe_predict_proba("kmeans", _FakeModel(), pd.DataFrame({"a": [0, 1]})) is None


def test_safe_predict_proba_degrades_to_none_on_failure():
    model = _FakeModel(inference_mode=InferenceMode.FILTERING, raise_proba=True)
    assert _safe_predict_proba("hmm_lambda", model, pd.DataFrame({"a": [0, 1]})) is None


# ---------------------------------------------------------------------------
# _write_posterior_columns
# ---------------------------------------------------------------------------

def test_write_posterior_columns_creates_one_column_per_state():
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    probs = np.array([[0.7, 0.3], [0.4, 0.6], [0.1, 0.9]])
    _write_posterior_columns(df, "post_", "hmm_lambda", probs)
    assert list(df.columns) == ["close", "post_hmm_lambda_s0", "post_hmm_lambda_s1"]
    np.testing.assert_array_equal(df["post_hmm_lambda_s0"].to_numpy(), [0.7, 0.4, 0.1])
    np.testing.assert_array_equal(df["post_hmm_lambda_s1"].to_numpy(), [0.3, 0.6, 0.9])


def test_write_posterior_columns_noop_on_none():
    df = pd.DataFrame({"close": [1.0, 2.0]})
    _write_posterior_columns(df, "post_", "kmeans", None)
    assert list(df.columns) == ["close"]


def test_written_columns_are_discoverable_by_evaluator():
    # the pipeline's emission convention must match what the evaluator parses
    from okmich_quant_labelling.utils.posterior_eval_util import discover_posterior_variants
    df = pd.DataFrame({"close": [1.0, 2.0]})
    _write_posterior_columns(df, "post_", "hmm_lambda", np.array([[0.5, 0.5], [0.5, 0.5]]))
    variants = discover_posterior_variants(df)
    assert variants == {"hmm_lambda": ["post_hmm_lambda_s0", "post_hmm_lambda_s1"]}


# ---------------------------------------------------------------------------
# _train_single_model returns posteriors in its tuple
# ---------------------------------------------------------------------------

def test_train_single_model_emits_probs_for_soft_model():
    X = pd.DataFrame({"a": np.linspace(0, 1, 20), "b": np.linspace(1, 0, 20)})
    labels = np.tile([0, 1], 10)
    proba = np.column_stack([np.linspace(0.9, 0.1, 20), np.linspace(0.1, 0.9, 20)])
    model = _FakeModel(inference_mode=InferenceMode.SMOOTHING, proba=proba, labels=labels)
    result = _train_single_model("hmm_lambda", model, X, "SYM", "lbl_")
    assert len(result) == 7  # name, model, labels, probs, sil, n_clusters, error
    name, fitted, out_labels, probs, sil, n_clusters, error = result
    assert error is None
    assert probs.shape == (20, 2)
    assert n_clusters == 2


def test_train_single_model_error_tuple_has_seven_slots():
    model = _FakeModel(labels=None)  # fit_predict returns None -> .astype fails
    result = _train_single_model("hmm_lambda", model, pd.DataFrame({"a": [0, 1]}), "SYM", "lbl_")
    assert len(result) == 7
    assert result[-1] is not None  # error string populated
    assert result[1] is None       # no fitted model on failure