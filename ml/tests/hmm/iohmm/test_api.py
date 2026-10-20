"""Duck typing of PomegranateHMM, input validation, persistence, model selection, reporting and emission families."""
from __future__ import annotations

import copy
import inspect

import matplotlib
import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, InferenceMode, InputTiming, PomegranateHMM, PomegranateIOHMM

from ._synthetic import FIT_KW, clock_sequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# forecast's hard-label entry (current_regime) is deprecated in PomegranateHMM and deliberately not carried over
_SIGNATURE_EXCEPTIONS = {"forecast": {"current_regime"}}


def _positional(fn, drop: set[str]) -> list[str]:
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return [p.name for p in inspect.signature(fn).parameters.values()
            if p.kind in kinds and p.name != "self" and p.name not in drop]


def test_duck_types_pomegranate_hmm():
    hmm = PomegranateHMM(DistType.NORMAL, 2)
    io = PomegranateIOHMM(DistType.NORMAL, 2)
    public = [name for name in dir(PomegranateHMM) if not name.startswith("_")] + list(vars(hmm))
    public = [name for name in public if not name.startswith("_")]
    # hasattr on the class, not the instance, so no property is evaluated on an unfitted model
    missing = [name for name in public if not (hasattr(PomegranateIOHMM, name) or name in vars(io))]
    assert not missing, f"PomegranateIOHMM lacks {missing}"
    for name in public:
        base_attr = getattr(PomegranateHMM, name, None)
        if not callable(base_attr) or isinstance(base_attr, property):
            continue
        drop = _SIGNATURE_EXCEPTIONS.get(name, set())
        base_params = _positional(base_attr, drop)
        io_params = _positional(getattr(PomegranateIOHMM, name), drop)
        assert io_params[:len(base_params)] == base_params, f"{name}: {io_params} does not extend {base_params}"


def test_input_validation(fitted_clock_model, clock_data):
    X, U, _ = clock_data
    with pytest.raises(ValueError, match="pass U"):
        fitted_clock_model.predict_proba(X)
    with pytest.raises(ValueError, match="rows"):
        fitted_clock_model.predict_proba(X, U=U[:-1])
    with pytest.raises(ValueError, match="columns"):
        fitted_clock_model.predict_proba(X, U=np.column_stack([U, U]))
    bad = U.copy()
    bad[10, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fitted_clock_model.predict_proba(X, U=bad)
    plain = PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(X[:2000])
    with pytest.raises(ValueError, match="without inputs"):
        plain.predict_proba(X[:2000], U=U[:2000])
    with pytest.raises(ValueError, match="constant"):
        PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(X[:2000], U=np.ones((2000, 1)))
    with pytest.raises(ValueError, match="input_timing"):
        PomegranateIOHMM(DistType.NORMAL, 2, input_timing=[InputTiming.KNOWN_AHEAD] * 2).fit(X[:500], U=U[:500])
    with pytest.raises(RuntimeError, match="not been fitted"):
        PomegranateIOHMM(DistType.NORMAL, 2).predict(X)


def test_save_load_round_trip(fitted_clock_model, clock_data, tmp_path):
    X, U, _ = clock_data
    path = tmp_path / "iohmm.joblib"
    fitted_clock_model.save(str(path))
    loaded = PomegranateIOHMM.load(str(path))
    np.testing.assert_array_equal(loaded.predict_proba(X, U=U), fitted_clock_model.predict_proba(X, U=U))


def test_transitions_and_forecast(fitted_clock_model, clock_data):
    model = fitted_clock_model
    X, U, _ = clock_data
    A = model.transition_matrices(U[:200])
    np.testing.assert_allclose(A[1:].sum(axis=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(model.transition_prob(U[57]), A[57], atol=1e-12)
    belief = model.predict_proba(X[:200], U=U[:200])[-1]
    u_future = U[200:203]
    expected = belief
    for row in u_future:
        expected = expected @ model.transition_prob(row)
    np.testing.assert_allclose(model.forecast(belief, 3, u_future=u_future), expected, atol=1e-12)
    with pytest.raises(ValueError, match="u_future"):
        model.forecast(belief, 3)
    assert model.expected_duration(0, U[57]) == pytest.approx(1 / (1 - A[57, 0, 0]))
    effects = model.input_effects(["clock"])
    assert len(effects) == 2 and set(effects["input"]) == {"clock"}


def test_selection_scores_count_input_weights(fitted_clock_model, clock_data):
    X, U, _ = clock_data
    scores = fitted_clock_model.selection_scores(X[:3000], U=U[:3000])
    k, d = 2, 1
    assert scores.n_params == (k - 1) + k * (k - 1) * (1 + d) + k * 2  # diag Normal in 1-D: mean + variance
    assert scores.log_likelihood == pytest.approx(fitted_clock_model.log_likelihood(X[:3000], U=U[:3000]))
    assert scores.path_entropy >= 0
    assert np.isfinite(scores.icl)


def test_multi_sequence_fit_scores_sequences_separately():
    X1, U1, _ = clock_sequence(1500, seed=21)
    X2, U2, _ = clock_sequence(1200, seed=22)
    X, U = np.vstack([X1, X2]), np.vstack([U1, U2])
    model = PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).fit(X, [1500, 1200], U=U)
    total = model.selection_scores(X, [1500, 1200], U=U, with_entropy=False).log_likelihood
    assert total == pytest.approx(model.log_likelihood(X1, U=U1) + model.log_likelihood(X2, U=U2))


def test_parallel_restarts_give_the_sequential_fit(fitted_three_state_model, three_state_data):
    """Same seeds, winner chosen in restart order: n_jobs changes the wall clock, never the model."""
    X, U, _, lengths = three_state_data
    parallel = PomegranateIOHMM(DistType.NORMAL, 3, input_timing=InputTiming.AT_BAR_CLOSE, n_restarts=3, n_jobs=3,
                                **FIT_KW).fit(X, lengths, U=U)
    sequential = fitted_three_state_model
    assert parallel.train_objective_ == sequential.train_objective_
    np.testing.assert_array_equal(parallel.objective_history_, sequential.objective_history_)
    np.testing.assert_array_equal(parallel.input_coefficients(), sequential.input_coefficients())
    for a, b in zip(parallel.parameters, sequential.parameters):
        np.testing.assert_array_equal(a["means"], b["means"])
        np.testing.assert_array_equal(a["covs"], b["covs"])
    with pytest.raises(ValueError, match="n_jobs"):
        PomegranateIOHMM(DistType.NORMAL, 2, n_restarts=2, n_jobs=0).fit(X[:500])


def test_restarts_run_one_by_one_when_the_threading_layer_cannot_share(monkeypatch, clock_data):
    import okmich_quant_ml.hmm.iohmm.model as model_module
    monkeypatch.setattr(model_module, "concurrent_calls_supported", lambda: False)
    X, U, _ = clock_data
    with pytest.warns(RuntimeWarning, match="workqueue"):
        fallback = PomegranateIOHMM(DistType.NORMAL, 2, n_restarts=2, n_jobs=2, **FIT_KW).fit(X[:3000], U=U[:3000])
    sequential = PomegranateIOHMM(DistType.NORMAL, 2, n_restarts=2, **FIT_KW).fit(X[:3000], U=U[:3000])
    assert fallback.train_objective_ == sequential.train_objective_


def test_train_picks_a_state_count(clock_data):
    X, U, _ = clock_data
    best = PomegranateIOHMM(DistType.NORMAL, 2, **FIT_KW).train(X[:4000], n_states_range=(2, 3), criterion="bic",
                                                                U=U[:4000])
    assert best.n_states in (2, 3) and best.n_inputs == 1


def test_inference_modes(fitted_clock_model, clock_data):
    X, U, _ = clock_data
    model = copy.deepcopy(fitted_clock_model)
    for mode in (InferenceMode.FILTERING, InferenceMode.SMOOTHING):
        model.inference_mode = mode
        np.testing.assert_allclose(model.predict_proba(X[:500], U=U[:500]).sum(axis=1), 1.0, atol=1e-12)
    for mode in (InferenceMode.VITERBI, InferenceMode.CAUSAL_VITERBI):
        model.inference_mode = mode
        assert model.predict(X[:500], U=U[:500]).shape == (500,)
        with pytest.raises(ValueError, match="not applicable"):
            model.predict_proba(X[:500], U=U[:500])


def test_reporting_and_plots(fitted_clock_model, clock_data):
    X, U, _ = clock_data
    model = fitted_clock_model
    summary = model.regime_summary(X[:500], U=U[:500])
    assert "Number of inputs: 1" in summary and "Input effects" in summary
    assert "PomegranateIOHMM" in repr(model)
    response = model.input_response(0)
    assert len(response) == 101 * model.n_states and np.all(response["expected_dwell"] >= 1.0)
    model.plot_input_response(0)
    model.plot_transition_matrix(u=U[57])
    model.plot_distributions(X[:500], U=U[:500])
    plt.close("all")


@pytest.mark.parametrize("dist_type, kwargs, transform", [
    (DistType.STUDENTT, {"covariance_type": "diag"}, lambda x: x),
    (DistType.GAMMA, {}, lambda x: np.abs(x) + 1e-3),
    (DistType.LAMDA, {}, lambda x: x),
])
def test_other_emission_families_fit_with_inputs(dist_type, kwargs, transform):
    X, U, _ = clock_sequence(2000, seed=5)
    model = PomegranateIOHMM(dist_type, 2, random_state=7, max_iter=50, **kwargs).fit(transform(X), U=U)
    proba = model.predict_proba(transform(X), U=U)
    assert np.all(np.isfinite(proba)) and np.allclose(proba.sum(axis=1), 1.0)
    assert len(model.parameters) == 2
