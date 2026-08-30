"""Regression tests for two fit/refit contract bugs.

1. ``train()`` (AIC/BIC model selection) must forward ``n_restarts`` to the instances it constructs, or a
   knife-edge fit silently loses its multi-restart stabilisation.
2. A refit that fails on every restart must not leave the PREVIOUS window's model live: ``self._model``
   is cleared up front, so a caught failure in a rolling backtest errors instead of serving stale
   predictions from the old window.
"""
from __future__ import annotations

import numpy as np
import pytest

from okmich_quant_ml.hmm import DistType, PomegranateHMM, PomegranateMixtureHMM


def _two_regime(n: int = 150, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(-2.0, 1.0, n), rng.normal(2.0, 1.0, n)]).reshape(-1, 1)


def test_train_forwards_n_restarts_single() -> None:
    X = _two_regime()
    best = PomegranateHMM(DistType.NORMAL, n_states=2, random_state=0, n_restarts=3).train(
        X, n_states_range=(2,), criterion="aic")
    assert best.n_restarts == 3, "model selection must not silently drop n_restarts"


def test_train_forwards_n_restarts_mixture() -> None:
    X = _two_regime()
    best = PomegranateMixtureHMM(DistType.NORMAL, n_states=2, n_components=2, random_state=0,
                                 n_restarts=3).train(X, n_states_range=(2,), n_criteria_range=(2,), criterion="aic")
    assert best.n_restarts == 3, "mixture model selection must not silently drop n_restarts"


def test_failed_refit_clears_the_previous_model(monkeypatch: pytest.MonkeyPatch) -> None:
    m = PomegranateHMM(DistType.NORMAL, n_states=2, random_state=0)
    m.fit(_two_regime(seed=1))
    assert m._model is not None, "precondition: a good fit leaves a live model"

    # force every restart to fail on the next fit (as an all-restart covariance failure would)
    monkeypatch.setattr(m, "_fit_single_start", lambda *a, **k: (None, ValueError("forced covariance failure")))
    with pytest.raises(RuntimeError):
        m.fit(_two_regime(seed=2))
    assert m._model is None, "a failed refit must not leave the prior window's model usable"
