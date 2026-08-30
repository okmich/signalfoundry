"""Persistence tests: param-dict schema, JSON round-trip, migration stub."""
from datetime import time

import numpy as np
import pytest

from okmich_quant_ml.hmm.hierarchical import (
    HHMMLevel,
    HierarchicalHMM,
    MacroRegime,
    PosteriorMode,
    SessionPolicy,
)
from okmich_quant_ml.hmm.hierarchical import persistence as P
from okmich_quant_ml.hmm.hierarchical.config import SUB_ALPHABET_SIZE, TOPOLOGY_NAME
from okmich_quant_ml.hmm.util import DistType

from .conftest import TRUE_EDGES, TRUE_EMISSIONS


def _two_zigzag_obs():
    """Two zigzags: one at 02:00 (inside a 22:00-06:00 low-liquidity window), one at 12:00 (outside)."""
    import pandas as pd
    from okmich_quant_ml.hmm.hierarchical import ZigzagDirection
    from okmich_quant_ml.hmm.hierarchical.observations import Zigzag, ZigzagObservations
    times = np.array([np.datetime64("2026-01-05T02:00"), np.datetime64("2026-01-05T12:00")], dtype="datetime64[ns]")
    zz = [Zigzag(0, ZigzagDirection.UP, 0, 1, 100.0, 101.0, 0.01, 1, pd.Timestamp(times[0]), pd.Timestamp(times[0])),
          Zigzag(1, ZigzagDirection.DOWN, 1, 2, 101.0, 100.0, 0.01, 2, pd.Timestamp(times[1]), pd.Timestamp(times[1]))]
    return ZigzagObservations(zz, np.array([15, 6]), np.array([1, 0]), np.array([2, 2]), np.array([1, 1]),
                              np.array([0.01, 0.01]), times)


class TestParamDictSchema:
    def test_schema_fields_and_shapes(self, fitted_hhmm):
        d = P.to_param_dict(fitted_hhmm, asset_group="fx_majors", k=1.5, flow_feature="bocpd",
                            session_policy="soft", fit_metadata={"n_zigzags": 3000})
        assert d["schema_version"] == P.HHMM_SCHEMA_VERSION
        assert d["topology"] == TOPOLOGY_NAME
        assert d["asset_group"] == "fx_majors"
        params = d["params"]
        assert len(params["pi_root"]) == 2
        assert np.array(params["A_macro"]).shape == (2, 2)
        assert np.array(params["A_production_run"]).shape == (3, 3)
        assert np.array(params["A_production_rev"]).shape == (3, 3)
        for key in ("B_run_pos", "B_run_neg", "B_rev_pos", "B_rev_neg"):
            assert len(params[key]) == SUB_ALPHABET_SIZE
            assert np.isclose(sum(params[key]), 1.0, atol=1e-6)

    def test_macro_transition_rows_sum_to_one(self, fitted_hhmm):
        d = P.to_param_dict(fitted_hhmm)
        A = np.array(d["params"]["A_macro"])
        assert np.allclose(A.sum(axis=1), 1.0, atol=1e-6)

    def test_a_macro_is_run_first_when_run_is_physical_block_1(self):
        # Run == physical block 1: A_macro must be ordered [Run, Reversal], not physical [0, 1].
        # In TRUE_EDGES the block-1 macro self-transition is 0.60 and block-0's is 0.85, so a
        # mis-ordered (physical) export would put 0.85 at A_macro[0,0].
        starts = np.full(4, 0.25)
        model = HierarchicalHMM.from_flat_params(
            starts=starts, edges=TRUE_EDGES, emissions=TRUE_EMISSIONS,
            macro_labels={1: MacroRegime.RUN.value, 0: MacroRegime.REVERSAL.value},
        )
        assert model.macro_block(MacroRegime.RUN) == 1
        A = np.array(P.to_param_dict(model)["params"]["A_macro"])
        assert A[0, 0] == pytest.approx(0.60, abs=1e-9)   # Run (block 1) self-transition
        assert A[1, 1] == pytest.approx(0.85, abs=1e-9)   # Reversal (block 0) self-transition

    @pytest.mark.slow
    def test_continuous_model_rejected(self, synthetic_stream):
        model = HierarchicalHMM(DistType.NORMAL, covariance_type="diag", random_state=1, max_iter=20)
        rng = np.random.default_rng(0)
        X = rng.normal(size=(500, 2))
        model.fit(X, magnitudes=np.abs(rng.normal(size=500)))
        with pytest.raises(NotImplementedError):
            P.to_param_dict(model)


class TestRoundTrip:
    def test_json_round_trip_matches_posteriors(self, fitted_hhmm, synthetic_stream, tmp_path):
        d = P.to_param_dict(fitted_hhmm, asset_group="fx_majors", k=1.5)
        path = tmp_path / "hhmm.json"
        P.save_param_dict(d, str(path))
        reloaded = P.from_param_dict(P.load_param_dict(str(path)))

        X = synthetic_stream["symbols"]
        for level in (HHMMLevel.MACRO, HHMMLevel.PRODUCTION):
            for mode in PosteriorMode:
                a = fitted_hhmm.predict_proba(X, level, mode, lag=4)
                b = reloaded.predict_proba(X, level, mode, lag=4)
                assert np.allclose(a, b, atol=1e-5), f"{level}/{mode}"

    def test_macro_labels_preserved(self, fitted_hhmm, tmp_path):
        reloaded = P.from_param_dict(P.to_param_dict(fitted_hhmm))
        assert reloaded.macro_labels_ == fitted_hhmm.macro_labels_

    def test_session_gate_preserved(self, fitted_hhmm):
        # A SOFT gate must survive save/load in BEHAVIOUR, not just attributes: the gated posterior
        # on a flagged observation must match before and after the round-trip.
        model = P.from_param_dict(P.to_param_dict(fitted_hhmm))  # clean copy; don't mutate the fixture
        windows = ((time(22, 0), time(6, 0)),)
        model.attach_session_policy(SessionPolicy.SOFT, low_liquidity_windows=windows, soft_downweight=0.4)
        d = P.to_param_dict(model)
        assert d["session_policy"] == SessionPolicy.SOFT.value  # attached gate fills the top-level field
        reloaded = P.from_param_dict(d)
        assert reloaded.session_policy is SessionPolicy.SOFT and reloaded.low_liquidity_windows == windows

        obs = _two_zigzag_obs()  # one zigzag inside the 22:00-06:00 window, one outside
        before = model.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=True)
        after = reloaded.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=True)
        assert np.allclose(before, after, atol=1e-6)
        # and the gate actually did something to the flagged row (not a silent no-op)
        plain = reloaded.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=False)
        assert not np.allclose(after[0], plain[0], atol=1e-6)

    def test_no_session_gate_is_behaviourally_continuous(self, fitted_hhmm):
        # With no gate attached, gating must be a behavioural no-op (whether the attr is absent or a
        # future default of CONTINUOUS) — assert behaviour, not the internal attribute.
        reloaded = P.from_param_dict(P.to_param_dict(fitted_hhmm))
        obs = _two_zigzag_obs()
        gated = reloaded.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=True)
        plain = reloaded.predict_proba(obs, HHMMLevel.MACRO, PosteriorMode.FILTER, apply_session_policy=False)
        assert np.allclose(gated, plain, atol=1e-9)

    def test_save_load_model_helpers(self, fitted_hhmm, synthetic_stream, tmp_path):
        path = tmp_path / "model.json"
        P.save_model(fitted_hhmm, str(path), asset_group="fx_majors")
        reloaded = P.load_model(str(path))
        a = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        b = reloaded.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        assert np.allclose(a, b, atol=1e-5)


class TestMigration:
    def test_newer_schema_rejected(self, fitted_hhmm):
        d = P.to_param_dict(fitted_hhmm)
        with pytest.raises(ValueError):
            P.from_param_dict({**d, "schema_version": P.HHMM_SCHEMA_VERSION + 1})

    def test_missing_reconstruction_rejected(self, fitted_hhmm):
        d = P.to_param_dict(fitted_hhmm)
        d.pop("_reconstruction")
        with pytest.raises(ValueError):
            P.from_param_dict(d)

    def test_migrate_is_identity_for_current_version(self, fitted_hhmm):
        d = P.to_param_dict(fitted_hhmm)
        assert P._migrate(dict(d))["schema_version"] == P.HHMM_SCHEMA_VERSION
