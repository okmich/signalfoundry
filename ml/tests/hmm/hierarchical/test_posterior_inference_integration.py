"""Integration: HHMM posteriors flow into the posterior_inference stack unchanged.

The HHMM's ``predict_proba`` returns the same ``(T, K)`` simplex contract the rest of the HMM stack
produces, so the inferers/transformers/pipeline and the fixed-lag sweep consume it directly.
"""
import numpy as np

from okmich_quant_ml.hmm.hierarchical import HHMMLevel, PosteriorMode
from okmich_quant_ml.posterior_inference import (
    ArgmaxInferer,
    EmaPosteriorTransformer,
    MarginGateInferer,
    PosteriorPipeline,
    ViterbiInferer,
    validate_posterior_matrix,
)


class TestPosteriorInferenceContract:
    def test_macro_posterior_validates(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        # posterior_inference's own validator accepts the HHMM output as a (T, K) simplex
        validate_posterior_matrix(macro, "hhmm-macro")

    def test_margin_gate_inferer(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        inferer = MarginGateInferer(theta_top=0.6, theta_margin=0.2)
        labels = inferer.infer(macro)
        assert labels.shape == (len(macro),)
        assert set(np.unique(labels)).issubset({0, 1})
        assert "gate_open_rate" in inferer.get_metadata()

    def test_ema_transformer_preserves_simplex(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        smoothed = EmaPosteriorTransformer(alpha=0.3).transform(macro)
        assert smoothed.shape == macro.shape
        assert np.allclose(smoothed.sum(1), 1.0, atol=1e-6)

    def test_posterior_pipeline(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        pipeline = PosteriorPipeline(transformers=[EmaPosteriorTransformer(alpha=0.3)], inferer=ArgmaxInferer())
        result = pipeline.run(macro)
        assert result.shape == (len(macro),)

    def test_viterbi_from_macro_transition(self, fitted_hhmm, synthetic_stream):
        macro = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.MACRO, PosteriorMode.FILTER)
        # 2x2 macro transition -> Viterbi cost matrix -> decode a MAP regime path
        import numpy as np
        from okmich_quant_ml.hmm.hierarchical import persistence as P
        A_macro = np.array(P.to_param_dict(fitted_hhmm)["params"]["A_macro"])
        viterbi = ViterbiInferer.from_transition_probabilities(A_macro)
        path = viterbi.infer(macro)
        assert path.shape == (len(macro),)
        assert set(np.unique(path)).issubset({0, 1})


class TestFixedLagContract:
    def test_fixed_lag_sweep_shapes(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"].reshape(-1, 1)
        sweep = fitted_hhmm.predict_proba_fixed_lag_sweep(X, [0, 1, 5])
        for lag in (0, 1, 5):
            assert sweep[lag].shape == (len(X), 4)
            assert np.allclose(sweep[lag].sum(1), 1.0, atol=1e-6)

    def test_fixed_lag_zero_equals_filter(self, fitted_hhmm, synthetic_stream):
        X = synthetic_stream["symbols"].reshape(-1, 1)
        lag0 = fitted_hhmm.predict_proba_fixed_lag(X, 0)
        filt = fitted_hhmm.predict_proba(synthetic_stream["symbols"], HHMMLevel.PRODUCTION, PosteriorMode.FILTER)
        assert np.allclose(lag0, filt, atol=1e-6)
