"""Tests for InferenceModelWrapper, HmmModelWrapper, and KerasModelWrapper."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from okmich_quant_ml.inference_model_wrappers import HmmModelWrapper, InferenceModelWrapper
from okmich_quant_ml.posterior_inference import ArgmaxInferer, PosteriorPipeline


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sklearn_artifact_dir(tmp_path: Path, with_pipeline: bool = False) -> Path:
    """Fit a tiny LogisticRegression and save artifacts in three-file format."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 4))
    y = (X[:, 0] > 0).astype(int)

    model = LogisticRegression(max_iter=200).fit(X, y)
    artifacts = {"model": model}

    artifact_dir = tmp_path / "sklearn_artifacts"
    artifact_dir.mkdir()
    joblib.dump(artifacts, artifact_dir / "model.joblib")

    pipeline_path = None
    if with_pipeline:
        scaler = StandardScaler().fit(X)
        pipeline_path = str(artifact_dir / "pipeline.joblib")
        joblib.dump(scaler, pipeline_path)

    return artifact_dir, str(artifact_dir / "model.joblib"), pipeline_path


def _make_hmm_artifact_dir(tmp_path: Path, metadata_overrides: dict | None = None) -> Path:
    """Fit a real lightweight PomegranateHMM and write a three-file export dir."""
    from okmich_quant_ml.hmm import create_simple_hmm_instance, InferenceMode
    from okmich_quant_ml.hmm.pomegranate import DistType

    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 2))

    model = create_simple_hmm_instance(
        DistType.NORMAL, n_states=3, random_state=0, max_iter=10,
        inference_mode=InferenceMode.FILTERING, covariance_type="diag",
    )
    model.fit(X)

    scaler = StandardScaler().fit(X)

    base_metadata = {
        "state_mapping": {"0": 1, "1": 0, "2": -1},
        "target_lag": 5,
        "n_states": 3,
    }
    if metadata_overrides:
        base_metadata.update(metadata_overrides)

    artifact_dir = tmp_path / "hmm_export"
    artifact_dir.mkdir()
    joblib.dump(model, artifact_dir / HmmModelWrapper.MODEL_FILENAME)
    joblib.dump(scaler, artifact_dir / HmmModelWrapper.PIPELINE_FILENAME)
    (artifact_dir / HmmModelWrapper.METADATA_FILENAME).write_text(
        json.dumps(base_metadata), encoding="utf-8"
    )
    return artifact_dir


# ---------------------------------------------------------------------------
# InferenceModelWrapper — sklearn
# ---------------------------------------------------------------------------

class TestInferenceModelWrapperSklearn:
    def test_predict_returns_probs_and_labels(self, tmp_path):
        artifact_dir, model_path, _ = _make_sklearn_artifact_dir(tmp_path)
        wrapper = InferenceModelWrapper({"type": "sklearn", "model_path": model_path})

        rng = np.random.default_rng(1)
        X = rng.standard_normal((10, 4))
        probs, labels = wrapper.predict(X)

        assert probs.shape == (10, 2)
        assert labels.shape == (10,)
        np.testing.assert_array_equal(labels, np.argmax(probs, axis=1))

    def test_predict_with_pipeline_transforms_features(self, tmp_path):
        artifact_dir, model_path, pipeline_path = _make_sklearn_artifact_dir(tmp_path, with_pipeline=True)
        wrapper = InferenceModelWrapper({"type": "sklearn", "model_path": model_path, "pipeline_path": pipeline_path})

        X = np.random.default_rng(2).standard_normal((8, 4))
        probs, labels = wrapper.predict(X)

        assert probs.shape == (8, 2)
        assert labels.shape == (8,)

    def test_no_pipeline_path_leaves_features_unchanged(self, tmp_path):
        artifact_dir, model_path, _ = _make_sklearn_artifact_dir(tmp_path)
        wrapper = InferenceModelWrapper({"type": "sklearn", "model_path": model_path})

        assert wrapper.transform_pipeline is None

    def test_unknown_model_type_raises(self, tmp_path):
        artifact_dir, model_path, _ = _make_sklearn_artifact_dir(tmp_path)
        with pytest.raises(ValueError, match="Unknown model type"):
            InferenceModelWrapper({"type": "xgboost", "model_path": model_path})

    def test_predict_accepts_dataframe_input(self, tmp_path):
        import pandas as pd
        artifact_dir, model_path, _ = _make_sklearn_artifact_dir(tmp_path)
        wrapper = InferenceModelWrapper({"type": "sklearn", "model_path": model_path})

        X_df = pd.DataFrame(np.random.default_rng(3).standard_normal((6, 4)))
        probs, labels = wrapper.predict(X_df)

        assert probs.shape == (6, 2)


# ---------------------------------------------------------------------------
# InferenceModelWrapper — prophet (mocked)
# ---------------------------------------------------------------------------

class TestInferenceModelWrapperProphet:
    def test_predict_delegates_to_get_features(self, tmp_path):
        fake_features = np.ones((5, 3))
        mock_service = MagicMock()
        mock_service.get_features.return_value = fake_features

        with patch("okmich_quant_ml.inference_model_wrappers.ProphetFeatureGenerationService",
                   return_value=mock_service):
            model_path = str(tmp_path / "model.json")
            Path(model_path).write_text("{}", encoding="utf-8")
            wrapper = InferenceModelWrapper({"type": "prophet", "model_path": model_path})
            result = wrapper.predict(np.ones((5, 2)))

        mock_service.get_features.assert_called_once()
        np.testing.assert_array_equal(result, fake_features)


# ---------------------------------------------------------------------------
# HmmModelWrapper
# ---------------------------------------------------------------------------

class TestHmmModelWrapperInit:
    def test_loads_all_artifacts_from_directory(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert wrapper.model is not None
        assert wrapper.transform_pipeline is not None
        assert isinstance(wrapper.metadata, dict)

    def test_resolves_parent_dir_from_model_file_path(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        model_file = str(artifact_dir / HmmModelWrapper.MODEL_FILENAME)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": model_file})

        assert wrapper.artifact_dir == artifact_dir

    def test_state_mapping_keys_normalized_to_int(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert all(isinstance(k, int) for k in wrapper.state_mapping)
        assert wrapper.state_mapping == {0: 1, 1: 0, 2: -1}

    def test_fixed_lag_defaults_from_metadata_target_lag(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path, metadata_overrides={"target_lag": 7})
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert wrapper.fixed_lag == 7

    def test_fixed_lag_falls_back_to_lag_key(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path, metadata_overrides={"target_lag": None, "lag": 3})
        # remove target_lag so fallback activates
        meta_path = artifact_dir / HmmModelWrapper.METADATA_FILENAME
        meta = json.loads(meta_path.read_text())
        del meta["target_lag"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert wrapper.fixed_lag == 3

    def test_explicit_fixed_lag_overrides_metadata(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)}, fixed_lag=2)

        assert wrapper.fixed_lag == 2

    def test_negative_fixed_lag_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        with pytest.raises(ValueError, match="fixed_lag must be >= 0"):
            HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)}, fixed_lag=-1)

    def test_wrong_model_type_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        with pytest.raises(ValueError, match="HmmModelWrapper only supports model_type='hmm'"):
            HmmModelWrapper({"type": "sklearn", "model_path": str(artifact_dir)})

    def test_missing_model_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="model_path"):
            HmmModelWrapper({"type": "hmm"})

    def test_invalid_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HmmModelWrapper({"type": "hmm", "model_path": str(tmp_path / "nonexistent")})

    def test_missing_model_file_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        (artifact_dir / HmmModelWrapper.MODEL_FILENAME).unlink()
        with pytest.raises(FileNotFoundError, match="hmm model"):
            HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

    def test_missing_pipeline_file_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        (artifact_dir / HmmModelWrapper.PIPELINE_FILENAME).unlink()
        with pytest.raises(FileNotFoundError, match="transform pipeline"):
            HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

    def test_missing_metadata_file_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        (artifact_dir / HmmModelWrapper.METADATA_FILENAME).unlink()
        with pytest.raises(FileNotFoundError, match="metadata"):
            HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

    def test_metadata_not_dict_raises(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        (artifact_dir / HmmModelWrapper.METADATA_FILENAME).write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(TypeError, match="JSON object"):
            HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

    def test_empty_state_mapping_normalizes_to_empty_dict(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path, metadata_overrides={"state_mapping": {}})
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert wrapper.state_mapping == {}

    def test_default_posterior_inferer_is_argmax(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        assert isinstance(wrapper.posterior_inferer, ArgmaxInferer)


class TestHmmModelWrapperPredict:
    def test_predict_without_fixed_lag_returns_probs_and_labels(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                  use_fixed_lag_posterior=False)

        X = np.random.default_rng(10).standard_normal((50, 2))
        probs, labels = wrapper.predict(X)

        assert probs.shape == (50, 3)
        assert labels.shape == (50,)
        np.testing.assert_array_equal(labels, np.argmax(probs, axis=1))

    def test_predict_with_fixed_lag_uses_single_lag_api(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                  use_fixed_lag_posterior=True, fixed_lag=3)

        X = np.random.default_rng(11).standard_normal((50, 2))
        probs, labels = wrapper.predict(X)

        assert probs.shape == (1, 3)
        assert labels.shape == (1,)
        np.testing.assert_array_equal(labels, np.argmax(probs, axis=1))

    def test_fixed_lag_and_no_fixed_lag_produce_different_posteriors(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper_fl = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                     use_fixed_lag_posterior=True, fixed_lag=5)
        wrapper_nfl = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                      use_fixed_lag_posterior=False)

        X = np.random.default_rng(12).standard_normal((80, 2))
        probs_fl, _ = wrapper_fl.predict(X)
        probs_nfl, _ = wrapper_nfl.predict(X)

        assert probs_fl.shape == (1, 3)
        matured_idx = X.shape[0] - 1 - wrapper_fl.fixed_lag
        probs_nfl_same_timestep = probs_nfl[[matured_idx]]
        # Fixed-lag and filtering posteriors differ at the same matured timestep.
        assert not np.allclose(probs_fl, probs_nfl_same_timestep)

    def test_predict_with_fixed_lag_raises_when_window_too_short(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                  use_fixed_lag_posterior=True, fixed_lag=5)

        X_short = np.random.default_rng(120).standard_normal((5, 2))
        with pytest.raises(ValueError, match="Not enough rows for fixed-lag matured posterior"):
            wrapper.predict(X_short)

    def test_fixed_lag_batch_t_minus_l_matches_live_asof_t(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        lag = 4
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)},
                                  use_fixed_lag_posterior=True, fixed_lag=lag)

        X = np.random.default_rng(121).standard_normal((35, 2))
        transformed = wrapper.transform_pipeline.transform(X)
        batch_probs = np.asarray(wrapper.model.predict_proba_fixed_lag(transformed, lag=lag), dtype=float)

        for t in range(lag, X.shape[0]):
            X_asof_t = X[: t + 1]
            live_probs, live_labels = wrapper.predict(X_asof_t)
            expected_probs = batch_probs[[t - lag]]
            expected_labels = np.argmax(expected_probs, axis=1)
            np.testing.assert_allclose(live_probs, expected_probs, rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(live_labels, expected_labels)

    def test_predict_accepts_dataframe_input(self, tmp_path):
        import pandas as pd
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        X_df = pd.DataFrame(np.random.default_rng(13).standard_normal((30, 2)))
        probs, labels = wrapper.predict(X_df)

        assert probs.shape == (30, 3)

    def test_custom_posterior_pipeline_applied(self, tmp_path):
        class DoubleTopStateTransformer:
            """Doubles the probability of the top state, then renormalizes."""
            def transform(self, probs: np.ndarray) -> np.ndarray:
                top = np.argmax(probs, axis=1)
                out = probs.copy()
                out[np.arange(len(top)), top] *= 2.0
                return out / out.sum(axis=1, keepdims=True)

        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper(
            {"type": "hmm", "model_path": str(artifact_dir)},
            use_fixed_lag_posterior=False,
            posterior_transformers=[DoubleTopStateTransformer()],
            posterior_inferer=ArgmaxInferer(),
        )

        X = np.random.default_rng(14).standard_normal((40, 2))
        _, labels = wrapper.predict(X)

        assert labels.shape == (40,)

    def test_posterior_invariant_rejects_non_normalized_rows(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        def _bad_predict_proba(X):
            return np.array([[0.5, 0.5, 0.5]], dtype=float)  # row sum = 1.5

        wrapper.model.predict_proba = _bad_predict_proba

        X = np.random.default_rng(15).standard_normal((10, 2))
        with pytest.raises(ValueError, match="Posterior rows must sum to 1"):
            wrapper.predict(X)

    def test_posterior_invariant_rejects_nan_inf(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})

        def _bad_predict_proba(X):
            return np.array([[0.2, np.nan, 0.8], [0.1, np.inf, 0.9]], dtype=float)

        wrapper.model.predict_proba = _bad_predict_proba

        X = np.random.default_rng(16).standard_normal((10, 2))
        with pytest.raises(ValueError, match="Posterior contains NaN or Inf values"):
            wrapper.predict(X)

    def test_posterior_invariant_can_be_disabled(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper(
            {"type": "hmm", "model_path": str(artifact_dir)},
            validate_posterior_invariants=False,
        )

        def _bad_predict_proba(X):
            return np.array([[0.5, 0.5, 0.5]], dtype=float)

        wrapper.model.predict_proba = _bad_predict_proba

        X = np.random.default_rng(17).standard_normal((10, 2))
        probs, labels = wrapper.predict(X)
        assert probs.shape == (1, 3)
        assert labels.shape == (1,)

    def test_fixed_lag_alignment_validation_passes(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)}, use_fixed_lag_posterior=True, fixed_lag=4)

        asof_timestamp = pd.Timestamp("2026-04-15 10:00:00")
        label_timestamp = pd.Timestamp("2026-04-15 09:40:00")
        wrapper.validate_fixed_lag_alignment(asof_timestamp=asof_timestamp, label_timestamp=label_timestamp, bar_timedelta="5min")

    def test_fixed_lag_alignment_validation_raises_on_misalignment(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)}, use_fixed_lag_posterior=True, fixed_lag=4)

        asof_timestamp = pd.Timestamp("2026-04-15 10:00:00")
        label_timestamp = pd.Timestamp("2026-04-15 09:45:00")
        with pytest.raises(ValueError, match="Fixed-lag timestamp misalignment"):
            wrapper.validate_fixed_lag_alignment(asof_timestamp=asof_timestamp, label_timestamp=label_timestamp, bar_timedelta="5min")

    def test_fixed_lag_alignment_validation_requires_fixed_lag_mode(self, tmp_path):
        artifact_dir = _make_hmm_artifact_dir(tmp_path)
        wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)}, use_fixed_lag_posterior=False, fixed_lag=4)

        with pytest.raises(ValueError, match="applies only when use_fixed_lag_posterior=True"):
            wrapper.validate_fixed_lag_alignment(
                asof_timestamp=pd.Timestamp("2026-04-15 10:00:00"),
                label_timestamp=pd.Timestamp("2026-04-15 09:40:00"),
                bar_timedelta="5min",
            )
