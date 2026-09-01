"""
Functional tests for transform_pipeline module.

Tests the end-to-end workflow of:
1. Creating transformation configs from recommendations
2. Building pipelines with various options
3. Saving and loading pipeline artifacts
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler

from okmich_quant_features.utils.transform_pipeline import (
    config_transformation_type,
    encode_transformation_recommendation,
    export_transformation_config,
    load_transformation_config,
    build_pipeline_from_config,
    save_pipeline_artifacts,
    load_pipeline_artifacts,
)


@pytest.fixture(scope="function")
def cleanup_temp_files():
    """
    Fixture to track and cleanup temporary files created during tests.

    Yields a list that tests can append paths to for cleanup.
    After the test completes, all tracked paths are removed.
    """
    temp_paths = []

    yield temp_paths

    # Cleanup after test
    for path in temp_paths:
        path = Path(path)
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except Exception as e:
            # Log but don't fail test on cleanup issues
            print(f"Warning: Could not cleanup {path}: {e}")


@pytest.fixture
def sample_recommendations():
    """Sample transformation recommendations DataFrame."""
    return pd.DataFrame({
        'feature': ['returns', 'volume', 'hurst', 'atr', 'rsi'],
        'transformations': ['yeo-johnson', 'log', 'logit', 'box-cox', 'none'],
        'reason': [
            'Skewed distribution',
            'Right-skewed',
            'Bounded [0,1]',
            'Positive only',
            'Already normal'
        ]
    })


@pytest.fixture
def sample_data():
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'returns': np.random.randn(n_samples) * 0.02,
        'volume': np.abs(np.random.randn(n_samples)) * 1000 + 1000,
        'hurst': np.random.uniform(0.3, 0.7, n_samples),
        'atr': np.abs(np.random.randn(n_samples)) * 10 + 50,
        'rsi': np.random.uniform(20, 80, n_samples),
    })


@pytest.fixture
def sample_target():
    """Generate sample target for testing."""
    np.random.seed(42)
    return np.random.choice([0, 1, 2], size=100)


class TestEncodeTransformationRecommendation:
    """Test encode_transformation_recommendation function."""

    def test_basic_encoding(self, sample_recommendations):
        """Test basic conversion of recommendations to config dict."""
        config = encode_transformation_recommendation(sample_recommendations)

        assert 'description' in config
        assert 'transformations' in config
        assert len(config['transformations']) == 5

    def test_transformation_types(self, sample_recommendations):
        """Test that transformation types are correctly parsed."""
        config = encode_transformation_recommendation(sample_recommendations)

        assert config['transformations']['returns']['type'] == 'yeo-johnson'
        assert config['transformations']['volume']['type'] == 'log'
        assert config['transformations']['hurst']['type'] == 'logit'
        assert config['transformations']['atr']['type'] == 'box-cox'
        assert config['transformations']['rsi']['type'] == 'passthrough'

    def test_custom_description(self, sample_recommendations):
        """Test custom description."""
        desc = "Custom pipeline config"
        config = encode_transformation_recommendation(sample_recommendations, description=desc)

        assert config['description'] == desc

    def test_reason_preservation(self, sample_recommendations):
        """Test that reasons are preserved in config."""
        config = encode_transformation_recommendation(sample_recommendations)

        assert config['transformations']['returns']['reason'] == 'Skewed distribution'
        assert config['transformations']['volume']['reason'] == 'Right-skewed'


class TestExportLoadConfig:
    """Test export and load transformation config functions."""

    def test_export_creates_file(self, sample_recommendations, tmp_path):
        """Test that export creates a JSON file."""
        output_path = tmp_path / "config.json"
        config = export_transformation_config(sample_recommendations, str(output_path))

        assert output_path.exists()
        assert config is not None

    def test_export_returns_dict(self, sample_recommendations, tmp_path):
        """Test that export returns the config dict."""
        output_path = tmp_path / "config.json"
        config = export_transformation_config(sample_recommendations, str(output_path))

        assert isinstance(config, dict)
        assert 'transformations' in config

    def test_load_config(self, sample_recommendations, tmp_path):
        """Test loading config from file."""
        output_path = tmp_path / "config.json"
        original_config = export_transformation_config(sample_recommendations, str(output_path))

        loaded_config = load_transformation_config(str(output_path))

        assert loaded_config == original_config

    def test_export_creates_parent_dirs(self, sample_recommendations, tmp_path):
        """Test that export creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "config.json"
        export_transformation_config(sample_recommendations, str(output_path))

        assert output_path.exists()


class TestBuildPipelineFromConfig:
    """Test pipeline building with various configurations."""

    def test_basic_pipeline(self, sample_recommendations, sample_data):
        """Test building a basic pipeline with default scaler."""
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='robust')

        # Fit and transform
        transformed = pipeline.fit_transform(sample_data)

        assert transformed.shape[0] == sample_data.shape[0]
        assert isinstance(transformed, np.ndarray)

    def test_no_scaler(self, sample_recommendations, sample_data):
        """Test pipeline without scaler."""
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type=None)

        transformed = pipeline.fit_transform(sample_data)
        assert transformed.shape[0] == sample_data.shape[0]

    def test_different_scalers(self, sample_recommendations, sample_data):
        """Test different scaler types."""
        config = encode_transformation_recommendation(sample_recommendations)

        for scaler_type in ['robust', 'standard', 'minmax']:
            pipeline = build_pipeline_from_config(config, scaler_type=scaler_type)
            transformed = pipeline.fit_transform(sample_data)
            assert transformed.shape[0] == sample_data.shape[0]

    def test_fitted_scaler(self, sample_recommendations, sample_data):
        """Test using a pre-fitted scaler."""
        config = encode_transformation_recommendation(sample_recommendations)

        # Pre-fit a scaler
        fitted_scaler = StandardScaler().fit(sample_data)

        pipeline = build_pipeline_from_config(config, fitted_scaler=fitted_scaler)
        transformed = pipeline.fit_transform(sample_data)

        assert transformed.shape[0] == sample_data.shape[0]

    def test_post_transformers_pca(self, sample_recommendations, sample_data):
        """Test pipeline with PCA as post-transformer."""
        config = encode_transformation_recommendation(sample_recommendations)

        # Pre-fit PCA
        pca = PCA(n_components=3).fit(sample_data)

        pipeline = build_pipeline_from_config(
            config,
            scaler_type='robust',
            post_transformers=[('pca', pca)]
        )

        transformed = pipeline.fit_transform(sample_data)

        # Should have 3 components from PCA
        assert transformed.shape == (sample_data.shape[0], 3)

    def test_multiple_post_transformers(self, sample_recommendations, sample_data, sample_target):
        """Test pipeline with multiple post-transformers."""
        config = encode_transformation_recommendation(sample_recommendations)

        # Pre-fit transformers
        pca = PCA(n_components=4).fit(sample_data)
        selector = SelectKBest(f_classif, k=2).fit(
            pca.transform(sample_data), sample_target
        )

        pipeline = build_pipeline_from_config(
            config,
            scaler_type='standard',
            post_transformers=[
                ('pca', pca),
                ('selector', selector)
            ]
        )

        # Fit pipeline (SelectKBest needs y, so pass it)
        transformed = pipeline.fit_transform(sample_data, sample_target)

        # Should have 2 features from SelectKBest
        assert transformed.shape == (sample_data.shape[0], 2)

    def test_empty_config(self, sample_data):
        """Test pipeline with empty transformation config."""
        config = {'description': 'Empty config', 'transformations': {}}
        pipeline = build_pipeline_from_config(config, scaler_type='robust')

        transformed = pipeline.fit_transform(sample_data)
        assert transformed.shape == sample_data.shape

    def test_config_from_file(self, sample_recommendations, sample_data, tmp_path):
        """Test building pipeline from config file path."""
        config_path = tmp_path / "config.json"
        export_transformation_config(sample_recommendations, str(config_path))

        pipeline = build_pipeline_from_config(str(config_path), scaler_type='robust')
        transformed = pipeline.fit_transform(sample_data)

        assert transformed.shape[0] == sample_data.shape[0]

    def test_invalid_scaler_type(self, sample_recommendations):
        """Test that invalid scaler type raises error."""
        config = encode_transformation_recommendation(sample_recommendations)

        with pytest.raises(ValueError, match="Unknown scaler"):
            build_pipeline_from_config(config, scaler_type='invalid')


class TestSaveLoadPipelineArtifacts:
    """Test saving and loading pipeline artifacts."""

    def test_save_artifacts(self, sample_recommendations, sample_data, tmp_path):
        """Test saving pipeline artifacts."""
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        pipeline.fit(sample_data)

        output_dir = tmp_path / "artifacts"
        feature_cols = list(sample_data.columns)
        metadata = {'version': '1.0', 'created': '2024-01-01'}

        save_pipeline_artifacts(pipeline, feature_cols, str(output_dir), metadata)

        assert (output_dir / 'pipeline.pkl').exists()
        assert (output_dir / 'feature_cols.json').exists()
        assert (output_dir / 'metadata.json').exists()

    def test_load_artifacts(self, sample_recommendations, sample_data, tmp_path):
        """Test loading pipeline artifacts."""
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        pipeline.fit(sample_data)

        output_dir = tmp_path / "artifacts"
        feature_cols = list(sample_data.columns)
        metadata = {'version': '1.0', 'model': 'test'}

        save_pipeline_artifacts(pipeline, feature_cols, str(output_dir), metadata)

        # Load artifacts
        artifacts = load_pipeline_artifacts(str(output_dir))

        assert 'pipeline' in artifacts
        assert 'feature_cols' in artifacts
        assert 'metadata' in artifacts
        assert artifacts['feature_cols'] == feature_cols
        assert artifacts['metadata']['version'] == '1.0'

    def test_loaded_pipeline_works(self, sample_recommendations, sample_data, tmp_path):
        """Test that loaded pipeline produces same results."""
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        pipeline.fit(sample_data)

        # Transform with original
        original_result = pipeline.transform(sample_data)

        # Save and load
        output_dir = tmp_path / "artifacts"
        save_pipeline_artifacts(pipeline, list(sample_data.columns), str(output_dir))
        artifacts = load_pipeline_artifacts(str(output_dir))

        # Transform with loaded
        loaded_result = artifacts['pipeline'].transform(sample_data)

        np.testing.assert_array_almost_equal(original_result, loaded_result)


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""

    def test_complete_workflow_with_pca(self, sample_recommendations, sample_data, sample_target, tmp_path):
        """Test complete workflow: recommendations → config → pipeline → save → load."""
        # 1. Create config from recommendations
        config = encode_transformation_recommendation(
            sample_recommendations,
            description="Test workflow"
        )

        # 2. Export config to file
        config_path = tmp_path / "config.json"
        export_transformation_config(sample_recommendations, str(config_path))

        # 3. Build pipeline with post-transformers
        pca = PCA(n_components=3).fit(sample_data)
        pipeline = build_pipeline_from_config(
            config,
            scaler_type='robust',
            post_transformers=[('pca', pca)]
        )

        # 4. Fit pipeline
        pipeline.fit(sample_data)
        train_transformed = pipeline.transform(sample_data)

        # 5. Save artifacts
        artifacts_dir = tmp_path / "artifacts"
        save_pipeline_artifacts(
            pipeline,
            list(sample_data.columns),
            str(artifacts_dir),
            metadata={'version': '1.0', 'n_components': 3}
        )

        # 6. Load artifacts and verify
        artifacts = load_pipeline_artifacts(str(artifacts_dir))
        loaded_transformed = artifacts['pipeline'].transform(sample_data)

        np.testing.assert_array_almost_equal(train_transformed, loaded_transformed)
        assert artifacts['metadata']['n_components'] == 3

    def test_workflow_with_fitted_scaler(self, sample_recommendations, sample_data, tmp_path):
        """Test workflow using fitted scaler instead of scaler type."""
        # Fit scaler on training data
        train_scaler = RobustScaler().fit(sample_data)

        # Create config and build pipeline with fitted scaler
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(
            config,
            fitted_scaler=train_scaler
        )

        pipeline.fit(sample_data)
        transformed = pipeline.transform(sample_data)

        # Save and load
        artifacts_dir = tmp_path / "artifacts"
        save_pipeline_artifacts(pipeline, list(sample_data.columns), str(artifacts_dir))
        artifacts = load_pipeline_artifacts(str(artifacts_dir))

        # Verify loaded pipeline works
        loaded_transformed = artifacts['pipeline'].transform(sample_data)
        np.testing.assert_array_almost_equal(transformed, loaded_transformed)

    def test_inference_workflow(self, sample_recommendations, sample_data, tmp_path):
        """Test realistic training → inference workflow."""
        # Training phase
        train_data = sample_data.iloc[:80]
        test_data = sample_data.iloc[80:]

        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='standard')
        pipeline.fit(train_data)

        # Save for inference
        artifacts_dir = tmp_path / "artifacts"
        save_pipeline_artifacts(
            pipeline,
            list(sample_data.columns),
            str(artifacts_dir),
            metadata={'train_size': len(train_data)}
        )

        # Inference phase (simulate loading in production)
        artifacts = load_pipeline_artifacts(str(artifacts_dir))
        inference_pipeline = artifacts['pipeline']

        # Transform test data
        test_transformed = inference_pipeline.transform(test_data)

        assert test_transformed.shape[0] == len(test_data)
        assert artifacts['metadata']['train_size'] == 80


def teardown_module(module):
    """
    Module-level teardown to cleanup any test artifacts.

    This runs once after all tests in the module complete.
    pytest's tmp_path fixture auto-cleans, but this ensures
    any other temp files are removed.
    """
    import gc
    gc.collect()  # Force garbage collection to release file handles

    # Clean up any leftover temp directories
    temp_dir = Path(tempfile.gettempdir())
    for item in temp_dir.glob("pytest-*"):
        try:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
        except Exception:
            pass  # Ignore cleanup errors


# =============================================================================
# Test Config Encoding Of Every Recommendation Type
# =============================================================================

class TestConfigTransformationType:
    """Test the recommendation-string to config-type mapping."""

    def test_single_tokens(self):
        assert config_transformation_type('yeo-johnson') == 'yeo-johnson'
        assert config_transformation_type('box-cox') == 'box-cox'
        assert config_transformation_type('logit') == 'logit'
        assert config_transformation_type('log') == 'log'

    def test_none_and_missing_are_passthrough(self):
        assert config_transformation_type('none') == 'passthrough'
        assert config_transformation_type(None) == 'passthrough'
        assert config_transformation_type(np.nan) == 'passthrough'

    def test_standardize_reaches_the_config(self):
        """Previously collapsed to 'passthrough', so the exported JSON under-reported what the
        EDA actually recommended."""
        assert config_transformation_type('standardize') == 'standardize'

    def test_quantile_or_rank_reaches_the_config(self):
        """A quantile transform is not affine, so no downstream scaler can stand in for it."""
        assert config_transformation_type('quantile/rank') == 'quantile'

    def test_compound_string_keeps_the_leading_transformation(self):
        """Regression: any occurrence of 'standardize' used to void the whole row."""
        assert config_transformation_type('box-cox, standardize') == 'box-cox'
        assert config_transformation_type('logit, quantile/rank, standardize') == 'logit'

    def test_log_is_not_confused_with_logit(self):
        assert config_transformation_type('logit') == 'logit'
        assert config_transformation_type('log') == 'log'


class TestPipelineCoversEveryRecommendationType:
    """Test that encoded types actually become pipeline steps."""

    @staticmethod
    def _frame(n=200):
        rng = np.random.default_rng(3)
        return pd.DataFrame({
            'heavy': rng.standard_t(df=1.5, size=n),
            'wide': rng.normal(500, 250, n),
            'plain': rng.normal(0, 1, n),
        })

    @staticmethod
    def _recommendations():
        return pd.DataFrame({
            'feature': ['heavy', 'wide', 'plain'],
            'transformations': ['quantile/rank', 'standardize', 'none'],
            'reason': ['Heavy outliers', 'High variance', 'Well-behaved'],
        })

    def test_quantile_group_becomes_a_pipeline_step(self):
        config = encode_transformation_recommendation(self._recommendations())
        assert config['transformations']['heavy']['type'] == 'quantile'

        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        step_names = [name for name, _, _ in pipeline.named_steps['transforms'].transformers]
        assert 'quantile' in step_names

    def test_quantile_transform_actually_changes_the_data(self):
        frame = self._frame()
        pipeline = build_pipeline_from_config(encode_transformation_recommendation(self._recommendations()),
                                              scaler_type=None)
        out = pd.DataFrame(pipeline.fit_transform(frame), columns=frame.columns)
        # A heavy-tailed column mapped to a normal output should lose its extreme kurtosis.
        assert out['heavy'].abs().max() < frame['heavy'].abs().max()

    def test_standardize_is_left_to_the_global_scaler(self):
        """Stacking a per-column StandardScaler under the default RobustScaler would scale
        those columns twice."""
        config = encode_transformation_recommendation(self._recommendations())
        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        step_names = [name for name, _, _ in pipeline.named_steps['transforms'].transformers]
        assert 'standardize' not in step_names
        assert 'scaler' in pipeline.named_steps

    def test_standardize_is_materialised_when_there_is_no_global_scaler(self):
        config = encode_transformation_recommendation(self._recommendations())
        pipeline = build_pipeline_from_config(config, scaler_type=None)
        step_names = [name for name, _, _ in pipeline.named_steps['transforms'].transformers]
        assert 'standardize' in step_names
        assert 'scaler' not in pipeline.named_steps

    def test_standardize_is_applied_when_no_global_scaler(self):
        frame = self._frame()
        pipeline = build_pipeline_from_config(encode_transformation_recommendation(self._recommendations()),
                                              scaler_type=None)
        out = pd.DataFrame(pipeline.fit_transform(frame), columns=frame.columns)
        assert abs(out['wide'].mean()) < 0.1
        assert abs(out['wide'].std(ddof=0) - 1.0) < 0.1

    def test_pipeline_round_trips_through_a_saved_config(self, tmp_path):
        config_path = tmp_path / 'config.json'
        export_transformation_config(self._recommendations(), str(config_path))
        pipeline = build_pipeline_from_config(str(config_path), scaler_type=None)
        out = pipeline.fit_transform(self._frame())
        assert out.shape == (200, 3)


class TestConsoleOutputIsEncodable:
    """U+2713 and U+2192 are unencodable in cp1252, the default Windows console codepage, so
    every print here raised UnicodeEncodeError -- build_pipeline_from_config died partway
    through returning its pipeline. Same class of bug as the arrow fixed in 9bf4368."""

    @staticmethod
    def _assert_cp1252_safe(text):
        try:
            text.encode('cp1252')
        except UnicodeEncodeError as exc:
            raise AssertionError(f"console output is not encodable on a cp1252 terminal: {exc}") from exc

    def test_build_pipeline_output(self, sample_recommendations, capsys):
        config = encode_transformation_recommendation(sample_recommendations)
        build_pipeline_from_config(config, scaler_type='robust')
        self._assert_cp1252_safe(capsys.readouterr().out)

    def test_passthrough_pipeline_output(self, capsys):
        build_pipeline_from_config({'transformations': {}}, scaler_type=None)
        self._assert_cp1252_safe(capsys.readouterr().out)

    def test_scaler_only_pipeline_output(self, capsys):
        build_pipeline_from_config({'transformations': {}}, scaler_type='robust')
        self._assert_cp1252_safe(capsys.readouterr().out)

    def test_export_and_load_config_output(self, sample_recommendations, tmp_path, capsys):
        path = tmp_path / 'config.json'
        export_transformation_config(sample_recommendations, str(path))
        load_transformation_config(str(path))
        self._assert_cp1252_safe(capsys.readouterr().out)

    def test_artifact_save_and_load_output(self, sample_recommendations, sample_data, tmp_path, capsys):
        config = encode_transformation_recommendation(sample_recommendations)
        pipeline = build_pipeline_from_config(config, scaler_type='robust')
        pipeline.fit(sample_data)
        save_pipeline_artifacts(pipeline, list(sample_data.columns), str(tmp_path / 'artifacts'),
                                metadata={'version': '1'})
        load_pipeline_artifacts(str(tmp_path / 'artifacts'))
        self._assert_cp1252_safe(capsys.readouterr().out)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_literal_quantile_token_is_not_dropped():
    """'quantile/rank' mapped to 'quantile' but a bare 'quantile' fell through to passthrough,
    so re-encoding an already-encoded frame silently lost the transformation."""
    assert config_transformation_type('quantile') == 'quantile'
    assert config_transformation_type('quantile/rank') == 'quantile'
