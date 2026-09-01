"""
Functional tests for transform module.

Tests both stateless functions and stateful transformer classes.
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from okmich_quant_features.utils.transform import (
    # Stateless functions
    logit_transform,
    log_transform,
    standardize,
    # Stateful transformers
    YeoJohnsonTransformer,
    BoxCoxTransformer,
    LogitTransformer,
    LogTransformer,
    # Helper functions
    get_transformer,
    primary_transformation,
    fit_transformation_recommendations,
    apply_transformation_recommendations,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_series():
    """Generate sample pandas Series."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.02, name='returns')


@pytest.fixture
def sample_array():
    """Generate sample numpy array."""
    np.random.seed(42)
    return np.random.randn(100) * 0.02


@pytest.fixture
def positive_series():
    """Generate positive pandas Series."""
    np.random.seed(42)
    return pd.Series(np.abs(np.random.randn(100)) * 10 + 50, name='atr')


@pytest.fixture
def bounded_series():
    """Generate bounded [0,1] pandas Series."""
    np.random.seed(42)
    return pd.Series(np.random.uniform(0.3, 0.7, 100), name='hurst')


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'returns': np.random.randn(100) * 0.02,
        'volume': np.abs(np.random.randn(100)) * 1000 + 1000,
        'hurst': np.random.uniform(0.3, 0.7, 100),
        'atr': np.abs(np.random.randn(100)) * 10 + 50,
    })


@pytest.fixture
def sample_recommendations():
    """Sample transformation recommendations."""
    return pd.DataFrame({
        'feature': ['returns', 'volume', 'hurst'],
        'transformations': ['yeo-johnson', 'log', 'logit'],
    })


# =============================================================================
# Test Stateless Functions
# =============================================================================

class TestStatelessFunctions:
    """Test stateless transformation functions."""

    def test_logit_transform_series(self, bounded_series):
        """Test logit transform with pandas Series."""
        result = logit_transform(bounded_series)

        assert isinstance(result, pd.Series)
        assert result.name == 'hurst'
        assert len(result) == len(bounded_series)
        assert not result.isnull().any()

    def test_logit_transform_array(self):
        """Test logit transform with numpy array."""
        data = np.array([0.3, 0.5, 0.7])
        result = logit_transform(data)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
        assert not np.isnan(result).any()

    def test_logit_transform_epsilon(self):
        """Test logit transform with custom epsilon."""
        data = np.array([0.0, 0.5, 1.0])
        result = logit_transform(data, epsilon=1e-6)

        assert not np.isinf(result).any()

    def test_log_transform_positive(self, positive_series):
        """Test log transform with positive values."""
        result = log_transform(positive_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(positive_series)
        assert not result.isnull().any()

    def test_log_transform_with_negatives(self):
        """Test log transform with negative values (auto-offset)."""
        data = pd.Series([-5, -2, 0, 3, 10])
        result = log_transform(data)

        assert not result.isnull().any()
        assert not np.isinf(result).any()

    def test_log_transform_custom_offset(self):
        """Test log transform with custom offset."""
        data = np.array([1, 2, 3, 4, 5])
        result = log_transform(data, offset=10)

        expected = np.log(data + 10)
        np.testing.assert_array_almost_equal(result, expected)

    def test_standardize_series(self, sample_series):
        """Test standardize with pandas Series."""
        result = standardize(sample_series)

        assert isinstance(result, pd.Series)
        assert np.abs(result.mean()) < 1e-10  # Mean ~0
        # Use ddof=0 for population std (same as numpy default)
        assert np.abs(result.std(ddof=0) - 1.0) < 1e-10  # Std ~1

    def test_standardize_array(self, sample_array):
        """Test standardize with numpy array."""
        result = standardize(sample_array)

        assert isinstance(result, np.ndarray)
        assert np.abs(result.mean()) < 1e-10
        assert np.abs(result.std() - 1.0) < 1e-10


# =============================================================================
# Test YeoJohnsonTransformer
# =============================================================================

class TestYeoJohnsonTransformer:
    """Test YeoJohnsonTransformer class."""

    def test_fit_transform(self, sample_series):
        """Test basic fit and transform."""
        transformer = YeoJohnsonTransformer()
        result = transformer.fit_transform(sample_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)

    def test_fit_transform_array(self, sample_array):
        """Test fit_transform with numpy array returns 2D."""
        transformer = YeoJohnsonTransformer()
        result = transformer.fit_transform(sample_array)

        # Should return 2D for sklearn compatibility
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (len(sample_array), 1)

    def test_separate_fit_transform(self, sample_series):
        """Test separate fit and transform calls."""
        transformer = YeoJohnsonTransformer()
        transformer.fit(sample_series)
        result = transformer.transform(sample_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_series)

    def test_inverse_transform(self, sample_series):
        """Test inverse transformation."""
        transformer = YeoJohnsonTransformer()
        transformed = transformer.fit_transform(sample_series)
        reconstructed = transformer.inverse_transform(transformed)

        # Should be close to original
        pd.testing.assert_series_equal(
            reconstructed, sample_series, check_exact=False, rtol=1e-5
        )

    def test_lambda_property(self, sample_series):
        """Test lambda parameter property."""
        transformer = YeoJohnsonTransformer()
        transformer.fit(sample_series)

        assert hasattr(transformer, 'lambda_')
        assert isinstance(transformer.lambda_, (float, np.floating))

    def test_standardize_option(self, sample_series):
        """Test with standardize=True."""
        transformer = YeoJohnsonTransformer(standardize=True)
        result = transformer.fit_transform(sample_series)

        assert isinstance(result, pd.Series)

    def test_sklearn_compatibility(self, sample_array):
        """Test compatibility with sklearn ColumnTransformer."""
        transformer = YeoJohnsonTransformer()

        # Should work in ColumnTransformer (requires 2D output)
        col_transformer = ColumnTransformer(
            [('yj', transformer, [0])],
            remainder='passthrough'
        )

        X = sample_array.reshape(-1, 1)
        result = col_transformer.fit_transform(X)

        assert result.shape[0] == len(sample_array)

    def test_joblib_serialization(self, sample_series, tmp_path):
        """Test saving and loading with joblib."""
        transformer = YeoJohnsonTransformer()
        transformer.fit(sample_series)

        # Save
        save_path = tmp_path / 'transformer.pkl'
        joblib.dump(transformer, save_path)

        # Load
        loaded = joblib.load(save_path)
        result = loaded.transform(sample_series)

        assert isinstance(result, pd.Series)


# =============================================================================
# Test BoxCoxTransformer
# =============================================================================

class TestBoxCoxTransformer:
    """Test BoxCoxTransformer class."""

    def test_fit_transform_positive(self, positive_series):
        """Test with strictly positive values."""
        transformer = BoxCoxTransformer()
        result = transformer.fit_transform(positive_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(positive_series)

    def test_negative_values_error(self, sample_series):
        """Test that negative values raise error."""
        transformer = BoxCoxTransformer()

        with pytest.raises(ValueError, match="strictly positive"):
            transformer.fit(sample_series)

    def test_inverse_transform(self, positive_series):
        """Test inverse transformation."""
        transformer = BoxCoxTransformer()
        transformed = transformer.fit_transform(positive_series)
        reconstructed = transformer.inverse_transform(transformed)

        pd.testing.assert_series_equal(
            reconstructed, positive_series, check_exact=False, rtol=1e-5
        )

    def test_lambda_property(self, positive_series):
        """Test lambda parameter property."""
        transformer = BoxCoxTransformer()
        transformer.fit(positive_series)

        assert hasattr(transformer, 'lambda_')
        assert isinstance(transformer.lambda_, (float, np.floating))

    def test_sklearn_compatibility(self, positive_series):
        """Test compatibility with sklearn ColumnTransformer."""
        transformer = BoxCoxTransformer()
        X = positive_series.values.reshape(-1, 1)

        col_transformer = ColumnTransformer(
            [('bc', transformer, [0])],
            remainder='passthrough'
        )

        result = col_transformer.fit_transform(X)
        assert result.shape[0] == len(positive_series)


# =============================================================================
# Test LogitTransformer
# =============================================================================

class TestLogitTransformer:
    """Test LogitTransformer class."""

    def test_fit_transform(self, bounded_series):
        """Test basic fit and transform."""
        transformer = LogitTransformer()
        result = transformer.fit_transform(bounded_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(bounded_series)
        assert not result.isnull().any()

    def test_custom_epsilon(self, bounded_series):
        """Test with custom epsilon."""
        transformer = LogitTransformer(epsilon=1e-6)
        result = transformer.fit_transform(bounded_series)

        assert isinstance(result, pd.Series)

    def test_array_returns_2d(self):
        """Test that array input returns 2D output."""
        data = np.array([0.3, 0.5, 0.7])
        transformer = LogitTransformer()
        result = transformer.fit_transform(data)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 1)

    def test_sklearn_compatibility(self, bounded_series):
        """Test compatibility with sklearn ColumnTransformer."""
        transformer = LogitTransformer()
        X = bounded_series.values.reshape(-1, 1)

        col_transformer = ColumnTransformer(
            [('logit', transformer, [0])],
            remainder='passthrough'
        )

        result = col_transformer.fit_transform(X)
        assert result.shape[0] == len(bounded_series)


# =============================================================================
# Test LogTransformer
# =============================================================================

class TestLogTransformer:
    """Test LogTransformer class."""

    def test_fit_transform_positive(self, positive_series):
        """Test with positive values."""
        transformer = LogTransformer()
        result = transformer.fit_transform(positive_series)

        assert isinstance(result, pd.Series)
        assert len(result) == len(positive_series)

    def test_fit_transform_with_negatives(self):
        """Test with negative values (auto-offset)."""
        data = pd.Series([-5, -2, 0, 3, 10])
        transformer = LogTransformer()
        result = transformer.fit_transform(data)

        assert not result.isnull().any()
        assert not np.isinf(result).any()

    def test_offset_computed(self):
        """Test that offset is computed correctly."""
        data = pd.Series([1, 2, 3, 4, 5])
        transformer = LogTransformer()
        transformer.fit(data)

        assert transformer.offset_ == 0  # All positive, no offset needed

    def test_offset_with_negatives(self):
        """Test offset computation with negative values."""
        data = pd.Series([-5, -2, 0, 3])
        transformer = LogTransformer()
        transformer.fit(data)

        assert transformer.offset_ > 0  # Offset should be positive

    def test_inverse_transform(self, positive_series):
        """Test inverse transformation."""
        transformer = LogTransformer()
        transformed = transformer.fit_transform(positive_series)
        reconstructed = transformer.inverse_transform(transformed)

        pd.testing.assert_series_equal(
            reconstructed, positive_series, check_exact=False, rtol=1e-10
        )

    def test_array_returns_2d(self):
        """Test that array input returns 2D output."""
        data = np.array([1, 2, 3, 4, 5])
        transformer = LogTransformer()
        result = transformer.fit_transform(data)

        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (5, 1)

    def test_sklearn_compatibility(self, positive_series):
        """Test compatibility with sklearn ColumnTransformer."""
        transformer = LogTransformer()
        X = positive_series.values.reshape(-1, 1)

        col_transformer = ColumnTransformer(
            [('log', transformer, [0])],
            remainder='passthrough'
        )

        result = col_transformer.fit_transform(X)
        assert result.shape[0] == len(positive_series)


# =============================================================================
# Test Helper Functions
# =============================================================================

class TestGetTransformer:
    """Test get_transformer factory function."""

    def test_yeo_johnson(self):
        """Test getting YeoJohnson transformer."""
        transformer = get_transformer('yeo-johnson')
        assert isinstance(transformer, YeoJohnsonTransformer)

    def test_box_cox(self):
        """Test getting BoxCox transformer."""
        transformer = get_transformer('box-cox')
        assert isinstance(transformer, BoxCoxTransformer)

    def test_logit(self):
        """Test getting Logit transformer."""
        transformer = get_transformer('logit')
        assert isinstance(transformer, LogitTransformer)

    def test_log(self):
        """Test getting Log transformer."""
        transformer = get_transformer('log')
        assert isinstance(transformer, LogTransformer)

    def test_with_kwargs(self):
        """Test passing kwargs to transformer."""
        transformer = get_transformer('yeo-johnson', standardize=True)
        assert transformer.standardize is True

    def test_invalid_type(self):
        """Test invalid transformation type."""
        with pytest.raises(ValueError, match="Unknown transformation"):
            get_transformer('invalid')


class TestApplyTransformationRecommendations:
    """Test apply_transformation_recommendations function."""

    def test_basic_application_replace_original(self, sample_dataframe, sample_recommendations):
        """Test applying transformations with replace_original=True (default)."""
        result = apply_transformation_recommendations(
            sample_dataframe,
            sample_recommendations,
            replace_original=True
        )

        # Should have same number of columns (replaced in-place)
        assert len(result.columns) == len(sample_dataframe.columns)
        # Original columns should exist but be transformed
        assert 'returns' in result.columns
        assert 'volume' in result.columns
        assert 'hurst' in result.columns
        # Should NOT have new suffixed columns
        assert 'returns_yeojohnson' not in result.columns
        assert 'volume_log' not in result.columns
        assert 'hurst_logit' not in result.columns

    def test_basic_application_keep_original(self, sample_dataframe, sample_recommendations):
        """Test applying transformations with replace_original=False."""
        result = apply_transformation_recommendations(
            sample_dataframe,
            sample_recommendations,
            replace_original=False
        )

        # Should have original + transformed columns
        assert len(result.columns) > len(sample_dataframe.columns)
        assert 'returns_yeojohnson' in result.columns
        assert 'volume_log' in result.columns
        assert 'hurst_logit' in result.columns

    def test_original_columns_preserved_when_not_replacing(self, sample_dataframe, sample_recommendations):
        """Test that original columns are preserved when replace_original=False."""
        result = apply_transformation_recommendations(
            sample_dataframe,
            sample_recommendations,
            replace_original=False
        )

        for col in sample_dataframe.columns:
            assert col in result.columns

    def test_none_transformation_skipped(self, sample_dataframe):
        """Test that 'none' transformations are skipped."""
        recommendations = pd.DataFrame({
            'feature': ['returns'],
            'transformations': ['none'],
        })

        result = apply_transformation_recommendations(
            sample_dataframe,
            recommendations,
            replace_original=True
        )

        # Columns should remain unchanged
        assert len(result.columns) == len(sample_dataframe.columns)
        # Original values should be unchanged
        pd.testing.assert_series_equal(result['returns'], sample_dataframe['returns'])

    def test_standardize_skipped(self, sample_dataframe):
        """Test that 'standardize' transformations are skipped."""
        recommendations = pd.DataFrame({
            'feature': ['returns'],
            'transformations': ['standardize'],
        })

        result = apply_transformation_recommendations(
            sample_dataframe,
            recommendations,
            replace_original=True
        )

        assert len(result.columns) == len(sample_dataframe.columns)
        # Original values should be unchanged
        pd.testing.assert_series_equal(result['returns'], sample_dataframe['returns'])

    def test_invalid_transformation_warning(self, sample_dataframe, capsys):
        """Test that invalid transformations print warning."""
        recommendations = pd.DataFrame({
            'feature': ['nonexistent_column'],
            'transformations': ['yeo-johnson'],
        })

        result = apply_transformation_recommendations(
            sample_dataframe,
            recommendations
        )

        captured = capsys.readouterr()
        assert 'Warning' in captured.out or 'Failed' in captured.out


# =============================================================================
# Test End-to-End Workflows
# =============================================================================

class TestEndToEndWorkflows:
    """Test complete transformation workflows."""

    def test_train_inference_workflow(self, sample_dataframe, tmp_path):
        """Test realistic training -> inference workflow."""
        # Training phase
        train_data = sample_dataframe.iloc[:80]
        test_data = sample_dataframe.iloc[80:]

        # Fit transformers on training data
        transformers = {}
        for col in ['returns', 'volume']:
            transformer = YeoJohnsonTransformer()
            transformer.fit(train_data[col])
            transformers[col] = transformer

        # Save transformers
        save_path = tmp_path / 'transformers.pkl'
        joblib.dump(transformers, save_path)

        # Inference phase (load and transform)
        loaded_transformers = joblib.load(save_path)

        for col, transformer in loaded_transformers.items():
            test_transformed = transformer.transform(test_data[col])
            assert len(test_transformed) == len(test_data)

    def test_pipeline_integration(self, sample_dataframe):
        """Test transformers in sklearn Pipeline."""
        from sklearn.preprocessing import StandardScaler

        # Create pipeline with custom transformer
        pipeline = Pipeline([
            ('yj', YeoJohnsonTransformer()),
            ('scaler', StandardScaler())
        ])

        # Fit on one column
        X = sample_dataframe['returns'].values.reshape(-1, 1)
        transformed = pipeline.fit_transform(X)

        assert transformed.shape == X.shape

    def test_column_transformer_multiple_features(self, sample_dataframe):
        """Test ColumnTransformer with multiple custom transformers."""
        col_transformer = ColumnTransformer([
            ('yj', YeoJohnsonTransformer(), ['returns']),
            ('log', LogTransformer(), ['volume']),
            ('logit', LogitTransformer(), ['hurst']),
        ])

        result = col_transformer.fit_transform(sample_dataframe)

        assert result.shape[0] == len(sample_dataframe)
        assert result.shape[1] == 3  # 3 transformed columns


def teardown_module(module):
    """Module-level teardown to cleanup any test artifacts."""
    import gc
    gc.collect()


# =============================================================================
# Test Inference Safety (fit once on train, transform anywhere)
# =============================================================================

class TestPrimaryTransformation:
    """Test the recommendation-string parser."""

    def test_single_token(self):
        assert primary_transformation('yeo-johnson') == 'yeo-johnson'

    def test_none_and_missing(self):
        assert primary_transformation('none') is None
        assert primary_transformation(None) is None
        assert primary_transformation(np.nan) is None

    def test_bare_standardize_is_not_fittable_here(self):
        """Scaling stays the pipeline scaler's job."""
        assert primary_transformation('standardize') is None

    def test_compound_string_keeps_the_fittable_transformation(self):
        """Regression: 'standardize' appearing anywhere used to void the whole row, so this
        recommendation silently passed the feature through untransformed."""
        assert primary_transformation('box-cox, standardize') == 'box-cox'
        assert primary_transformation('logit, quantile/rank, standardize') == 'logit'

    def test_first_token_wins(self):
        assert primary_transformation('log, box-cox') == 'log'


class TestFitTransformationRecommendations:
    """Test fitting transformers once for reuse."""

    def test_returns_one_fitted_transformer_per_feature(self, sample_dataframe, sample_recommendations):
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)
        assert set(fitted) == {'returns', 'volume', 'hurst'}

    def test_fitted_transformers_carry_state(self, sample_dataframe, sample_recommendations):
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)
        assert fitted['volume'].offset_ is not None
        assert fitted['returns'].lambda_ is not None

    def test_standardize_only_row_is_skipped(self, sample_dataframe):
        recommendations = pd.DataFrame({'feature': ['returns'], 'transformations': ['standardize']})
        assert fit_transformation_recommendations(sample_dataframe, recommendations) == {}

    def test_missing_column_warns_and_is_skipped(self, sample_dataframe, capsys):
        recommendations = pd.DataFrame({'feature': ['nope'], 'transformations': ['log']})
        fitted = fit_transformation_recommendations(sample_dataframe, recommendations)
        assert fitted == {}
        assert 'Warning' in capsys.readouterr().out

    def test_verbose_false_is_silent(self, sample_dataframe, capsys):
        recommendations = pd.DataFrame({'feature': ['nope'], 'transformations': ['log']})
        fit_transformation_recommendations(sample_dataframe, recommendations, verbose=False)
        assert capsys.readouterr().out == ''


class TestInferenceSafeApplication:
    """The train/serve skew this module exists to prevent."""

    @staticmethod
    def _shifted_batch():
        """An inference batch drawn from a visibly different distribution."""
        rng = np.random.default_rng(7)
        return pd.DataFrame({
            'returns': rng.normal(0.01, 0.05, 50),
            'volume': np.abs(rng.normal(0, 1, 50)) * 5000 + 8000,
            'hurst': rng.uniform(0.45, 0.55, 50),
            'atr': np.abs(rng.normal(0, 1, 50)) * 30 + 200,
        })

    def test_fitted_transformers_are_not_refitted(self, sample_dataframe, sample_recommendations):
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)
        lambda_before = fitted['returns'].lambda_
        offset_before = fitted['volume'].offset_

        apply_transformation_recommendations(self._shifted_batch(), sample_recommendations,
                                             fitted_transformers=fitted)

        assert fitted['returns'].lambda_ == lambda_before
        assert fitted['volume'].offset_ == offset_before

    def test_inference_output_differs_from_a_refit_on_the_batch(self, sample_dataframe, sample_recommendations):
        """If these matched, the fitted path would be doing nothing."""
        batch = self._shifted_batch()
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)

        safe = apply_transformation_recommendations(batch, sample_recommendations, fitted_transformers=fitted)
        refit = apply_transformation_recommendations(batch, sample_recommendations)

        # 'returns' is Yeo-Johnson, whose lambda is estimated from the data it is fitted on.
        # 'volume' is a log with a zero offset here (its values never reach zero), which happens
        # to be batch-invariant -- only the fitted-parameter transforms expose the skew.
        assert not np.allclose(safe['returns'].to_numpy(), refit['returns'].to_numpy())

    def test_a_bar_transforms_identically_regardless_of_its_batch(self, sample_dataframe, sample_recommendations):
        """The defining property of inference safety: batch composition must not change a row."""
        batch = self._shifted_batch()
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)

        whole = apply_transformation_recommendations(batch, sample_recommendations, fitted_transformers=fitted)
        first_ten = apply_transformation_recommendations(batch.iloc[:10], sample_recommendations,
                                                         fitted_transformers=fitted)

        pd.testing.assert_frame_equal(first_ten, whole.iloc[:10])

    def test_refitting_path_violates_that_property(self, sample_dataframe, sample_recommendations):
        """Sanity check on the test above -- proves it is actually measuring something."""
        batch = self._shifted_batch()
        whole = apply_transformation_recommendations(batch, sample_recommendations)
        first_ten = apply_transformation_recommendations(batch.iloc[:10], sample_recommendations)
        assert not np.allclose(first_ten['returns'].to_numpy(), whole['returns'].iloc[:10].to_numpy())

    def test_missing_fitted_transformer_raises_instead_of_refitting(self, sample_dataframe, sample_recommendations):
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)
        del fitted['volume']
        with pytest.raises(KeyError, match='volume'):
            apply_transformation_recommendations(sample_dataframe, sample_recommendations,
                                                 fitted_transformers=fitted)

    def test_empty_fitted_mapping_still_raises(self, sample_dataframe, sample_recommendations):
        """An empty dict is not None, so it means inference mode, not 'please fit'."""
        with pytest.raises(KeyError):
            apply_transformation_recommendations(sample_dataframe, sample_recommendations, fitted_transformers={})

    def test_survives_a_joblib_round_trip(self, sample_dataframe, sample_recommendations, tmp_path):
        import joblib
        batch = self._shifted_batch()
        fitted = fit_transformation_recommendations(sample_dataframe, sample_recommendations)
        expected = apply_transformation_recommendations(batch, sample_recommendations, fitted_transformers=fitted)

        path = tmp_path / 'transformers.pkl'
        joblib.dump(fitted, path)
        reloaded = apply_transformation_recommendations(batch, sample_recommendations,
                                                        fitted_transformers=joblib.load(path))

        pd.testing.assert_frame_equal(reloaded, expected)

    def test_compound_recommendation_is_applied_not_dropped(self, sample_dataframe):
        """Regression: 'box-cox, standardize' used to pass the feature through untouched."""
        recommendations = pd.DataFrame({'feature': ['atr'], 'transformations': ['box-cox, standardize']})
        result = apply_transformation_recommendations(sample_dataframe, recommendations)
        assert not np.allclose(result['atr'].to_numpy(), sample_dataframe['atr'].to_numpy())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
