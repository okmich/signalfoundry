"""Tests for the observation pipeline: aggregation, alternation, alphabet, buckets, flow."""
import numpy as np
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import (
    ALPHABET_SIZE,
    AssetGroup,
    MarketData,
    SignedVolumeFlowFeature,
    ZigzagDirection,
    ZigzagObservationPipeline,
    aggregate_zigzags,
    calibrate_k,
    events_per_hour,
    get_asset_group_config,
    realized_vol,
    vol_scaled_threshold,
)
from okmich_quant_ml.hmm.hierarchical.observations import bucketize, quantile_boundaries


class TestRealizedVolAndThreshold:
    def test_realized_vol_finite_and_nonneg_after_warmup(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        assert len(sigma) == len(price_series)
        fv = sigma.first_valid_index()
        assert fv is not None
        assert np.isfinite(sigma.loc[fv:]).all()
        assert (sigma.loc[fv:] >= 0).all()

    def test_realized_vol_is_causal_prefix_stable(self, price_series):
        # Forward-only fill => early bars must be identical whether or not future data exists.
        # (The old bfill seeded warmup bars from a *future* estimate and would fail this.)
        full = realized_vol(price_series, pd.Timedelta("24h"))
        t = len(price_series) * 2 // 3
        prefix = realized_vol(price_series.iloc[:t], pd.Timedelta("24h"))
        a, b = full.iloc[:t].to_numpy(), prefix.to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b))  # same NaN warmup pattern
        both = ~np.isnan(a)
        assert np.allclose(a[both], b[both], atol=1e-12)  # no future leakage into early bars

    def test_realized_vol_warmup_is_nan_not_backfilled(self):
        idx = pd.date_range("2026-01-01", periods=40, freq="1min", tz="UTC")
        s = pd.Series(100.0 * np.exp(np.cumsum(np.full(40, 1e-3))), index=idx)
        sigma = realized_vol(s, window=10, min_periods=5)
        fv = sigma.first_valid_index()
        warmup = sigma.loc[:fv].iloc[:-1]
        assert len(warmup) >= 1 and warmup.isna().all()  # leading bars stay NaN, never back-filled

    def test_timedelta_window_requires_datetime_index(self):
        s = pd.Series(np.arange(1, 100, dtype=float))  # RangeIndex
        with pytest.raises(TypeError):
            realized_vol(s, pd.Timedelta("24h"))

    def test_vol_scaled_threshold_scales_with_k(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        assert vol_scaled_threshold(2.0, sigma) == pytest.approx(2.0 * vol_scaled_threshold(1.0, sigma))

    def test_vol_scaled_threshold_rejects_bad_k(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        with pytest.raises(ValueError):
            vol_scaled_threshold(0.0, sigma)


class TestAggregation:
    def test_zigzags_strictly_alternate(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        theta = vol_scaled_threshold(1.5, sigma)
        zz = aggregate_zigzags(price_series, theta)
        assert len(zz) > 10
        dirs = [z.direction for z in zz]
        assert all(dirs[i] != dirs[i + 1] for i in range(len(dirs) - 1))

    def test_zigzag_magnitude_positive_and_causal(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        zz = aggregate_zigzags(price_series, vol_scaled_threshold(1.5, sigma))
        for z in zz:
            assert z.magnitude > 0
            # completion (confirmation) is at or after the closing pivot
            assert z.confirm_bar >= z.end_bar >= z.start_bar
            assert z.end_time >= z.start_time

    def test_larger_threshold_fewer_events(self, price_series):
        sigma = realized_vol(price_series, pd.Timedelta("24h"))
        few = aggregate_zigzags(price_series, vol_scaled_threshold(3.0, sigma))
        many = aggregate_zigzags(price_series, vol_scaled_threshold(1.0, sigma))
        assert len(few) < len(many)

    def test_theta_must_be_positive(self, price_series):
        with pytest.raises(ValueError):
            aggregate_zigzags(price_series, 0.0)


class TestCalibration:
    def test_calibrated_k_hits_target_band(self, price_series):
        cfg = get_asset_group_config(AssetGroup.FX_MAJORS)
        sigma = realized_vol(price_series, cfg.realized_vol_window)
        k = calibrate_k(price_series, sigma, cfg.target_events_per_hour)
        theta = vol_scaled_threshold(k, sigma)
        rate = events_per_hour(aggregate_zigzags(price_series, theta))
        lo, hi = cfg.target_events_per_hour
        # Achieved rate should be within a tolerant neighbourhood of the target band.
        assert lo * 0.5 <= rate <= hi * 1.5


class TestDiscretization:
    def test_quantile_boundaries_monotonic(self):
        scores = np.random.default_rng(0).normal(size=500)
        b = quantile_boundaries(scores, 3)
        assert len(b) == 2
        assert b[0] < b[1]

    def test_bucketize_monotonic(self):
        scores = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        b = quantile_boundaries(scores, 3)
        buckets = bucketize(scores, b)
        assert buckets.min() >= 0 and buckets.max() <= 2
        # sorted scores yield non-decreasing bucket indices
        assert np.all(np.diff(bucketize(np.sort(scores), b)) >= 0)

    def test_degenerate_all_equal(self):
        b = quantile_boundaries(np.full(50, 3.0), 3)
        assert b[0] < b[1]  # still strictly increasing
        buckets = bucketize(np.full(50, 3.0), b)
        assert len(np.unique(buckets)) == 1  # everything in one bucket


class TestPipeline:
    def test_symbols_in_alphabet_and_direction_matches(self, observations):
        assert observations.symbols.min() >= 0
        assert observations.symbols.max() < ALPHABET_SIZE
        for sym, z in zip(observations.symbols, observations.zigzags):
            assert sym // 9 == int(z.direction)

    def test_buckets_in_range(self, observations):
        assert set(np.unique(observations.strengths)).issubset({0, 1, 2})
        assert set(np.unique(observations.flows)).issubset({0, 1, 2})

    def test_transform_reproducible(self, fitted_pipeline, price_series):
        a = fitted_pipeline.transform(price_series)
        b = fitted_pipeline.transform(price_series)
        assert np.array_equal(a.symbols, b.symbols)

    def test_transform_before_fit_raises(self, price_series):
        pipe = ZigzagObservationPipeline(get_asset_group_config(AssetGroup.FX_MAJORS))
        with pytest.raises(RuntimeError):
            pipe.transform(price_series)

    def test_event_times_and_model_input(self, observations):
        assert observations.event_times.shape == (observations.n_zigzags,)
        X = observations.to_model_input()
        assert X.shape == (observations.n_zigzags, 1)
        assert X.dtype == np.int64

    def test_signed_volume_flow_requires_volume(self, price_series, fx_config):
        pipe = ZigzagObservationPipeline(fx_config, flow_feature=SignedVolumeFlowFeature())
        with pytest.raises(ValueError):
            pipe.fit(price_series)  # MarketData(close) only -> no volume

    def test_signed_volume_flow_path(self, price_series, fx_config):
        rng = np.random.default_rng(1)
        n = len(price_series)
        md = MarketData(
            close=price_series,
            high=price_series * (1 + np.abs(rng.normal(0, 5e-4, n))),
            low=price_series * (1 - np.abs(rng.normal(0, 5e-4, n))),
            volume=pd.Series(rng.lognormal(5, 0.5, n), index=price_series.index),
        )
        pipe = ZigzagObservationPipeline(fx_config, flow_feature=SignedVolumeFlowFeature())
        obs = pipe.fit_transform(price_series, md)
        assert obs.n_zigzags > 5
        assert set(np.unique(obs.flows)).issubset({0, 1, 2})
