"""Tests for the HHMM configuration contract: enums, alphabet, and asset-group configs."""
import pandas as pd
import pytest

from okmich_quant_ml.hmm.hierarchical import config as C
from okmich_quant_ml.hmm.hierarchical.config import (
    ALPHABET_SIZE,
    N_STATES,
    SUB_ALPHABET_SIZE,
    AssetGroup,
    FlowBucket,
    FlowFeatureKind,
    SessionPolicy,
    TrendStrength,
    ZigzagDirection,
    decode_symbol,
    direction_of_symbol,
    encode_symbol,
    get_asset_group_config,
    symbols_for_direction,
)


class TestAlphabet:
    def test_sizes(self):
        assert ALPHABET_SIZE == 18
        assert SUB_ALPHABET_SIZE == 9
        assert N_STATES == 4

    def test_encode_decode_round_trip(self):
        for d in ZigzagDirection:
            for s in TrendStrength:
                for f in FlowBucket:
                    sym = encode_symbol(d, s, f)
                    assert 0 <= sym < ALPHABET_SIZE
                    assert decode_symbol(sym) == (d, s, f)

    def test_all_18_symbols_unique(self):
        seen = {encode_symbol(d, s, f) for d in ZigzagDirection for s in TrendStrength for f in FlowBucket}
        assert seen == set(range(ALPHABET_SIZE))

    def test_direction_partition(self):
        assert list(symbols_for_direction(ZigzagDirection.DOWN)) == list(range(0, 9))
        assert list(symbols_for_direction(ZigzagDirection.UP)) == list(range(9, 18))
        for s in range(9):
            assert direction_of_symbol(s) is ZigzagDirection.DOWN
        for s in range(9, 18):
            assert direction_of_symbol(s) is ZigzagDirection.UP

    @pytest.mark.parametrize("bad", [-1, 18, 100])
    def test_decode_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            decode_symbol(bad)


class TestAssetGroupConfigs:
    def test_all_four_groups_present(self):
        for g in AssetGroup:
            cfg = get_asset_group_config(g)
            assert cfg.group is g
            assert cfg.k > 0
            lo, hi = cfg.target_events_per_hour
            assert 0 < lo <= hi

    def test_group_defaults_match_plan(self):
        fx = get_asset_group_config(AssetGroup.FX_MAJORS)
        assert fx.flow_feature is FlowFeatureKind.BOCPD
        assert fx.session_policy is SessionPolicy.SOFT
        assert fx.low_liquidity_windows  # SOFT carries at least one window
        assert 0 < fx.soft_downweight <= 1.0

        crypto = get_asset_group_config(AssetGroup.CRYPTO_CFD)
        assert crypto.session_policy is SessionPolicy.CONTINUOUS
        assert crypto.weekend_mode is True
        assert crypto.weekend_threshold_multiplier >= 1.0

        for g in (AssetGroup.INDEXES, AssetGroup.XAU):
            cfg = get_asset_group_config(g)
            assert cfg.session_policy is SessionPolicy.HARD

    def test_realized_vol_window_is_timedelta(self):
        for g in AssetGroup:
            assert isinstance(get_asset_group_config(g).realized_vol_window, pd.Timedelta)

    def test_all_groups_registry(self):
        registry = C.all_asset_group_configs()
        assert set(registry) == set(AssetGroup)


class TestConfigValidation:
    def test_negative_k_rejected(self):
        with pytest.raises(ValueError):
            C.AssetGroupConfig(
                group=AssetGroup.XAU, k=-1.0, target_events_per_hour=(3, 8),
                realized_vol_window=pd.Timedelta("8h"), flow_feature=FlowFeatureKind.BOCPD,
                session_policy=SessionPolicy.HARD, refit_cadence=(5, 10),
                refit_cadence_unit=C.RefitCadenceUnit.SESSIONS,
            )

    def test_low_liquidity_windows_require_soft(self):
        from datetime import time
        with pytest.raises(ValueError):
            C.AssetGroupConfig(
                group=AssetGroup.XAU, k=1.5, target_events_per_hour=(3, 8),
                realized_vol_window=pd.Timedelta("8h"), flow_feature=FlowFeatureKind.BOCPD,
                session_policy=SessionPolicy.HARD, refit_cadence=(5, 10),
                refit_cadence_unit=C.RefitCadenceUnit.SESSIONS,
                low_liquidity_windows=((time(22, 0), time(6, 0)),),
            )
