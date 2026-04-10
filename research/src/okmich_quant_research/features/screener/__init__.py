"""
Feature Screener
================
Empirical 5-stage funnel for data-driven feature selection.

    >>> from okmich_quant_research.features.screener import FeatureScreener
    >>> screener = FeatureScreener()
    >>> result = screener.screen_for_regimes(X, regime_labels)
    >>> result = screener.screen_for_returns(X, forward_returns, horizon=5)
    >>> print(result.confirmed)
    >>> print(result.top_features())
"""
from .screener import FeatureScreener
from ._result import ScreenerResult, StageReport

__all__ = ["FeatureScreener", "ScreenerResult", "StageReport"]