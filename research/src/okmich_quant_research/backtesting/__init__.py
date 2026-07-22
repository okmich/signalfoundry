from .cluster_comparison_pipeline import ClusteringComparisonPipeline, ClusteringComparisonPipelineConfig
from .hmm_wfa_optimizer_with_backtester import HMMWalkForwardAnalysisBacktestOptimizer, HMMWindowResult
from .keras_models_wfa_optimizer import ModelWalkForwardAnalysisOptimizer, WindowResult
from .keras_models_wfa_optimizer_with_backtester import ModelWalkForwardAnalysisBacktestOptimizer, WindowBacktestResult
from .utils import generate_seq_feature_func_for_training, generate_seq_features_for_inference
from .vectorbt_backtester import VectorBtBacktester
from .regime_performance_analyzer import RegimePerformanceAnalyzer
from .temporal_performance_analyzer import TemporalPerformanceAnalyzer
from .signal_adapter import signal_to_portfolio, positions_to_signals
from .vbt_export import QuantframeExportMixin, FileFormat
from .timing_significance import (
    net_bar_returns,
    circular_shift_null,
    CircularShiftNull,
    beta_timing_decomposition,
    BetaTimingDecomposition,
)

# Listed in `__all__` so the alias is a declared export and lint autofix won't prune it.
__all__ = [
    "ClusteringComparisonPipeline",
    "ClusteringComparisonPipelineConfig",
    "HMMWalkForwardAnalysisBacktestOptimizer",
    "HMMWindowResult",
    "ModelWalkForwardAnalysisOptimizer",
    "WindowResult",
    "ModelWalkForwardAnalysisBacktestOptimizer",
    "WindowBacktestResult",
    "generate_seq_feature_func_for_training",
    "generate_seq_features_for_inference",
    "VectorBtBacktester",
    "RegimePerformanceAnalyzer",
    "TemporalPerformanceAnalyzer",
    "signal_to_portfolio",
    "positions_to_signals",
    "QuantframeExportMixin",
    "FileFormat",
    "net_bar_returns",
    "circular_shift_null",
    "CircularShiftNull",
    "beta_timing_decomposition",
    "BetaTimingDecomposition"
]
