"""
Receiver Operating Characteristics (ROC) toolkit for trading indicator validation.

Source / Attribution
--------------------
The statistical methodology in this package — ROC table generation, optimal threshold search, and
Monte Carlo Permutation Testing (MCPT) — is taken directly from the work of Timothy Masters:

    Timothy Masters, "Statistically Sound Indicators For Financial Market Prediction", Apress, 2013.

    C++ reference implementations: ROC.CPP, distributed with the book's companion code.

Python implementations were authored for this project following Masters' algorithmic descriptions and C++ source code.

Package contents
----------------
roc_analysis      ROCAnalyzer  – core ROC engine (table + MCPT)
roc_evaluator     BatchROCEvaluator – multi-indicator batch runner
signal_generator  SignalGenerator – convert indicators to signal files
stationarity      StationarityTester – structural-break / regime-change test
roc_plots         ROCVisualizer – static matplotlib visualisations
roc_dashboard     InteractiveROCDashboard – interactive Plotly/Dash dashboard
"""

from .roc_analysis import ROCAnalyzer, ROCResults
from .roc_evaluator import BatchROCEvaluator
from .signal_generator import SignalGenerator
from .stationarity import StationarityTester, BreakTestResult

__all__ = [
    "ROCAnalyzer",
    "ROCResults",
    "BatchROCEvaluator",
    "SignalGenerator",
    "StationarityTester",
    "BreakTestResult",
]