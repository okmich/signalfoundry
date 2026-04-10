"""
Diagnostic Labelers — For Analysis Only

All labelers in this module use future data (leaks_future=True).

Valid uses:
- Post-hoc analysis and visualization
- Understanding historical market structure
- HMM state alignment diagnostics
- Oracle upper-bound testing
- Feature engineering (use segment properties as features, not labels)

NOT valid for:
- Supervised ML training for live trading
- Backtesting (will overstate edge)

Submodules:
- trend: AmplitudeBasedLabeler, auto_label, CTL, Oracle
- returns: AutoLabelRegression, AmplitudeBasedRegressionLabeler (regression targets)
- ruptures: Changepoint detection labelers
"""

__all__ = []
