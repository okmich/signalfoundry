"""
okmich_quant_pipeline.directional_change
=========================================
End-to-end **Directional Change (DC) regime detection pipelines**.

Directional Change is an event-driven price sampling framework: instead of sampling prices at fixed time intervals, it
decomposes a price series into alternating up/down moves whenever price moves a threshold fraction *theta* from the last
confirmed extreme.  Each such move is a *DC trend* with measurable properties — Total Move Value (TMV), duration in
bars (T), and log(R) — that expose regime structure invisible in OHLCV.

Reference: Chen & Tsang (2021), "Detecting Regime Change in Computational
Finance", Chapters 3–6.

Relationship to okmich_quant_features.directional_change
---------------------------------------------------------
These two packages have a strict layered dependency:

    okmich_quant_features.directional_change   ← low-level primitives
    okmich_quant_pipeline.directional_change   ← pipelines built on top

``okmich_quant_features.directional_change`` provides the building blocks:
    - ``parse_dc_events``          parse a price series into DC trends (offline)
    - ``dc_live_features``         compute per-bar TMV / T without lookahead (live)
    - ``log_r``                    log(R) feature per trend (HMM input)
    - ``idc_parse``                Intrinsic DC (IDC) parser
    - ``extract_dc_classification_features``  CGP-style feature extraction
    - ``parse_dual_dc`` / ``label_bbtheta``   TSFDC dual-threshold primitives

``okmich_quant_pipeline.directional_change`` assembles those primitives into trained, saveable, and deployable pipelines
that output per-bar regime probabilities.  You do **not** use the features package directly in production — you call a pipeline.

Pipeline families (in order of complexity)
-------------------------------------------
1. **DCRegimePipeline** — baseline single-theta pipeline.
   DC parse → log(R) → HMM (Viterbi) → MinMaxScaler → sklearn classifier.
   Outputs P(Regime1), P(Regime2) per bar.  Save/load artifacts.
   Use this as the starting point.

2. **ITAPipeline** — Intrinsic Time Analysis.
   Fits theta via Bayesian optimisation of IDC scaling-law parameters.
   Use when you want the threshold to be data-driven rather than manual.

3. **TSFDCPipeline** — Two-State Fixed DC.
   Dual-threshold DC (theta_up, theta_down) + bbtheta classifier.
   Use when up and down moves have asymmetric volatility.

4. **MTDCPipeline** — Multi-Threshold DC.
   Runs DC across a pool of theta values; combines signals via GA-learned weights or consensus vote.
   Use for robustness across theta sensitivity.

5. **ATDCPipeline** — Adaptive Threshold DC.
   Dynamically adjusts theta via a fitness metric (AdaptationMode).
   Use in non-stationary regimes where a fixed theta drifts out of range.

Typical usage — baseline pipeline
-----------------------------------
    from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode, create_simple_hmm_instance

    from sklearn.naive_bayes import GaussianNB
    from okmich_quant_pipeline.directional_change import DCRegimePipeline

    hmm = create_simple_hmm_instance(DistType.NORMAL, n_states=2, inference_mode=InferenceMode.VITERBI)
    pipeline = DCRegimePipeline(theta=0.003, hmm=hmm, classifier=GaussianNB())
    pipeline.fit(train_prices)          # pd.Series of close prices
    pipeline.save("artifacts/dc_v1/")

    proba = pipeline.predict_proba(live_prices)
    # proba columns: p_regime1, p_regime2, direction, tmv_current,
    #                t_current, upward_dcc, downward_dcc

    # Load later
    pipeline2 = DCRegimePipeline.load("artifacts/dc_v1/")
    proba2 = pipeline2.predict_proba(live_prices)

Typical usage — multi-threshold pipeline
------------------------------------------
    from okmich_quant_pipeline.directional_change import MTDCPipeline, generate_theta_pool, FitnessMode, ConsensusMode

    theta_pool = generate_theta_pool()   # uses MTDC_THETA_*_DEFAULT bounds
    pipeline = MTDCPipeline(theta_pool=theta_pool, fitness_mode=FitnessMode.SHARPE)
    pipeline.fit(train_prices)
    proba = pipeline.predict_proba(live_prices)

Theta selection guidance
------------------------
- 5-minute FX:  start with theta=0.003 (0.3%).  Use THETA_BOUNDS_5MIN with
  ``optimise_idc_params`` to find the data-driven optimum.
- Tick data:    use THETA_BOUNDS_TICK.
- When unsure:  run MTDCPipeline across a pool and let the GA weight selection
  determine which thresholds carry the most signal.
"""
from ._pipeline import DCRegimePipeline
from ._rcd_hmm import assign_regime_labels, fit_rcd_hmm, get_current_regime, s1_state_index
from ._ita_sim import run_ita_simulation
from ._bayesian_opt import ALPHA_BOUNDS, THETA_BOUNDS_5MIN, THETA_BOUNDS_TICK, optimise_idc_params
from ._ita_algo1 import run_ita_algorithm1
from ._ita_pipeline import ITAPipeline, run_sliding_window
from ._cgp_gp import build_gp_toolbox, run_gp_regression
from ._cgpts_model import predict_trend_end, train_cgpts_model
from ._cgpts_algo import run_cgpts_algorithm
from ._tsfdc_classifier import predict_bbtheta, train_bbtheta_classifier
from ._tsfdc_engine import run_tsfdc_algorithm
from ._tsfdc_pipeline import TSFDCPipeline, run_tsfdc_sliding_window
from ._tsfdc_threshold_search import evaluate_threshold_pair, search_optimal_thresholds
from ._mtdc_thresholds import MTDC_THETA_MAX_DEFAULT, MTDC_THETA_MIN_DEFAULT, MTDC_THETA_STEP_DEFAULT, ConsensusMode, FitnessMode, generate_theta_pool, select_top_thresholds
from ._mtdc_ga import train_ga_weights
from ._mtdc_engine import run_mtdc_algorithm
from ._mtdc_pipeline import MTDCPipeline, run_mtdc_sliding_window, search_optimal_k
from ._atdc_adapter import AdaptationMode, compute_adapted_theta, compute_metric
from ._atdc_engine import run_atdc_algorithm
from ._atdc_pipeline import ATDCPipeline, run_atdc_sliding_window

__all__ = [
    "DCRegimePipeline",
    "fit_rcd_hmm",
    "s1_state_index",
    "assign_regime_labels",
    "get_current_regime",
    "run_ita_simulation",
    "optimise_idc_params",
    "THETA_BOUNDS_TICK",
    "THETA_BOUNDS_5MIN",
    "ALPHA_BOUNDS",
    "run_ita_algorithm1",
    "ITAPipeline",
    "run_sliding_window",
    "build_gp_toolbox",
    "run_gp_regression",
    "train_cgpts_model",
    "predict_trend_end",
    "run_cgpts_algorithm",
    "train_bbtheta_classifier",
    "predict_bbtheta",
    "run_tsfdc_algorithm",
    "TSFDCPipeline",
    "run_tsfdc_sliding_window",
    "evaluate_threshold_pair",
    "search_optimal_thresholds",
    "generate_theta_pool",
    "MTDC_THETA_MIN_DEFAULT",
    "MTDC_THETA_MAX_DEFAULT",
    "MTDC_THETA_STEP_DEFAULT",
    "select_top_thresholds",
    "FitnessMode",
    "ConsensusMode",
    "train_ga_weights",
    "run_mtdc_algorithm",
    "MTDCPipeline",
    "run_mtdc_sliding_window",
    "search_optimal_k",
    "AdaptationMode",
    "compute_metric",
    "compute_adapted_theta",
    "run_atdc_algorithm",
    "ATDCPipeline",
    "run_atdc_sliding_window",
]
