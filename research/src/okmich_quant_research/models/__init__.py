"""
Quantitative Model Research Workflow

This module provides an end-to-end pipeline for training and evaluating machine learning models on financial time series
data with regime-aware backtesting.

Supports both:
- **Unsupervised learning**: HMM, clustering for regime detection
- **Supervised learning**: Classification/regression with walk-forward validation

Quick Start
-----------
    from okmich_quant_research.models import ExperimentRunner

    # Run complete experiment from config
    runner = ExperimentRunner.from_yaml('path/to/config.yaml')
    result = runner.run()

    # Get best model and predictions
    best_model = result.get_best_model()

Components
----------
Unsupervised:
- ModelTrainer: Train HMM/clustering models with hyperparameter configs
- RegimeEvaluator: Backtest models across different market regimes

Supervised:
- SupervisedTrainer: Train sklearn/Keras models with walk-forward validation
- SupervisedEvaluator: Evaluate classification/regression metrics

Common:
- FeatureStore: Cache computed features by symbol/timeframe/version
- FeatureSelector: Select optimal features using statistical methods
- ObjectivesEngine: Rank models by custom objectives
- ExperimentRunner: Orchestrate the full workflow
- ExperimentTracker: Persist and reload experiments
- ReportGenerator: Generate performance reports and visualizations

Workflow
--------
1. Load/cache features via FeatureStore
2. Select best features via FeatureSelector
3. (Supervised only) Generate targets via external function
4. Train models via ModelTrainer or SupervisedTrainer
5. Evaluate via RegimeEvaluator or SupervisedEvaluator
6. Rank by objectives via ObjectivesEngine
7. Track results via ExperimentTracker
8. Generate reports via ReportGenerator
"""

from .experiment_runner import ExperimentRunner, ExperimentResult
from .model_trainer import ModelTrainer, ModelMetadata, TrainedModel
from .regime_evaluator import RegimeEvaluator, RegimeEvaluationResult
from .regime_classification_trainer import RegimeClassificationTrainer
from .supervised_trainer import SupervisedTrainer, SupervisedTrainedModel, SupervisedModelMetadata, WalkForwardResult
from .supervised_evaluator import SupervisedEvaluator, SupervisedEvaluationResult
from .objectives import ObjectivesEngine, ModelRanking, ObjectiveScore
from .feature_store import FeatureStore, get_default_feature_store
from .feature_selector import FeatureSelector, FeatureSelectionResult
from .utils.config_parser import ConfigParser
from .experiment_tracker import ExperimentTracker, load_experiment
from .report_generator import ReportGenerator, generate_report
from .prediction_store import PredictionStore
from .model_monitor import ModelMonitor, ModelHealthStatus
from .model_registry import ModelRegistry
