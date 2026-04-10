import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from okmich_quant_labelling.regime import (CausalRegimeLabeler, CausalStrategy, HmmDirectStrategy,
                                           HmmViterbiDistillationStrategy, MarketPropertyType)
from .feature_selector import FeatureSelector
from .model_trainer import ModelTrainer
from .objectives import ObjectivesEngine
from .prediction_store import PredictionStore
from .regime_evaluator import RegimeEvaluator, RegimeEvaluationResult
from .regime_classification_trainer import RegimeClassificationTrainer
from .supervised_trainer import SupervisedTrainer
from .supervised_evaluator import SupervisedEvaluator, SupervisedEvaluationResult
from .utils.config_parser import ConfigParser
from .utils.feature_logging import FeatureFunctionLogger
# ExperimentTracker and ReportGenerator imported lazily inside _workflow() to
# avoid circular imports (both modules import ExperimentResult from this module).


class ExperimentResult:

    def __init__(self, experiment_name: str, config: Dict[str, Any], features_df: pd.DataFrame,
                 selected_features: List[str], trained_models: List[Any], evaluation_result: Any, rankings: List[Any],
                 output_dir: str):
        self.experiment_name = experiment_name
        self.config = config
        self.features_df = features_df
        self.selected_features = selected_features
        self.trained_models = trained_models
        self.evaluation_result = evaluation_result
        self.rankings = rankings
        self.output_dir = output_dir

    def get_best_model(self):
        if self.rankings:
            return self.rankings[0]
        return None

    def get_labels_df(self, model_name: Optional[str] = None) -> pd.DataFrame:
        if model_name is None:
            best = self.get_best_model()
            if best is None:
                raise ValueError("No models available")
            model_name = best.model_name

        # Find the trained model
        for trained_model in self.trained_models:
            # Support both unsupervised (TrainedModel.labels) and
            # supervised (SupervisedTrainedModel.walk_forward_results)
            tm_name = (
                trained_model.metadata.model_name
                if hasattr(trained_model, "metadata")
                else getattr(trained_model, "model_name", None)
            )
            if tm_name != model_name:
                continue

            # Supervised model: reconstruct from walk-forward fold predictions
            if hasattr(trained_model, "walk_forward_results"):
                records = []
                for wf in trained_model.walk_forward_results:
                    n = len(wf.predictions)
                    records.append(
                        pd.DataFrame(
                            {"label": wf.predictions, "true_label": wf.true_labels},
                            index=pd.RangeIndex(n),  # positional; no DatetimeIndex stored
                        )
                    )
                if not records:
                    raise ValueError(
                        f"Supervised model '{model_name}' has no walk-forward results."
                    )
                return pd.concat(records, ignore_index=True)

            # Unsupervised model: global labels array aligned to features_df.
            # After reload features_df may be empty; fall back to positional index.
            if hasattr(trained_model, "labels") and trained_model.labels is not None:
                idx = (
                    self.features_df.index
                    if len(self.features_df) == len(trained_model.labels)
                    else pd.RangeIndex(len(trained_model.labels))
                )
                return pd.DataFrame({"label": trained_model.labels}, index=idx)

            raise ValueError(
                f"Model '{model_name}' has neither walk_forward_results nor labels. "
                "Cannot reconstruct labels."
            )

        raise ValueError(f"Model not found: {model_name}")


class ExperimentRunner:

    def __init__(self, config: ConfigParser):
        self.config = config
        self.output_dir = None
        self._setup_output_dir()

    def _setup_output_dir(self):
        experiment_name = self.config.get_experiment_name()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_folder = self.config.get_output_folder()
        if output_folder:
            base_dir = Path(output_folder)
        else:
            base_dir = Path("experiments/research") / f"{experiment_name}_{timestamp}"

        self.output_dir = base_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "features").mkdir(exist_ok=True)
        (self.output_dir / "eda").mkdir(exist_ok=True)
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "labels").mkdir(exist_ok=True)
        (self.output_dir / "metrics").mkdir(exist_ok=True)
        print(f"Output directory: {self.output_dir}")

    def run(self) -> ExperimentResult:
        """
        Run complete experiment.

        Returns
        -------
        ExperimentResult
            Complete experiment results
        """
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {self.config.get_experiment_name()}")
        print("=" * 80)

        # Save config
        config_path = self.output_dir / "config.yaml"
        self.config.save(str(config_path))
        print(f"[OK] Config saved to: {config_path}")

        # 1. Load data
        print("\n[STEP 1] Loading data...")
        data_df = self._load_data()
        print(f"   [OK] Loaded {len(data_df)} samples")
        return self._workflow(data_df)

    def run_with_data(self, data_df: pd.DataFrame) -> ExperimentResult:
        """
        Run experiment with pre-loaded data.

        Parameters
        ----------
        data_df : pd.DataFrame
            Pre-loaded market data with OHLCV columns

        Returns
        -------
        ExperimentResult
            Complete experiment results
        """
        print("\n" + "=" * 80)
        print(f"EXPERIMENT: {self.config.get_experiment_name()}")
        print("=" * 80)

        # Save config
        config_path = self.output_dir / "config.yaml"
        self.config.save(str(config_path))
        print(f"[OK] Config saved to: {config_path}")

        # Use provided data
        print(f"\n[STEP 1] Using provided data: {len(data_df)} samples")
        return self._workflow(data_df)

    def _load_data(self) -> pd.DataFrame:
        symbol = self.config.get_symbol()
        timeframe = self.config.get_timeframe()
        data_config = self.config.get_data_config()
        max_samples = self.config.get_max_samples()
        source_folder = data_config.get("source_folder")

        df = pd.read_parquet(Path(source_folder) / timeframe / f"{symbol}.parquet")
        return df if max_samples is None else df[-max_samples:]

    def _workflow(self, data_df):
        # 2. Feature engineering
        print("\n[STEP 2] Feature engineering...")
        features_df = self._engineer_features(data_df)
        print(f"   [OK] Generated {len(features_df.columns)} features")

        # Determine model type for branching
        model_type = self.config.get_model_type()
        is_supervised = model_type in ("supervised_classification", "supervised_regression")
        is_regime_classification = model_type == "regime_classification"

        # 2.5 Target engineering (for supervised learning only)
        target_series = None
        if is_supervised:
            print("\n[STEP 2.5] Target engineering...")
            target_series = self._engineer_targets(data_df)
            # Align target with features
            common_index = features_df.index.intersection(target_series.index)
            features_df = features_df.loc[common_index]
            target_series = target_series.loc[common_index]
            # Drop NaN targets
            valid_mask = ~target_series.isna()
            features_df = features_df.loc[valid_mask]
            target_series = target_series.loc[valid_mask]
            print(f"   [OK] Generated {len(target_series)} target samples")

        # 3. Feature selection
        print("\n[STEP 3] Feature selection...")
        selected_features = self._select_features(features_df)
        print(f"   [OK] Selected {len(selected_features)} features")

        # 4. Model training
        print("\n[STEP 4] Model training...")
        if is_supervised:
            trained_models = self._train_supervised(features_df, selected_features, target_series)
        elif is_regime_classification:
            trained_models = self._train_regime_classification(features_df, selected_features)
        else:
            trained_models = self._train_models(features_df, selected_features)
        print(f"   [OK] Trained {len(trained_models)} models")

        # 5. Evaluation
        if is_supervised:
            print("\n[STEP 5] Supervised evaluation...")
            task_type = self.config.get_task_type()
            evaluation_result = self._evaluate_supervised(trained_models, task_type)
        elif is_regime_classification:
            print("\n[STEP 5] Regime classification evaluation...")
            evaluation_result = self._evaluate_regime_classification(trained_models, features_df)
        else:
            print("\n[STEP 5] Regime evaluation...")
            evaluation_result = self._evaluate_regimes(features_df, trained_models)
        print(f"   [OK] Evaluated {len(trained_models)} models")

        # 6. Objectives-based ranking
        print("\n[STEP 6] Model ranking...")
        rankings = self._rank_models(evaluation_result)
        print(f"   [OK] Ranked {len(rankings)} models")

        # Create result
        result = ExperimentResult(
            experiment_name=self.config.get_experiment_name(),
            config=self.config.to_dict(),
            features_df=features_df,
            selected_features=selected_features,
            trained_models=trained_models,
            evaluation_result=evaluation_result,
            rankings=rankings,
            output_dir=str(self.output_dir),
        )

        # Save final summary
        self._save_summary(result)

        # 7. Register models in ModelRegistry
        print("\n[STEP 7] Registering models...")
        from .model_registry import ModelRegistry
        registry = ModelRegistry(
            registry_path=str(self.output_dir.parent / "model_registry.json")
        )
        exp_name = self.config.get_experiment_name()
        for tm in trained_models:
            model_path = str(
                self.output_dir / "models" / f"{tm.metadata.model_name}.joblib"
            )
            registry.register(
                experiment_name=exp_name,
                model_name=tm.metadata.model_name,
                metrics=evaluation_result.metrics_summary.get(
                    tm.metadata.model_name, {}
                ),
                model_path=model_path
                if Path(model_path).exists()
                else None,
                tags={
                    "model_type": model_type,
                    "research_type": self.config.get_research_type(),
                },
            )
        best_model = result.get_best_model()
        if best_model:
            registry.promote(exp_name, best_model.model_name)
            print(f"   [OK] Registered {len(trained_models)} models; promoted: {best_model.model_name}")
        else:
            print(f"   [OK] Registered {len(trained_models)} models (no best model to promote)")

        # 8. Model health check (ModelMonitor)
        if self.config.should_save_predictions() and best_model:
            print("\n[STEP 8] Model health check...")
            from .model_monitor import ModelMonitor
            store = PredictionStore(
                store_root=str(self.output_dir.parent / "prediction_store")
            )
            key = f"{exp_name}__{best_model.model_name}"
            try:
                preds_df = store.load(key)
                monitor = ModelMonitor()
                if model_type == "regime_classification":
                    labels = pd.Series(preds_df["predicted"].values)
                    status, health_metrics = monitor.check_regime_health(labels)
                else:
                    predictions = pd.Series(preds_df["predicted"].values)
                    actuals = pd.Series(preds_df["actual"].values)
                    if actuals.notna().sum() >= 10:
                        status, health_metrics = monitor.check_regression_health(
                            predictions, actuals
                        )
                    else:
                        status, health_metrics = None, {}
                if status is not None:
                    print(f"   [OK] {best_model.model_name} health: {status.value}")
            except Exception as e:
                print(f"   [WARN] Health check skipped: {e}")

        # 9. Track experiment in persistent index (lazy import avoids circular dep)
        from .experiment_tracker import ExperimentTracker
        print("\n[STEP 9] Tracking experiment...")
        tracker = ExperimentTracker(experiments_root=str(self.output_dir.parent))
        tracker.save_experiment(
            result,
            tags=[self.config.get_model_type(), self.config.get_research_type()],
        )

        # 10. Generate HTML report (lazy import avoids circular dep)
        if self.config.should_generate_report():
            from .report_generator import ReportGenerator
            print("\n[STEP 10] Generating report...")
            ReportGenerator().generate_report(
                result, output_path=str(self.output_dir / "report.html")
            )

        print("\n" + "=" * 80)
        print("EXPERIMENT COMPLETE")
        print("=" * 80)

        best_model = result.get_best_model()
        if best_model is not None:
            print(f"\nBest model: {best_model.model_name}")
            print(f"Score: {best_model.composite_score:.4f}")
        else:
            print("\nNo models were trained")
        print(f"\nResults saved to: {self.output_dir}")
        return result

    def _engineer_features(self, data_df: pd.DataFrame) -> pd.DataFrame:
        # Get feature engineering config
        external_fn = self.config.get_external_function_config()

        if external_fn is None:
            raise ValueError("No feature engineering function configured")

        module_path = external_fn.get("module")
        function_name = external_fn.get("function")
        params = external_fn.get("params", {})

        # Import and execute function
        print(f"   Loading: {module_path}.{function_name}")

        # Add repo root to sys.path so playground.* modules are importable
        # experiment_runner.py lives 6 levels deep inside the repo:
        #   <repo>/projects/research/src/okmich_quant_research/models/experiment_runner.py
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        module = importlib.import_module(module_path)
        feature_fn = getattr(module, function_name)

        # Execute feature function
        features_df = feature_fn(data_df, **params)

        # Merge with original data (ensure we have OHLC, return, etc.)
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in features_df.columns and col in data_df.columns:
                features_df[col] = data_df[col]

        # Add return column if missing
        if "return" not in features_df.columns:
            if "return" in data_df.columns:
                features_df["return"] = data_df["return"]
            elif "close" in data_df.columns:
                features_df["return"] = np.log(
                    data_df["close"] / data_df["close"].shift(1)
                )

        # Drop NaN rows
        features_df = features_df.dropna()
        print(f"   After dropping NaN: {len(features_df)} samples")

        # Log feature function metadata
        fe_config = self.config.get_feature_engineering_config()
        save_source = fe_config.get("save_source_code", False)
        version = fe_config.get("version", "v1")

        logger = FeatureFunctionLogger()
        logger.log_function(
            module=module_path,
            function=function_name,
            params=params,
            version=version,
            save_source=save_source,
        )

        # Save metadata
        metadata_path = self.output_dir / "feature_metadata.json"
        logger.save_metadata(str(metadata_path))

        # Save source code if requested
        if save_source:
            source_path = self.output_dir / "feature_source.py"
            logger.save_source_code(str(source_path))

        print(f"   [OK] Feature metadata saved")

        return features_df

    def _select_features(self, features_df: pd.DataFrame) -> List[str]:
        """Select features using automated selection."""
        auto_selection = self.config.get_auto_selection_enabled()

        if not auto_selection:
            # Use all features (excluding OHLC and return)
            exclude_cols = ["open", "high", "low", "close", "return"]
            selected = [col for col in features_df.columns if col not in exclude_cols]

            # Save simple selection file for tracker
            features_dir = self.output_dir / "features"
            features_dir.mkdir(exist_ok=True)
            import json

            with open(features_dir / "selected_features.json", "w") as f:
                json.dump(
                    {
                        "selected_features": selected,
                        "n_selected": len(selected),
                        "auto_selection_enabled": False,
                    },
                    f,
                    indent=2,
                )

            return selected

        # Get selection parameters
        top_n = self.config.get_auto_selection_top_n()
        vif_threshold = self.config.get_auto_selection_vif_threshold()
        min_importance = self.config.get_auto_selection_min_importance()

        # EDA config
        eda_enabled = self.config.get_eda_enabled()
        save_plots = self.config.get_eda_save_plots() if eda_enabled else False

        # Prepare features (exclude OHLC and return from selection)
        exclude_cols = ["open", "high", "low", "close", "return"]
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]
        features_for_selection = features_df[feature_cols]

        # Create synthetic target for EDA (use returns if available, otherwise close)
        if "return" in features_df.columns:
            target = features_df["return"]
        elif "close" in features_df.columns:
            target = features_df["close"].pct_change().fillna(0)
        else:
            # Fallback: use first feature as target
            target = features_for_selection.iloc[:, 0]

        # Run feature selection
        selector = FeatureSelector(
            features=features_for_selection, target=target, target_type="continuous"
        )

        result = selector.select(
            top_n=top_n,
            vif_threshold=vif_threshold,
            min_importance=min_importance,
            save_eda_artifacts=save_plots,
            output_dir=str(self.output_dir / "eda"),
        )

        # Save selection results
        result.save(str(self.output_dir / "features"))

        return result.selected_features

    def _engineer_targets(self, data_df: pd.DataFrame) -> pd.Series:
        """Generate targets using external function (for supervised learning)."""
        target_config = self.config.get_target_external_function_config()

        if target_config is None:
            raise ValueError("No target engineering function configured")

        module_path = target_config.get("module")
        function_name = target_config.get("function")
        params = target_config.get("params", {})

        # Import and execute function
        print(f"   Loading: {module_path}.{function_name}")

        # Add repo root to sys.path (same logic as _engineer_features)
        project_root = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        module = importlib.import_module(module_path)
        target_fn = getattr(module, function_name)

        # Execute target function
        target_series = target_fn(data_df, **params)

        # Ensure it's a Series
        if isinstance(target_series, pd.DataFrame):
            target_col = self.config.get_target_column()
            if target_col in target_series.columns:
                target_series = target_series[target_col]
            else:
                target_series = target_series.iloc[:, 0]

        # Log target function metadata
        te_config = self.config.get_target_engineering_config()
        logger = FeatureFunctionLogger()
        logger.log_function(
            module=module_path,
            function=function_name,
            params=params,
            version=te_config.get("version", "v1"),
            save_source=te_config.get("save_source_code", False),
        )

        # Save metadata
        metadata_path = self.output_dir / "target_metadata.json"
        logger.save_metadata(str(metadata_path))

        print(f"   [OK] Target metadata saved")
        return target_series

    def _train_supervised(self, features_df: pd.DataFrame, selected_features: List[str],
                          target_series: pd.Series) -> List[Any]:
        """Train supervised learning models with walk-forward validation."""
        task_type = self.config.get_task_type()
        framework = self.config.get_model_framework()

        trainer = SupervisedTrainer(random_state=42, scale_features=True)

        # Get walk-forward parameters
        train_period = self.config.get_train_period()
        test_period = self.config.get_test_period()
        step_period = self.config.get_step_period()
        anchored = self.config.is_anchored_walk_forward()
        max_train_bars = self.config.get_max_train_period()
        embargo_bars = self.config.get_embargo_bars()

        # Get model configs
        model_configs = self.config.get_sklearn_models()
        keras_builder = self.config.get_keras_model_builder()
        tuner_params = self.config.get_tuner_params()
        keras_training_params = self.config.get_keras_training_params()

        # Get sequence_length for Keras RNN models (must be resolved before
        # placeholder substitution in builder_params below)
        sequence_length = self.config.get_sequence_length()

        # Load keras model builder function if needed
        model_builder_fn = None
        if framework == "keras" and keras_builder:
            module = importlib.import_module(keras_builder["module"])
            builder_factory = getattr(module, keras_builder["function"])

            # Check if this is a builder factory (needs parameters)
            # or a direct model builder function
            builder_params = keras_builder.get("params", {})
            if builder_params:
                # Factory pattern: call with params to get actual builder
                # Substitute dynamic values
                num_features = len(selected_features)
                num_classes = len(target_series.dropna().unique())

                # Replace placeholders with actual values
                resolved_params = {}
                for key, value in builder_params.items():
                    if value == "__num_features__":
                        resolved_params[key] = num_features
                    elif value == "__num_classes__":
                        resolved_params[key] = num_classes
                    elif value == "__sequence_length__":
                        resolved_params[key] = sequence_length
                    else:
                        resolved_params[key] = value

                model_builder_fn = builder_factory(**resolved_params)
            else:
                # Direct model builder function
                model_builder_fn = builder_factory

        trained_models = trainer.walk_forward_train(
            features_df=features_df,
            target_series=target_series,
            feature_cols=selected_features,
            train_period=train_period,
            test_period=test_period,
            step_period=step_period,
            model_configs=model_configs,
            model_builder_fn=model_builder_fn,
            tuner_params=tuner_params,
            keras_training_params=keras_training_params,
            task_type=task_type,
            framework=framework,
            anchored=anchored,
            sequence_length=sequence_length,
            max_train_bars=max_train_bars,
            embargo_bars=embargo_bars,
        )

        # Save models
        if self.config.get_output_save_models():
            trainer.save_models(trained_models, str(self.output_dir / "models"))

        # Persist predictions
        if self.config.should_save_predictions():
            self._save_predictions(trained_models)
        return trained_models

    def _evaluate_supervised(self, trained_models: List[Any], task_type: str) -> Any:
        """Evaluate supervised learning models."""
        evaluator = SupervisedEvaluator()
        result = evaluator.evaluate(trained_models, task_type=task_type)

        # Save evaluation results
        result.save(str(self.output_dir / "metrics"))

        return result

    def _build_hmm(self, hmm_config: dict):
        """Instantiate a PomegranateHMM from config dict."""
        import inspect
        algorithm = hmm_config.get("algorithm", "PomegranateHMM")
        n_states = hmm_config.get("n_states", 3)
        distribution = hmm_config.get("distribution", "normal")

        module = importlib.import_module("okmich_quant_ml.hmm")
        cls = getattr(module, algorithm)
        sig = inspect.signature(cls)
        params: Dict[str, Any] = {}
        if "n_states" in sig.parameters:
            params["n_states"] = n_states
        if "distribution" in sig.parameters:
            params["distribution"] = distribution
        return cls(**params)

    def _build_label_generator(self):
        strategy = self.config.get_labelling_strategy().upper()
        yardstick = self.config.get_labelling_yardstick()
        acknowledges_lookahead = self.config.get_labelling_acknowledges_lookahead()

        if strategy == "A":
            hmm = self._build_hmm(self.config.get_labelling_hmm_config())
            return HmmDirectStrategy(hmm=hmm, yardstick=yardstick)

        if strategy == "B2":
            hmm = self._build_hmm(self.config.get_labelling_hmm_config())
            return HmmViterbiDistillationStrategy(hmm=hmm, yardstick=yardstick,
                                                  acknowledges_lookahead=acknowledges_lookahead)

        if strategy == "C":
            causal_cfg = self.config.get_labelling_causal_config()
            if isinstance(yardstick, str):
                normalized_yardstick = MarketPropertyType(yardstick.lower())
            elif isinstance(yardstick, MarketPropertyType):
                normalized_yardstick = yardstick
            else:
                raise TypeError(f"Unsupported yardstick type: {type(yardstick).__name__}")

            labeler = CausalRegimeLabeler(
                yardstick=normalized_yardstick,
                metric_window=causal_cfg.get("metric_window", 20),
                lookback_window=causal_cfg.get("lookback_window", 100),
                upper_pct=causal_cfg.get("upper_pct", 0.70),
                lower_pct=causal_cfg.get("lower_pct", 0.30),
                min_persistence=causal_cfg.get("min_persistence", 3),
                min_r_squared=causal_cfg.get("min_r_squared", 0.30),
                use_atr=causal_cfg.get("use_atr", True),
            )
            return CausalStrategy(labeler=labeler)

        raise ValueError(
            f"Strategy '{strategy}' not supported via YAML config. "
            "Supported: A, B2, C. (Strategy B requires a custom oracle object.)"
        )

    def _train_regime_classification(self, features_df: pd.DataFrame, selected_features: List[str]) -> List[Any]:
        """Train walk-forward regime classification models."""
        label_generator = self._build_label_generator()
        trainer = RegimeClassificationTrainer(
            label_generator=label_generator,
            random_state=42,
            scale_features=True,
        )

        strategy = self.config.get_labelling_strategy().upper()
        classifier_configs = self.config.get_regime_classifier_configs() if strategy != "A" else None

        trained_models = trainer.walk_forward_train(
            features_df=features_df,
            feature_cols=selected_features,
            price_col=self.config.get_labelling_price_col(),
            return_col=self.config.get_labelling_return_col(),
            train_period=self.config.get_train_period(),
            test_period=self.config.get_test_period(),
            step_period=self.config.get_step_period(),
            anchored=self.config.is_anchored_walk_forward(),
            max_train_bars=self.config.get_max_train_period(),
            embargo_bars=self.config.get_embargo_bars(),
            model_configs=classifier_configs,
        )

        if self.config.get_output_save_models():
            SupervisedTrainer(random_state=42).save_models(trained_models, str(self.output_dir / "models"))

        # Persist predictions
        if self.config.should_save_predictions():
            self._save_predictions(trained_models)

        return trained_models

    def _evaluate_regime_classification(self, trained_models: List[Any], features_df: pd.DataFrame) -> Any:
        """
        Evaluate regime classification models.

        Strategy A (pure HMM): reconstructs label series from walk-forward test predictions and routes through
        RegimeEvaluator for persistence/Sharpe metrics.

        Strategies B/B2/C (classifiers): aggregates per-fold val_accuracy / val_f1_weighted from
        WalkForwardResult.metrics into a SupervisedEvaluationResult.
        true_labels are always NaN for these strategies (labels were generated in-sample), so we cannot compute aggregate sklearn metrics.
        """
        strategy = self.config.get_labelling_strategy().upper()
        if strategy == "A":
            return self._evaluate_hmm_direct(trained_models, features_df)
        return self._evaluate_classifier_regime(trained_models)

    def _evaluate_hmm_direct(self, trained_models: List[Any], features_df: pd.DataFrame) -> Any:
        """Strategy A: reconstruct label series → RegimeEvaluator."""
        return_col = self.config.get_labelling_return_col()
        price_col = self.config.get_labelling_price_col()

        required_cols = [c for c in [price_col, return_col] if c in features_df.columns]
        labels_df = features_df[required_cols].copy()
        label_cols = []

        for tm in trained_models:
            fold_series = []
            for wf in tm.walk_forward_results:
                try:
                    test_slice = features_df.loc[wf.test_start:wf.test_end]
                    if len(test_slice) == len(wf.predictions):
                        fold_series.append(
                            pd.Series(wf.predictions, index=test_slice.index)
                        )
                except Exception:
                    pass
            if fold_series:
                col = tm.metadata.model_name
                labels_df[col] = pd.concat(fold_series)
                label_cols.append(col)

        if label_cols:
            evaluator = RegimeEvaluator()
            result = evaluator.evaluate(df=labels_df, label_cols=label_cols,
                                        returns_col=return_col, price_col=price_col,
                                        include_regime_returns=True)
        else:
            print("   [WARN] Strategy A: no recoverable test predictions — empty result")
            result = RegimeEvaluationResult(path_structure_stats=pd.DataFrame(), regime_returns_stats=None,
                                            label_mapping=None,
                                            metrics_summary={tm.metadata.model_name: {} for tm in trained_models})

        result.save(str(self.output_dir / "metrics"))
        return result

    def _evaluate_classifier_regime(self, trained_models: List[Any]) -> Any:
        """
        Strategies B/B2/C: aggregate per-fold metrics from WalkForwardResult.metrics.

        true_labels are always NaN for regime classifiers (labels are generated
        in-sample and not available on the test fold for B/B2).  For Strategy C
        (causal), val_accuracy / val_f1_weighted are pre-computed inside the
        training loop and stored in wf.metrics.
        """
        print("\nEvaluating regime classification models (per-fold metrics)...")
        metrics_summary: Dict[str, Any] = {}
        per_window_records = []

        for tm in trained_models:
            model_name = tm.metadata.model_name
            print(f"\n[{model_name}]")
            fold_metrics_list = []

            for wf in tm.walk_forward_results:
                per_window_records.append({
                    "model_name": model_name,
                    "window_idx": wf.window_idx,
                    "train_start": wf.train_start,
                    "train_end": wf.train_end,
                    "test_start": wf.test_start,
                    "test_end": wf.test_end,
                    "n_train": wf.n_train_samples,
                    "n_test": wf.n_test_samples,
                    **wf.metrics,
                })
                if wf.metrics:
                    fold_metrics_list.append(wf.metrics)

            # Average numeric metrics across folds — union keys across ALL folds
            # so metrics absent from fold 0 but present in later folds are not lost
            agg: Dict[str, float] = {}
            if fold_metrics_list:
                all_keys: set = set()
                for m in fold_metrics_list:
                    all_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
                for k in all_keys:
                    vals = [
                        float(m[k]) for m in fold_metrics_list
                        if k in m and not np.isnan(float(m[k]))
                    ]
                    agg[k] = float(np.mean(vals)) if vals else np.nan

            metrics_summary[model_name] = agg
            for k, v in agg.items():
                print(f"   {k}: {v:.4f}" if not np.isnan(v) else f"   {k}: N/A")

        result = SupervisedEvaluationResult(
            metrics_summary=metrics_summary,
            per_window_metrics=pd.DataFrame(per_window_records),
        )
        result.save(str(self.output_dir / "metrics"))
        return result

    def _save_predictions(self, trained_models: List[Any]) -> None:
        """Persist walk-forward predictions to PredictionStore."""
        exp_name = self.config.get_experiment_name()
        store = PredictionStore(
            store_root=str(self.output_dir.parent / "prediction_store")
        )
        saved = 0
        for tm in trained_models:
            fold_dfs = []
            for wf in getattr(tm, "walk_forward_results", []):
                n = len(wf.predictions)
                true = wf.true_labels[:n] if len(wf.true_labels) >= n else np.full(n, np.nan)
                fold_dfs.append(pd.DataFrame({
                    "fold_idx": wf.window_idx,
                    "predicted": wf.predictions,
                    "actual": true,
                    "model_name": tm.metadata.model_name,
                    "test_start": wf.test_start,
                    "test_end": wf.test_end,
                }))
            if fold_dfs:
                df = pd.concat(fold_dfs, ignore_index=True)
                store.save(f"{exp_name}__{tm.metadata.model_name}", df, overwrite=True)
                saved += 1
        print(f"   [OK] Predictions saved ({saved} models -> PredictionStore)")

    def _train_models(self, features_df: pd.DataFrame, selected_features: List[str]) -> List[Any]:
        """Train models."""
        model_type = self.config.get_model_type()

        trainer = ModelTrainer()

        if model_type == "clustering":
            trained_models = trainer.train_clustering(
                features_df=features_df,
                feature_cols=selected_features,
                algorithms=self.config.get_model_algorithms(),
                n_clusters_range=self.config.get_model_n_states_range(),
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Save models
        if self.config.get_output_save_models():
            trainer.save_models(trained_models, str(self.output_dir / "models"))
        return trained_models

    def _evaluate_regimes(self, features_df: pd.DataFrame, trained_models: List[Any]) -> Any:
        # Create labels DataFrame with OHLC and return
        required_cols = ["open", "high", "low", "close", "return"]
        available_cols = [col for col in required_cols if col in features_df.columns]
        labels_df = features_df[available_cols].copy()

        for model in trained_models:
            labels_df[model.metadata.model_name] = model.labels

        # Get label columns
        label_cols = [m.metadata.model_name for m in trained_models]

        # Determine if supervised or unsupervised
        research_type = self.config.get_research_type()

        evaluator = RegimeEvaluator()

        if research_type == "supervised":
            result = evaluator.evaluate(
                df=labels_df,
                label_cols=label_cols,
                returns_col="return",
                price_col="close",
                include_regime_returns=True,
            )
        elif research_type == "unsupervised":
            # Get unsupervised mapping params
            mapping_config = self.config.get_evaluation_unsupervised_label_mapping()
            params = mapping_config.get("params", {}) if mapping_config else {}

            # Extract mapping type and type-specific parameters
            mapping_type = params.get("mapping_type", "trend")

            # Trend mapping parameters
            trend_method = params.get("trend_method", "conservative")
            cost_threshold = params.get("cost_threshold", None)
            min_sharpe = params.get("min_sharpe", 0.3)

            # Volatility mapping parameters
            vol_proxy_col = params.get("vol_proxy_col", None)

            # Path structure mapping parameters
            lookback = params.get("lookback", 14)

            # Check if regime_returns_potentials is enabled
            eval_config = self.config.to_dict().get("evaluation", {})
            regime_returns_config = eval_config.get("regime_returns_potentials", {})
            include_regime_returns = regime_returns_config.get("enabled", True)

            # Note: momentum_range and choppiness_range are auto-inferred from n_states
            # in evaluate_unsupervised() based on the actual number of unique labels

            result = evaluator.evaluate_unsupervised(
                df=labels_df,
                label_cols=label_cols,
                returns_col="return",
                price_col="close",
                mapping_type=mapping_type,
                trend_method=trend_method,
                cost_threshold=cost_threshold,
                min_sharpe=min_sharpe,
                vol_proxy_col=vol_proxy_col,
                lookback=lookback,
                include_regime_returns=include_regime_returns,
            )
        else:
            raise ValueError(f"Unknown research type: {research_type}")

        # Save evaluation results
        result.save(str(self.output_dir / "metrics"))

        # Save labels
        if self.config.get_output_save_labels():
            labels_output = self.output_dir / "labels"
            for model in trained_models:
                model_labels = labels_df[[model.metadata.model_name]].copy()
                model_labels.to_parquet(
                    labels_output / f"{model.metadata.model_name}_labels.parquet"
                )

        return result

    def _rank_models(self, evaluation_result: Any) -> List[Any]:
        objectives = self.config.get_objectives_primary()

        engine = ObjectivesEngine(objectives=objectives)

        # Get regime_stats_df if it exists (unsupervised) or None (supervised)
        regime_stats_df = getattr(evaluation_result, "path_structure_stats", None)

        rankings = engine.rank_models(
            metrics_summary=evaluation_result.metrics_summary,
            regime_stats_df=regime_stats_df,
        )

        # Save rankings
        rankings_path = self.output_dir / "metrics" / "rankings.json"
        engine.save_rankings(rankings, str(rankings_path))
        return rankings

    def _save_summary(self, result: ExperimentResult):
        best_model = result.get_best_model()

        # Build best_model section (handle None case)
        best_model_summary = None
        if best_model is not None:
            best_model_summary = {
                "name": best_model.model_name,
                "rank": best_model.rank,
                "composite_score": best_model.composite_score,
                "objectives": {
                    obj.name: {
                        "value": obj.value,
                        "normalized_score": obj.normalized_score,
                        "weighted_score": obj.weighted_score,
                        "achieved": obj.achieved,
                    }
                    for obj in best_model.objective_scores
                },
            }

        summary = {
            "experiment_name": result.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "n_models_trained": len(result.trained_models),
            "n_features_selected": len(result.selected_features),
            "best_model": best_model_summary,
            "output_dir": result.output_dir,
        }

        summary_path = self.output_dir / "experiment_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ExperimentRunner":
        config = ConfigParser.from_yaml(config_path)
        return cls(config=config)
