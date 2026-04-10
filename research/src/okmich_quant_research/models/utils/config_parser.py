import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class ConfigParser:
    """
    Parse and validate experiment configuration from YAML.

    Examples
    --------
    >>> parser = ConfigParser()
    >>> config = parser.load('models_research_configs/research/my_exp.yaml')
    >>> config.get_experiment_name()
    'hmm_us500_v1'
    >>> config.get_objectives()
    [{'name': 'regime_discriminability', 'weight': 0.35, ...}, ...]
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
    ):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}

        if config_dict:
            self.config = config_dict
            self._normalize()
            self._validate()
        elif config_path:
            self.load(config_path)

    @classmethod
    def from_yaml(cls, config_path: str) -> "ConfigParser":
        return cls(config_path=config_path)

    def load(self, config_path: str) -> "ConfigParser":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.config_path = str(config_path)
        self._normalize()
        self._validate()

        return self

    def _normalize(self):
        """Promote model.walk_forward to the canonical top-level walk_forward key.

        Walk-forward params must be set under ``model.walk_forward`` in YAML (where
        validation also reads them).  Runtime getters and CLI overrides use the
        top-level ``walk_forward`` key.  This method merges both so that neither
        path is silently ignored.
        """
        model_wf = self.config.get("model", {}).get("walk_forward", {})
        top_wf = self.config.get("walk_forward", {})

        if not model_wf and not top_wf:
            return

        if model_wf and top_wf:
            # Both present — check for conflicts, warn if they differ
            conflicts = {
                k for k in set(model_wf) & set(top_wf) if model_wf[k] != top_wf[k]
            }
            if conflicts:
                warnings.warn(
                    f"Both 'model.walk_forward' and top-level 'walk_forward' are "
                    f"present with conflicting values for key(s) {sorted(conflicts)}. "
                    "Top-level values will take precedence (CLI overrides). "
                    "Remove one of them from the config to suppress this warning.",
                    stacklevel=4,
                )
            # Merge: model_wf is the base, top_wf overrides (CLI wins)
            self.config["walk_forward"] = {**model_wf, **top_wf}
        elif model_wf:
            # Only model.walk_forward present — promote it to top-level
            self.config["walk_forward"] = dict(model_wf)

    def _validate(self):
        """Validate configuration structure."""
        required_sections = ["experiment_name", "data", "model", "objectives"]

        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")

        # Additional validation per model type
        model_type = self.config["model"].get("type", "")
        _supervised_types = {
            "supervised_classification", "supervised_regression", "regime_classification"
        }

        if model_type in ("supervised_classification", "supervised_regression"):
            # Target engineering is required for supervised learning
            if "target_engineering" not in self.config:
                raise ValueError(
                    f"Supervised model type '{model_type}' requires 'target_engineering' section"
                )

        # Walk-forward params are required for all supervised/regime types.
        # Check the canonical top-level walk_forward (populated by _normalize()).
        if model_type in _supervised_types:
            wf = self.config.get("walk_forward", {})
            missing_wf = [k for k in ("train_period", "test_period") if k not in wf]
            if missing_wf:
                raise ValueError(
                    f"walk_forward is missing required keys for type '{model_type}': "
                    f"{missing_wf}. Set them under 'walk_forward:' or 'model.walk_forward:'."
                )
            for k in ("train_period", "test_period", "step_period"):
                if k in wf and not isinstance(wf[k], int):
                    raise ValueError(
                        f"walk_forward.{k} must be an integer (got {type(wf[k]).__name__})"
                    )

        if model_type == "regime_classification":
            if "labelling" not in self.config.get("model", {}):
                raise ValueError(
                    "model type 'regime_classification' requires 'model.labelling' section"
                )
            strategy = self.config["model"]["labelling"].get("strategy", "C").upper()
            _supported_strategies = {"A", "B2", "C"}
            if strategy not in _supported_strategies:
                raise ValueError(
                    f"Unsupported labelling strategy '{strategy}'. "
                    f"Supported via YAML config: {sorted(_supported_strategies)}. "
                    "(Strategy B requires a custom oracle object passed in code.)"
                )

    # Experiment metadata
    def get_experiment_name(self) -> str:
        """Get experiment name."""
        return self.config["experiment_name"]

    def get_research_type(self) -> str:
        """Get research type (supervised/unsupervised).

        If 'research_type' is absent from the config, infer it from the model
        type to prevent clustering/HMM experiments being silently routed through
        the supervised evaluation branch.
        """
        if "research_type" in self.config:
            return self.config["research_type"]
        _supervised_model_types = {
            "supervised_classification",
            "supervised_regression",
            "regime_classification",
        }
        model_type = self.config.get("model", {}).get("type", "")
        if model_type and model_type not in _supervised_model_types:
            return "unsupervised"
        return "supervised"

    # Data configuration
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config["data"]

    def get_symbol(self) -> str:
        """Get symbol."""
        return self.config["data"]["symbol"]

    def get_timeframe(self) -> str:
        """Get timeframe."""
        return self.config["data"]["timeframe"]

    def get_max_samples(self) -> Optional[int]:
        """Get max samples."""
        return self.config["data"].get("max_samples")

    # Feature engineering configuration
    def get_feature_engineering_config(self) -> Dict[str, Any]:
        """Get feature engineering configuration."""
        return self.config.get("feature_engineering", {})

    def get_feature_version(self) -> str:
        """Get feature version."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("version", "default")

    def get_external_function_config(self) -> Optional[Dict[str, Any]]:
        """Get external feature function configuration."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("external_function")

    def get_feature_factory(self) -> Optional[str]:
        """Get built-in feature factory."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("factory")

    def should_save_source_code(self) -> bool:
        """Whether to save source code snapshot."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("save_source_code", False)

    # Auto feature selection
    def get_auto_selection_config(self) -> Dict[str, Any]:
        """Get auto feature selection configuration."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("auto_selection", {})

    def get_auto_selection_enabled(self) -> bool:
        """Alias for is_auto_selection_enabled."""
        return self.is_auto_selection_enabled()

    def get_auto_selection_top_n(self) -> int:
        """Alias for get_top_n_features."""
        return self.get_top_n_features()

    def get_auto_selection_vif_threshold(self) -> float:
        """Alias for get_vif_threshold."""
        return self.get_vif_threshold()

    def get_auto_selection_min_importance(self) -> float:
        """Get minimum feature importance threshold."""
        return self.get_auto_selection_config().get("min_importance", 0.01)

    def is_auto_selection_enabled(self) -> bool:
        """Check if auto feature selection is enabled."""
        return self.get_auto_selection_config().get("enabled", False)

    def get_top_n_features(self) -> int:
        """Get number of top features to select."""
        return self.get_auto_selection_config().get("top_n", 15)

    def get_vif_threshold(self) -> float:
        """Get VIF threshold."""
        return self.get_auto_selection_config().get("vif_threshold", 10.0)

    # EDA configuration
    def get_eda_config(self) -> Dict[str, Any]:
        """Get EDA configuration."""
        fe_config = self.get_feature_engineering_config()
        return fe_config.get("eda", {})

    def is_eda_enabled(self) -> bool:
        """Check if EDA is enabled."""
        return self.get_eda_config().get("enabled", True)

    def get_eda_n_top_features(self) -> int:
        """Get number of top features for EDA."""
        return self.get_eda_config().get("n_top_features", 20)

    def should_save_eda_plots(self) -> bool:
        """Whether to save EDA plots."""
        return self.get_eda_config().get("save_plots", True)

    def get_eda_enabled(self) -> bool:
        """Alias for is_eda_enabled."""
        return self.is_eda_enabled()

    def get_eda_save_plots(self) -> bool:
        """Alias for should_save_eda_plots."""
        return self.should_save_eda_plots()

    # Model configuration
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config["model"]

    def get_model_type(self) -> str:
        """Get model type (clustering, etc.)."""
        return self.config["model"]["type"]

    def get_n_states_range(self) -> List[int]:
        """Get n_states range for models."""
        model_config = self.get_model_config()
        return model_config.get("n_states_range", [2, 3, 4])

    def get_model_n_states_range(self) -> List[int]:
        """Alias for get_n_states_range."""
        return self.get_n_states_range()

    def get_model_algorithms(self) -> List[str]:
        """Get clustering algorithms config."""
        return self.get_model_config().get("algorithms", [])

    # Target engineering configuration (for supervised learning)
    def get_target_engineering_config(self) -> Dict[str, Any]:
        """Get target engineering configuration."""
        return self.config.get("target_engineering", {})

    def get_target_external_function_config(self) -> Optional[Dict[str, Any]]:
        """Get external target function configuration."""
        te_config = self.get_target_engineering_config()
        return te_config.get("external_function")

    def get_target_column(self) -> str:
        """Get target column name after generation."""
        return self.get_target_engineering_config().get("target_column", "target")

    def get_task_type(self) -> str:
        """Get task type: 'classification' or 'regression'."""
        return self.get_target_engineering_config().get("task_type", "classification")

    # Walk-forward configuration (for supervised learning)
    def get_walk_forward_config(self) -> Dict[str, Any]:
        """Get walk-forward validation configuration."""
        return self.config.get("walk_forward", {})

    def get_train_period(self) -> int:
        """Get training period in samples."""
        return self.get_walk_forward_config().get("train_period", 5000)

    def get_test_period(self) -> int:
        """Get test period in samples."""
        return self.get_walk_forward_config().get("test_period", 1000)

    def get_step_period(self) -> int:
        """Get step size between windows."""
        return self.get_walk_forward_config().get("step_period", 500)

    def is_anchored_walk_forward(self) -> bool:
        """Whether to use anchored (expanding) windows."""
        return self.get_walk_forward_config().get("anchored", False)

    def get_max_train_period(self) -> Optional[int]:
        """Cap on training window for capped-expanding WFA. None = uncapped."""
        return self.get_walk_forward_config().get("max_train_period", None)

    def get_embargo_bars(self) -> int:
        """Bars to skip between train end and test start (label leakage guard)."""
        return self.get_walk_forward_config().get("embargo_bars", 0)

    # Regime classification labelling configuration
    def get_labelling_config(self) -> Dict[str, Any]:
        """Get labelling configuration (regime_classification only)."""
        return self.get_model_config().get("labelling", {})

    def get_labelling_strategy(self) -> str:
        """Get labelling strategy. Supported via YAML: A, B2, C.
        (Strategy B requires a custom oracle object passed in code.)"""
        return self.get_labelling_config().get("strategy", "C")

    def get_labelling_yardstick(self) -> str:
        """Get labelling yardstick: DIRECTION, MOMENTUM, VOLATILITY, PATH_STRUCTURE."""
        return self.get_labelling_config().get("yardstick", "DIRECTION")

    def get_labelling_acknowledges_lookahead(self) -> bool:
        """Whether caller acknowledges look-ahead (required True for strategies B/B2)."""
        return self.get_labelling_config().get("acknowledges_lookahead", False)

    def get_labelling_price_col(self) -> str:
        """Price column to pass to label generators."""
        return self.get_labelling_config().get("price_col", "close")

    def get_labelling_return_col(self) -> str:
        """Return column to pass to label generators."""
        return self.get_labelling_config().get("return_col", "return")

    def get_labelling_hmm_config(self) -> Dict[str, Any]:
        """HMM config for strategies A/B2 (algorithm, n_states, distribution)."""
        return self.get_labelling_config().get("hmm", {})

    def get_labelling_causal_config(self) -> Dict[str, Any]:
        """CausalRegimeLabeler config for strategy C."""
        return self.get_labelling_config().get("causal_labeler", {})

    def get_regime_classifier_configs(self) -> List[Dict[str, Any]]:
        """sklearn classifier configs for strategies B/B2/C."""
        return self.get_labelling_config().get("classifier_configs", [])

    # Supervised model configuration
    def get_supervised_model_config(self) -> Dict[str, Any]:
        """Get supervised model-specific config."""
        return self.get_model_config().get("supervised", {})

    def get_model_framework(self) -> str:
        """Get model framework: 'sklearn' or 'keras'."""
        return self.get_supervised_model_config().get("framework", "sklearn")

    def get_sklearn_models(self) -> List[Dict[str, Any]]:
        """Get sklearn model configurations."""
        return self.get_supervised_model_config().get("sklearn_models", [])

    def get_keras_model_builder(self) -> Optional[Dict[str, Any]]:
        """Get keras model builder configuration."""
        return self.get_supervised_model_config().get("keras_model_builder")

    def get_tuner_params(self) -> Dict[str, Any]:
        """Get Keras Tuner parameters."""
        return self.get_supervised_model_config().get(
            "tuner_params",
            {"max_trials": 5, "objective": "val_accuracy"},
        )

    def get_keras_training_params(self) -> Dict[str, Any]:
        """Get Keras training parameters (epochs, batch_size, etc.)."""
        return self.get_supervised_model_config().get(
            "training_params",
            {"epochs": 50, "batch_size": 32, "validation_split": 0.2},
        )

    def get_sequence_length(self) -> Optional[int]:
        """Get sequence length for RNN models (None for non-sequential)."""
        return self.get_supervised_model_config().get("sequence_length")

    # Supervised evaluation metrics
    def get_supervised_metrics(self) -> List[str]:
        """Get supervised evaluation metrics."""
        task_type = self.get_task_type()
        defaults = {
            "classification": ["accuracy", "precision", "recall", "f1_score", "auc_roc"],
            "regression": ["mse", "rmse", "mae", "r2", "mape"],
        }
        eval_config = self.get_evaluation_config()
        return eval_config.get("metrics", defaults.get(task_type, []))

    def should_save_predictions(self) -> bool:
        """Whether to save model predictions."""
        return self.get_output_config().get("save_predictions", True)

    # Objectives configuration
    def get_objectives(self) -> List[Dict[str, Any]]:
        """Get objectives configuration."""
        objectives = self.config.get("objectives", [])

        # Handle both formats: list of dicts or dict with 'primary'/'secondary'
        if isinstance(objectives, dict):
            primary = objectives.get("primary", [])
            # Flatten if needed
            if isinstance(primary, list):
                return primary
            return [primary]

        return objectives

    def get_primary_objectives(self) -> List[Dict[str, Any]]:
        """Get primary objectives."""
        objectives = self.config.get("objectives", [])

        if isinstance(objectives, dict):
            return objectives.get("primary", [])

        # If list format, all are primary
        return objectives

    def get_secondary_objectives(self) -> List[str]:
        """Get secondary objectives (informative only)."""
        objectives = self.config.get("objectives", [])

        if isinstance(objectives, dict):
            return objectives.get("secondary", [])

        return []

    # Evaluation configuration
    def get_evaluation_config(self) -> Dict[str, Any]:
        """Get evaluation configuration."""
        return self.config.get("evaluation", {})

    def get_label_eval_functions(self) -> List[str]:
        """Get label evaluation functions to run."""
        eval_config = self.get_evaluation_config()

        # Default functions
        defaults = [
            "path_structure_label_classification",
        ]

        # Check if specified in config
        funcs_config = eval_config.get("label_eval_functions", {})
        if isinstance(funcs_config, dict):
            enabled_funcs = []
            for func_name, func_config in funcs_config.items():
                if isinstance(func_config, dict) and func_config.get("enabled", True):
                    enabled_funcs.append(func_name)
            return enabled_funcs if enabled_funcs else defaults

        return defaults

    def is_unsupervised_label_mapping_enabled(self) -> bool:
        """Check if unsupervised label mapping is enabled."""
        eval_config = self.get_evaluation_config()
        mapping_config = eval_config.get("unsupervised_label_mapping", {})
        return mapping_config.get("enabled", False)

    def get_unsupervised_mapping_params(self) -> Dict[str, Any]:
        """Get unsupervised label mapping parameters."""
        eval_config = self.get_evaluation_config()
        mapping_config = eval_config.get("unsupervised_label_mapping", {})
        return mapping_config.get("params", {})

    def get_evaluation_unsupervised_label_mapping(self) -> Optional[Dict[str, Any]]:
        """Get unsupervised label mapping config."""
        eval_config = self.get_evaluation_config()
        return eval_config.get("unsupervised_label_mapping")

    def get_objectives_primary(self) -> List[Dict[str, Any]]:
        """Alias for get_primary_objectives."""
        return self.get_primary_objectives()

    # Output configuration
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get("output", {})

    def get_output_folder(self) -> Optional[str]:
        """Get output folder path."""
        return self.get_output_config().get("folder")

    def should_save_models(self) -> bool:
        """Whether to save models."""
        return self.get_output_config().get("save_models", True)

    def should_save_labels(self) -> bool:
        """Whether to save labels."""
        return self.get_output_config().get("save_labels", True)

    def should_save_plots(self) -> bool:
        """Whether to save plots."""
        return self.get_output_config().get("save_plots", True)

    def should_generate_report(self) -> bool:
        """Whether to generate HTML report."""
        return self.get_output_config().get("generate_report", True)

    def get_output_save_models(self) -> bool:
        """Alias for should_save_models."""
        return self.should_save_models()

    def get_output_save_labels(self) -> bool:
        """Alias for should_save_labels."""
        return self.should_save_labels()

    # Constraints
    def get_constraints(self) -> Dict[str, Any]:
        """Get constraints configuration."""
        return self.config.get("constraints", {})

    def get_min_observations_per_regime(self) -> int:
        """Get minimum observations per regime."""
        return self.get_constraints().get("min_observations_per_regime", 100)

    # Full config access
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self.config.copy()

    def save(self, output_path: str):
        """Save configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
