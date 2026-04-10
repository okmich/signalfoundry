import hashlib
import importlib
import inspect
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional


class FeatureFunctionLogger:
    """
    Logger for feature engineering functions.

    Captures metadata about feature engineering functions to ensure
    experiment reproducibility.

    Examples
    --------
    >>> logger = FeatureFunctionLogger()
    >>> metadata = logger.log_function(
    ...     module='playground.experiment.custom_features',
    ...     function='my_feature_generator',
    ...     params={'lookback': 20},
    ...     version='v1'
    ... )
    >>> logger.save_metadata('experiments/exp_001/feature_metadata.json')
    >>> logger.save_source_code('experiments/exp_001/feature_engineering_code.py')
    """

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.function: Optional[Callable] = None
        self.source_code: Optional[str] = None

    def log_function(
        self,
        module: str,
        function: str,
        params: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        save_source: bool = False,
    ) -> Dict[str, Any]:
        """
        Log feature engineering function metadata.

        Parameters
        ----------
        module : str
            Module path (e.g., 'playground.experiment.custom_features')
        function : str
            Function name (e.g., 'my_feature_generator')
        params : dict, optional
            Parameters passed to the function
        version : str, optional
            Feature version identifier
        save_source : bool, default=False
            Whether to save source code snapshot

        Returns
        -------
        dict
            Metadata dictionary
        """
        # Import function
        try:
            mod = importlib.import_module(module)
            func = getattr(mod, function)
            self.function = func
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot import {module}.{function}: {e}")

        # Capture metadata
        self.metadata = {
            "module": module,
            "function": function,
            "params": params or {},
            "version": version or "unversioned",
            "timestamp": datetime.now().isoformat(),
        }

        # Capture source code if requested
        if save_source:
            try:
                source = inspect.getsource(func)
                self.source_code = source
                self.metadata["code_hash"] = self._hash_code(source)
            except Exception as e:
                print(f"Warning: Could not capture source code: {e}")
                self.source_code = None

        return self.metadata

    def _hash_code(self, code: str) -> str:
        """Generate hash of source code."""
        return hashlib.sha256(code.encode()).hexdigest()[:16]

    def save_metadata(self, filepath: str):
        """Save metadata to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(self.metadata, f, indent=2)

    def save_source_code(self, filepath: str):
        """Save source code snapshot to Python file."""
        if self.source_code is None:
            raise ValueError(
                "No source code captured. Set save_source=True when logging."
            )

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        header = f"""# Feature Engineering Source Code Snapshot
# Captured: {self.metadata.get('timestamp', 'unknown')}
# Module: {self.metadata.get('module', 'unknown')}
# Function: {self.metadata.get('function', 'unknown')}
# Version: {self.metadata.get('version', 'unknown')}
# Hash: {self.metadata.get('code_hash', 'unknown')}

"""

        with open(filepath, "w") as f:
            f.write(header)
            f.write(self.source_code)

    def get_function(self) -> Callable:
        """Get the logged function."""
        if self.function is None:
            raise ValueError("No function logged yet.")
        return self.function


def load_feature_function(metadata_path: str) -> Callable:
    """
    Load feature engineering function from metadata.

    Parameters
    ----------
    metadata_path : str
        Path to feature_metadata.json

    Returns
    -------
    Callable
        The feature engineering function

    Examples
    --------
    >>> func = load_feature_function('experiments/exp_001/feature_metadata.json')
    >>> features = func(df, lookback=20)
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    module = metadata["module"]
    function = metadata["function"]

    try:
        mod = importlib.import_module(module)
        func = getattr(mod, function)
        return func
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Cannot load function {module}.{function}: {e}")


def log_builtin_factory(factory_name: str, version: str) -> Dict[str, Any]:
    module = "okmich_quant_features.feature_factory"

    logger = FeatureFunctionLogger()
    metadata = logger.log_function(
        module=module,
        function=factory_name,
        version=version,
        save_source=False,  # Built-in, no need to snapshot
    )

    return metadata
