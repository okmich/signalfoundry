"""
Meta-Model Module for Signal Filtering.

Predicts whether a primary signal will result in a profitable trade,
using any labeling method as the target (TBM, fixed-horizon returns, regime labels, etc.).

Usage:
------
>>> from okmich_quant_ml.meta_model import (
...     prepare_meta_dataset,
...     MetaModel,
...     MetaModelConfig,
... )
>>>
>>> # Prepare dataset
>>> X, y = prepare_meta_dataset(features, labels, target_col="label")
>>>
>>> # Train
>>> model = MetaModel(MetaModelConfig(threshold=0.6), XGBClassifier())
>>> model.fit(X, y)
>>>
>>> # Inference
>>> if model.should_trade(current_features):
...     prob = model.predict_proba(current_features)
"""

from .meta_model import (
    MetaModelConfig,
    MetaModel,
    prepare_meta_dataset,
)

__all__ = [
    "MetaModelConfig",
    "MetaModel",
    "prepare_meta_dataset",
]
