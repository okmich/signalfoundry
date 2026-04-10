"""
RegimeLabelGenerator — sub-package for regime label generation strategies.

Each strategy generates training targets for the regime classification pipeline.
Strategy selection determines the trade-off between label quality, causality, and model complexity.

Strategies
----------
A  HmmDirectStrategy
        Pure HMM.  HMM is fitted on features and IS the deployed model.
        Viterbi states are mapped to yardstick labels for economic validation
        and state mapping.  leaks_future=False.

B  OracleDistillationStrategy
        Oracle labeller generates look-ahead labels over the training fold.
        A supervised classifier is trained on these labels; oracle is NOT
        deployed.  leaks_future=True (acknowledges_lookahead=True required).

B2 HmmViterbiDistillationStrategy
        HMM is fitted on features; Viterbi states are mapped to yardstick
        labels.  A supervised classifier is trained on these labels; HMM is
        NOT deployed.  leaks_future=True (acknowledges_lookahead=True required).

C  CausalStrategy
        CausalRegimeLabeler generates rolling causal labels — no look-ahead,
        no HMM.  A supervised classifier is trained on these labels.
        leaks_future=False.

Usage
-----
>>> from okmich_quant_labelling.regime.label_generator import (
...     RegimeLabelGenerator,
...     HmmDirectStrategy,
...     OracleDistillationStrategy,
...     HmmViterbiDistillationStrategy,
...     CausalStrategy,
...     create_label_generator,
... )

>>> # Strategy C — simplest, fully causal
>>> from okmich_quant_labelling.regime import CausalRegimeLabeler, MarketPropertyType
>>> labeler = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION)
>>> gen = CausalStrategy(labeler)
>>> labels = gen.generate_labels(train_df, price_col="close")

>>> # Strategy A — HMM direct
>>> from okmich_quant_ml.hmm import create_simple_hmm_instance, DistType
>>> hmm = create_simple_hmm_instance(DistType.NORMAL, n_states=3)
>>> gen = HmmDirectStrategy(hmm, yardstick="direction")
>>> labels = gen.generate_labels(train_df, X=feature_array)

>>> # Factory shorthand
>>> gen = create_label_generator("C", labeler=labeler)
>>> gen = create_label_generator("A", hmm=hmm, yardstick="direction")
"""

from ._base import RegimeLabelGenerator
from .strategy_a import HmmDirectStrategy
from .strategy_b import OracleDistillationStrategy
from .strategy_b2 import HmmViterbiDistillationStrategy
from .strategy_c import CausalStrategy

__all__ = [
    "RegimeLabelGenerator",
    "HmmDirectStrategy",
    "OracleDistillationStrategy",
    "HmmViterbiDistillationStrategy",
    "CausalStrategy",
    "create_label_generator",
]


def create_label_generator(strategy: str, **kwargs) -> RegimeLabelGenerator:
    """
    Factory function for RegimeLabelGenerator instances.

    Parameters
    ----------
    strategy : {'A', 'B', 'B2', 'C'}
        Which labelling strategy to instantiate.
    **kwargs
        Passed directly to the strategy constructor.  See each class for
        required and optional parameters.

    Returns
    -------
    RegimeLabelGenerator

    Examples
    --------
    Strategy A::

        from okmich_quant_ml.hmm import create_simple_hmm_instance, DistType
        hmm = create_simple_hmm_instance(DistType.NORMAL, n_states=3)
        gen = create_label_generator("A", hmm=hmm, yardstick="direction")

    Strategy B::

        from okmich_quant_labelling.diagnostics.regime import AmplitudeBasedLabeler
        oracle = AmplitudeBasedLabeler(minamp=50)
        gen = create_label_generator(
            "B", oracle=oracle, acknowledges_lookahead=True
        )

    Strategy B2::

        hmm = create_simple_hmm_instance(DistType.NORMAL, n_states=3)
        gen = create_label_generator(
            "B2",
            hmm=hmm,
            yardstick="direction",
            acknowledges_lookahead=True,
        )

    Strategy C::

        from okmich_quant_labelling.regime import CausalRegimeLabeler, MarketPropertyType
        labeler = CausalRegimeLabeler(yardstick=MarketPropertyType.DIRECTION)
        gen = create_label_generator("C", labeler=labeler)
    """
    _STRATEGIES = {
        "A": HmmDirectStrategy,
        "B": OracleDistillationStrategy,
        "B2": HmmViterbiDistillationStrategy,
        "C": CausalStrategy,
    }

    key = strategy.upper()
    if key not in _STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. "
            f"Choose from: {list(_STRATEGIES)}"
        )
    return _STRATEGIES[key](**kwargs)
