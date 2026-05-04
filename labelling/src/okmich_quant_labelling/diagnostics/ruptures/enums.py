"""Cost-function enum for ruptures-based offline labelling."""

from __future__ import annotations

from enum import StrEnum


class UnivariateCost(StrEnum):
    """Ruptures cost functions compatible with the univariate ``(T, 1)`` input contract.

    Supported: ``L1``, ``L2``, ``NORMAL``, ``RANK``, ``RBF``. ``RBF`` is supported on the univariate input shape used here
    (``ruptures.costs.CostRbf`` does work on multivariate inputs, but the labeller deliberately constrains itself to ``(T, 1)``).

    Performance note: ``RBF`` is O(n²) in segment length and is materially slower than ``L1`` / ``L2`` / ``NORMAL`` on long
    series. For series longer than ~10k bars, prefer ``L2`` or ``NORMAL`` unless ``RBF`` is empirically required for your
    changepoint structure.

    Excluded:
    - ``linear`` (piecewise linear regression on multivariate input) and the multivariate ``KernelCPD`` regimes — they require
      a future multivariate variant of ``label_with_posteriors`` and a corresponding multivariate BOCPD observation model.
    - ``ar`` — ``rpt.Pelt(model="ar")`` uses ruptures' default AR order, which is a hidden research choice, and ``min_size``
      must be raised to respect that order. AR will be added when an explicit ``ar_order`` parameter and a ``min_size``
      interlock are implemented.
    """

    L1 = "l1"
    L2 = "l2"
    NORMAL = "normal"
    RANK = "rank"
    RBF = "rbf"
