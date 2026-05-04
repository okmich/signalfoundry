"""Cost-function enum for ruptures-based offline labelling."""

from __future__ import annotations

from enum import StrEnum


class UnivariateCost(StrEnum):
    """Ruptures cost functions compatible with the univariate ``(T, 1)`` input contract.

    ``linear`` (piecewise linear regression on multivariate input) and the multivariate ``rbf`` / ``KernelCPD`` regimes are deliberately excluded:
    they require a future multivariate variant of ``label_with_posteriors`` and a corresponding multivariate BOCPD observation model.

    ``ar`` is also excluded for now: ``rpt.Pelt(model="ar")`` uses ruptures' default AR order, which is a hidden research choice, and ``min_size``
    must be raised to respect that order. AR will be added when an explicit ``ar_order`` parameter and a ``min_size`` interlock are implemented.
    """

    L1 = "l1"
    L2 = "l2"
    NORMAL = "normal"
    RANK = "rank"
    RBF = "rbf"
