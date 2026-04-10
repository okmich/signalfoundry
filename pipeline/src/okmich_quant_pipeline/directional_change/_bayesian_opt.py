"""
Bayesian optimisation of IDC threshold parameters (θ, α).

Finds the (theta, alpha) pair that maximises cumulative return of ITA Rules 1–3
on a training price window. The HMM is not fitted during optimisation — each
candidate is evaluated using run_ita_simulation() which applies the raw DC
trading rules without a regime filter, keeping evaluation fast.

The optimal (θ, α) should be re-optimised every training window as no stable
pair exists across months or currency pairs (Hu et al. 2022, Table 2).

Search space defaults (Wu & Han 2023, Section 4.2):
  Tick data  : theta ∈ [0.0003, 0.003], alpha ∈ [0.10, 1.00]
  5-min bars : theta ∈ [0.001,  0.005], alpha ∈ [0.10, 1.00]

References
----------
Wu, Y. & Han, J. (2023). Intelligent Trading Strategy Based on Improved
    Directional Change and Regime Change Detection. arXiv:2309.15383v1.
    Section 3.1 (IDC parameters), Section 4.2 (Bayesian optimisation protocol),
    Algorithm 1 Step 1 (optimisation as first pipeline stage).

Hu, Z., Li, Y. & Wu, Y. (2022). Incorporating Improved Directional Change and
    Regime Change Detection to Formulate Trading Strategies in Foreign Exchange
    Markets. SSRN:4048864.
    Section 2.2.1 (asymmetric threshold), Section 2.3.1 (optimisation motivation),
    Table 2 (empirical evidence that optimal (θ, α) is unstable across months).
"""
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

from okmich_quant_features.directional_change import idc_parse

from ._ita_sim import run_ita_simulation

# Search space presets — Wu & Han (2023) Section 4.2
THETA_BOUNDS_TICK: tuple[float, float] = (0.0003, 0.003)
THETA_BOUNDS_5MIN: tuple[float, float] = (0.001, 0.005)
ALPHA_BOUNDS: tuple[float, float] = (0.10, 1.00)

# Minimum DC events required for a non-degenerate objective evaluation
_MIN_DC_EVENTS = 20


def optimise_idc_params(prices: pd.Series, theta_bounds: tuple[float, float] = THETA_BOUNDS_TICK, alpha_bounds: tuple[float, float] = ALPHA_BOUNDS, n_calls: int = 100, n_initial: int = 10, random_state: int = 42) -> tuple[float, float]:
    """
    Find the (theta, alpha) pair that maximises ITA cumulative return on a training window.

    Uses Gaussian-process Bayesian optimisation (scikit-optimize gp_minimize) with
    Expected Improvement acquisition. The objective evaluates ITA Rules 1–3 without
    an HMM regime filter — all DC confirmation bars are treated as regime S1 (normal).

    Parameters
    ----------
    prices : pd.Series
        Close price series for the training window, in chronological order.
    theta_bounds : tuple[float, float]
        Search bounds for the DC upward threshold θ. Use THETA_BOUNDS_TICK for
        tick/sub-minute data or THETA_BOUNDS_5MIN for 5-minute bars.
    alpha_bounds : tuple[float, float]
        Search bounds for the asymmetric attenuation coefficient α ∈ (0, 1].
    n_calls : int
        Total number of objective evaluations including initial random points.
        Wu & Han (2023) use 100 (default).
    n_initial : int
        Number of initial random evaluations before GP surrogate is used.
    random_state : int
        Seed for reproducibility of random initial points and GP fitting.

    Returns
    -------
    tuple[float, float]
        (optimal_theta, optimal_alpha)

    Notes
    -----
    - Candidates producing fewer than 20 non-NaN RDC values are penalised with
      CRR = 0.0 (degenerate threshold: too few DC events to be tradeable).
    - Acquisition function: Expected Improvement (EI) — same as Wu & Han (2023).
    - Re-optimise every training window: no stable (θ, α) exists across periods
      (Hu et al. 2022, Table 2).

    References
    ----------
    Wu & Han (2023) Section 4.2 / Algorithm 1 Step 1.
    Hu et al. (2022) Section 2.2.1 / Section 2.3.1.
    """
    def _objective(params: list) -> float:
        theta, alpha = float(params[0]), float(params[1])
        idc = idc_parse(prices, theta=theta, alpha=alpha)

        n_dc_events = int(idc["rdc"].notna().sum())
        if n_dc_events < _MIN_DC_EVENTS:
            return 1e6  # penalise degenerate threshold strongly

        result = run_ita_simulation(idc, prices, theta=theta)
        # gp_minimize minimises — negate CRR to maximise it
        return -result["cumulative_return"]

    opt = gp_minimize(
        func=_objective,
        dimensions=[
            Real(theta_bounds[0], theta_bounds[1], name="theta"),
            Real(alpha_bounds[0], alpha_bounds[1], name="alpha"),
        ],
        n_calls=n_calls,
        n_initial_points=n_initial,
        acq_func="EI",
        random_state=random_state,
    )

    return float(opt.x[0]), float(opt.x[1])
