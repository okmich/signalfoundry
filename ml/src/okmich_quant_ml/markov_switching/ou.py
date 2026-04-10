"""
Markov-Switching Ornstein-Uhlenbeck (MS-OU) model for regime-dependent mean reversion.

Mathematical foundation
-----------------------
Continuous-time OU process per regime s:

    dX_t = κ^(s) (θ^(s) − X_t) dt + σ^(s) dW_t

Exact discretisation with time step Δt:

    X_{t+Δt} = θ^(s) + φ^(s) (X_t − θ^(s)) + η_t
    η_t ~ N(0, σ²_disc^(s))

where:
    φ^(s)        = e^{−κ^(s) Δt}                            (AR(1) coefficient)
    c^(s)        = θ^(s) (1 − φ^(s))                        (AR(1) intercept)
    σ²_disc^(s)  = σ²^(s) / (2κ^(s)) · (1 − φ^(s)²)

Inverse reparametrisation (discrete → continuous):
    κ^(s)        = −ln(φ^(s)) / Δt
    θ^(s)        = c^(s) / (1 − φ^(s))
    σ²^(s)       = σ²_disc^(s) · 2κ^(s) / (1 − φ^(s)²)

Mean-reversion requires  0 < φ^(s) < 1  (i.e. κ > 0).
A fitted regime with φ ≤ 0 or φ ≥ 1 is *not* interpretable as OU and a
warning is issued; the regime is flagged as non-mean-reverting.

Half-life:  t_{1/2} = ln(2) / κ = −Δt · ln(2) / ln(φ)
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .ar import MarkovSwitchingAR


class MarkovSwitchingOU(MarkovSwitchingAR):
    """
    Markov-Switching Ornstein-Uhlenbeck model.

    A specialisation of ``MarkovSwitchingAR`` with ``order=1`` that
    reparametrises the fitted AR(1) coefficients into the continuous-time OU
    parameters κ (speed), θ (long-run mean), and σ (diffusion).

    Parameters
    ----------
    n_regimes : int, default=2
        Number of regimes.
    dt : float, default=1.0
        Observation time step.  Use ``1/252`` for daily data when κ and σ
        should be expressed in annualised units, or ``1`` to keep units in
        bar-multiples.
    switching_variance : bool, default=True
        Regime-dependent diffusion σ^(s).
    random_state : int, default=42

    Attributes
    ----------
    ou_parameters_ : pd.DataFrame  (set after fit)
        Per-regime OU parameters: kappa, theta, sigma, half_life,
        is_mean_reverting.

    Examples
    --------
    >>> ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1/252)
    >>> ms_ou.fit(spread)
    >>> params = ms_ou.get_ou_parameters()
    >>> print(params[['kappa', 'theta', 'half_life']])
    """

    def __init__(self, n_regimes: int = 2, dt: float = 1.0, switching_variance: bool = True, random_state: int = 42):
        if dt <= 0:
            raise ValueError("dt must be positive")
        super().__init__(
            n_regimes=n_regimes,
            order=1,  # OU is always AR(1)
            switching_variance=switching_variance,
            random_state=random_state,
        )
        self.dt = dt
        self.ou_parameters_: pd.DataFrame | None = None

    # order is fixed — prevent accidental override
    @property  # type: ignore[override]
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        if value != 1:
            raise ValueError("MarkovSwitchingOU requires order=1 (OU is an AR(1) process)")
        self._order = value

    # ─── Fitting ────────────────────────────────────────────────────────────────

    def fit(self, y: np.ndarray, exog: np.ndarray | None = None,
            num_restarts: int = 3) -> MarkovSwitchingOU:
        super().fit(y, exog=exog, num_restarts=num_restarts)
        self.ou_parameters_ = self._build_ou_parameters()
        self._warn_non_mean_reverting()
        return self

    def train(self, y: np.ndarray, n_regimes_range: tuple[int, int] | None = None,
              criterion: str = "bic", exog: np.ndarray | None = None,
              **_ignored) -> MarkovSwitchingOU:
        """
        Fit with automatic n_regimes selection.  order is fixed at 1.

        Parameters
        ----------
        y : np.ndarray
        n_regimes_range : (min, max), default=(2, 3)
        criterion : 'aic' | 'bic', default='bic'
        exog : np.ndarray, optional
        """
        n_regimes_range = n_regimes_range or (2, 3)
        # delegate to parent with order fixed at 1
        super().train(y, order_range=(1, 1), n_regimes_range=n_regimes_range,
                      criterion=criterion, exog=exog)
        self.ou_parameters_ = self._build_ou_parameters()
        self._warn_non_mean_reverting()
        return self

    # ─── OU reparametrisation ───────────────────────────────────────────────────

    def _reparametrize_to_ou(self, regime: int) -> dict[str, float]:
        """Convert discrete AR(1) parameters to continuous OU parameters.

        Returns
        -------
        dict with keys: phi, kappa, theta, sigma2_disc, sigma2_cont, half_life,
            is_mean_reverting
        """
        ar_coeffs, intercept, sigma2_disc = self._get_regime_params(regime)
        phi = float(ar_coeffs[0])

        mr = 0.0 < phi < 1.0

        if mr:
            kappa = -np.log(phi) / self.dt
            theta = intercept / (1.0 - phi)
            # σ²_disc = σ²_cont / (2κ) · (1 − φ²)  →  σ²_cont = σ²_disc · 2κ / (1 − φ²)
            sigma2_cont = sigma2_disc * 2.0 * kappa / (1.0 - phi ** 2)
            half_life = np.log(2.0) / kappa
        else:
            # Non-mean-reverting regime: return NaN for OU-specific params
            kappa = np.nan
            theta = intercept / (1.0 - phi) if phi != 1.0 else np.nan
            sigma2_cont = np.nan
            half_life = np.nan

        return {
            "phi": phi,
            "kappa": kappa,
            "theta": theta,
            "sigma2_disc": sigma2_disc,
            "sigma2_cont": sigma2_cont,
            "half_life": half_life,
            "is_mean_reverting": mr,
        }

    def _build_ou_parameters(self) -> pd.DataFrame:
        rows = []
        for r in range(self.n_regimes):
            p = self._reparametrize_to_ou(r)
            rows.append({
                "regime": r,
                "phi": p["phi"],
                "kappa": p["kappa"],
                "theta": p["theta"],
                "sigma": np.sqrt(p["sigma2_cont"]) if not np.isnan(p["sigma2_cont"]) else np.nan,
                "sigma_disc": np.sqrt(p["sigma2_disc"]),
                "half_life": p["half_life"],
                "is_mean_reverting": p["is_mean_reverting"],
            })
        return pd.DataFrame(rows)

    def _warn_non_mean_reverting(self) -> None:
        for _, row in self.ou_parameters_.iterrows():
            if not row["is_mean_reverting"]:
                warnings.warn(
                    f"Regime {int(row['regime'])} has φ={row['phi']:.4f}, which is outside (0, 1). "
                    "This regime cannot be interpreted as an OU process. "
                    "Consider refitting or using MarkovSwitchingAR instead.",
                    UserWarning,
                    stacklevel=3,
                )

    # ─── Public API ─────────────────────────────────────────────────────────────

    def get_ou_parameters(self) -> pd.DataFrame:
        """
        Return per-regime OU parameters.

        Returns
        -------
        pd.DataFrame with columns:
            regime, phi, kappa, theta, sigma, sigma_disc, half_life, is_mean_reverting

        Notes
        -----
        ``kappa``, ``theta``, ``sigma``, and ``half_life`` are NaN for regimes
        where φ ∉ (0, 1) (i.e. non-mean-reverting regimes).
        """
        self._validate_fitted()
        return self.ou_parameters_.copy()

    def is_mean_reverting(self, regime: int) -> bool:
        """Return True if ``regime`` satisfies 0 < φ < 1 (κ > 0)."""
        self._validate_fitted()
        row = self.ou_parameters_.loc[self.ou_parameters_["regime"] == regime]
        if row.empty:
            raise ValueError(f"Regime {regime} not found")
        return bool(row["is_mean_reverting"].iloc[0])

    def interpret_regimes(self) -> dict[int, str]:
        """
        Regime interpretation using OU parameters.

        Returns a label based on mean-reversion speed and diffusion level:
            e.g. {0: 'Fast MR (low_vol)', 1: 'Slow MR (high_vol)'}

        Non-mean-reverting regimes (φ ≥ 1 or φ ≤ 0) are labelled accordingly.
        """
        self._validate_fitted()
        params = self.ou_parameters_
        mr_params = params[params["is_mean_reverting"]]

        median_kappa = mr_params["kappa"].median() if not mr_params.empty else np.nan
        median_sigma = params["sigma_disc"].median()

        interpretations = {}
        for _, row in params.iterrows():
            r = int(row["regime"])
            vol = "low_vol" if row["sigma_disc"] <= median_sigma else "high_vol"

            if not row["is_mean_reverting"]:
                phi = row["phi"]
                if phi >= 1.0:
                    label = f"Non-stationary (φ={phi:.3f})"
                else:
                    label = f"Oscillating (φ={phi:.3f})"
                interpretations[r] = f"{label} ({vol})"
            else:
                speed = "Fast MR" if row["kappa"] >= median_kappa else "Slow MR"
                hl = row["half_life"]
                interpretations[r] = f"{speed} t½={hl:.1f} ({vol})"

        return interpretations

    # ─── Plotting ───────────────────────────────────────────────────────────────

    def plot_mean_reversion(self, figsize: tuple[int, int] = (12, 5)) -> Figure:
        """
        Plot the mean-reversion profile for each regime.

        Shows the expected path back to θ from a 2σ deviation, with 68% and
        95% forecast bands, over a horizon of 3× the longest half-life.
        """
        self._validate_fitted()

        params = self.ou_parameters_
        mr_params = params[params["is_mean_reverting"] & params["half_life"].notna()]

        if mr_params.empty:
            raise RuntimeError("No mean-reverting regimes to plot.")

        max_hl = mr_params["half_life"].max()
        steps = max(int(3 * max_hl / self.dt), 50)
        t = np.arange(steps) * self.dt

        colors = plt.cm.Set2(np.linspace(0, 1, self.n_regimes))
        fig, ax = plt.subplots(figsize=figsize)

        for _, row in mr_params.iterrows():
            r = int(row["regime"])
            kappa = row["kappa"]
            theta = row["theta"]
            sigma_cont = row["sigma"]

            # start 2σ_disc above long-run mean
            x0_deviation = 2.0 * row["sigma_disc"]

            # E[X_{t} | X_0 = θ + x0_deviation] = θ + x0_deviation · e^{−κt}
            mean_path = theta + x0_deviation * np.exp(-kappa * t)

            # Var[X_t | X_0] = σ²/(2κ) · (1 − e^{−2κt})
            var_path = (sigma_cont ** 2) / (2 * kappa) * (1 - np.exp(-2 * kappa * t))
            std_path = np.sqrt(var_path)

            label = self.interpret_regimes()[r]
            ax.plot(t, mean_path - theta, color=colors[r], linewidth=2, label=f"R{r}: {label}")
            ax.fill_between(t, (mean_path - std_path) - theta, (mean_path + std_path) - theta,
                            alpha=0.25, color=colors[r])
            ax.fill_between(t, (mean_path - 2 * std_path) - theta, (mean_path + 2 * std_path) - theta,
                            alpha=0.12, color=colors[r])

            # half-life marker
            hl = row["half_life"]
            ax.axvline(x=hl, color=colors[r], linestyle=":", linewidth=1, alpha=0.6)

        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", label="Long-run mean θ")
        ax.set_xlabel(f"Time ({self.dt:.4g} units per bar)", fontsize=10)
        ax.set_ylabel("Deviation from θ", fontsize=10)
        ax.set_title("Mean-Reversion Profiles by Regime\n(starting from +2σ deviation)", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig