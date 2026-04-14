from __future__ import annotations

from scipy.stats import norm

import matplotlib.pyplot as plt
import numpy as np
import torch
from pomegranate.distributions import Exponential, Gamma, LogNormal, Normal, StudentT
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM
from typing import Optional, Sequence

from .base_pomegranate import BasePomegranateHMM
from .distribution import Lambda
from .util import DistType, InferenceMode


class PomegranateMixtureHMM(BasePomegranateHMM):
    """
    Pomegranate DenseHMM with mixture model emissions for regime detection.

    Each hidden state emits from a mixture of the same distribution type (e.g., StudentT, Normal, LogNormal), providing multi-modal behavior
    within each regime while controlling parameter complexity.

    Parameters
    ----------
    distribution_type : DistType
        The base distribution type for mixture components
    n_states : int
        Number of hidden regimes
    n_components : int
        Number of mixture components per state
    random_state : int
        Random seed for reproducibility
    max_iter : int
        Maximum EM iterations
    inference_mode : InferenceMode, optional
        Inference mode (FILTERING, SMOOTHING, VITERBI)
    **dist_kwargs
        Additional distribution-specific parameters (e.g., dofs for StudentT)
    """

    def __init__(self, distribution_type: DistType, n_states: int = 2, n_components: int = 2, *, random_state: int = 100,
                 max_iter: int = 100, inference_mode: Optional[InferenceMode] = None, **dist_kwargs):
        super().__init__(distribution_type, n_states, random_state=random_state, max_iter=max_iter,
                         inference_mode=inference_mode, **dist_kwargs)
        self.n_components = n_components

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, n_states_range: Sequence[int] = (2, 3, 4, 5), n_criteria_range: Sequence[int] = (2, 3, 4, 5), criterion: str = "aic") -> "PomegranateMixtureHMM":
        """Pick the best configuration using AIC or BIC."""
        _VALID_CRITERIA = {"aic", "bic"}
        if criterion not in _VALID_CRITERIA:
            raise ValueError(f"Unknown criterion '{criterion}'. Allowed values: {sorted(_VALID_CRITERIA)}")

        best_score, best_inst = np.inf, None
        for j in n_criteria_range:
            for k in n_states_range:
                for inst_kwargs in self._iter_train_dist_kwargs():
                    inst = self.__class__(self.distribution_type, n_states=k, n_components=j,
                                          random_state=self.random_state, max_iter=self.max_iter,
                                          inference_mode=self.inference_mode, **inst_kwargs)
                    inst.fit(X, lengths)
                    aic, bic = inst.get_aic_bic(X)
                    score = aic if criterion == "aic" else bic
                    if score < best_score:
                        best_score, best_inst = score, inst
        return best_inst

    def get_aic_bic(self, X: np.ndarray) -> tuple[float, float]:
        """Calculate AIC and BIC scores for the fitted model."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = X.reshape(-1, 1) if X.ndim == 1 else X

        ll = float(self.log_likelihood(X))

        # Emission parameter counting for mixture model
        n_features = X.shape[1]
        cov_params = self._cov_param_count(n_features)
        params_per_state = (
            self.n_components * n_features  # means
            + self.n_components * cov_params  # covariances (structure-aware)
            + (self.n_components - 1)  # mixture weights
        )
        if self.distribution_type == DistType.STUDENTT:
            params_per_state += self.n_components  # dofs per component

        n_emission_params = self.n_states * params_per_state
        n_trans_params = self.n_states * (self.n_states - 1)
        n_params = (self.n_states - 1) + n_trans_params + n_emission_params

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(X.shape[0])

        return float(aic), float(bic)

    # ------------------------------------------------------------------
    # Interpretation & Visualization Methods
    # ------------------------------------------------------------------
    def get_mixture_parameters(self) -> list[dict]:
        """
        Extract mixture parameters (weights and component parameters) for each state.

        Returns
        -------
        list[dict]
            List of dictionaries (one per state) containing:
            - 'weights': Mixture component weights
            - 'components': List of component parameters (means, covs, etc.)
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        results = []
        for state_idx, mixture_dist in enumerate(self._model.distributions):
            state_params = {"state": state_idx, "weights": None, "components": []}

            # Extract mixture weights
            if hasattr(mixture_dist, "priors"):
                weights = mixture_dist.priors
                if hasattr(weights, "detach"):
                    weights = weights.detach().cpu().numpy()
                state_params["weights"] = weights

            # Extract component parameters
            if hasattr(mixture_dist, "distributions"):
                for comp_idx, comp_dist in enumerate(mixture_dist.distributions):
                    comp_params = {"component": comp_idx}

                    # Extract based on distribution type
                    if hasattr(comp_dist, "means"):
                        comp_params["mean"] = comp_dist.means.detach().cpu().numpy()
                    if hasattr(comp_dist, "covs"):
                        comp_params["cov"] = comp_dist.covs.detach().cpu().numpy()
                    if hasattr(comp_dist, "dofs"):
                        comp_params["dof"] = comp_dist.dofs.detach().cpu().numpy()
                    if hasattr(comp_dist, "mu"):
                        comp_params["mu"] = comp_dist.mu.detach().cpu().numpy()
                    if hasattr(comp_dist, "sigma"):
                        comp_params["sigma"] = comp_dist.sigma.detach().cpu().numpy()

                    state_params["components"].append(comp_params)

            results.append(state_params)

        return results

    def plot_component_weights(self, ax=None, figsize=(10, 6)):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        params = self.get_mixture_parameters()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Prepare data for grouped bar chart
        x = np.arange(self.n_states)
        width = 0.8 / self.n_components

        for comp_idx in range(self.n_components):
            weights = [
                p["weights"][comp_idx] if p["weights"] is not None else 0
                for p in params
            ]
            offset = (comp_idx - self.n_components / 2 + 0.5) * width
            ax.bar(x + offset, weights, width, label=f"Component {comp_idx}")

        ax.set_xlabel("State")
        ax.set_ylabel("Mixture Weight")
        ax.set_title(
            f"Mixture Component Weights per State ({self.distribution_type.name})"
        )
        ax.set_xticks(x)
        ax.set_xticklabels([f"State {i}" for i in range(self.n_states)])
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        return ax

    def plot_distributions(self, X: Optional[np.ndarray] = None, feature_idx: int = 0, x_range: Optional[tuple] = None,
                           n_points: int = 200, figsize=(12, 4)):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        params = self.get_mixture_parameters()

        # Determine x_range
        if x_range is None:
            if X is not None:
                x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
                margin = (x_max - x_min) * 0.2
                x_range = (x_min - margin, x_max + margin)
            else:
                # Infer from component means
                all_means = []
                for p in params:
                    for comp in p["components"]:
                        if "mean" in comp:
                            all_means.append(comp["mean"][feature_idx])
                x_center = np.mean(all_means)
                x_range = (x_center - 5, x_center + 5)

        x = np.linspace(x_range[0], x_range[1], n_points)

        fig, axes = plt.subplots(1, self.n_states, figsize=figsize, squeeze=False)
        axes = axes.flatten()

        for state_idx in range(self.n_states):
            ax = axes[state_idx]
            state_params = params[state_idx]

            # Plot data histogram if provided
            if X is not None:
                states = self.predict(X)
                state_data = X[states == state_idx, feature_idx]
                if len(state_data) > 0:
                    ax.hist(
                        state_data,
                        bins=30,
                        density=True,
                        alpha=0.3,
                        color="gray",
                        label="Data",
                    )

            # Plot mixture distribution
            total_density = np.zeros(n_points)
            weights = state_params["weights"]

            for comp_idx, comp in enumerate(state_params["components"]):
                weight = (
                    weights[comp_idx]
                    if weights is not None
                    else 1.0 / self.n_components
                )

                # Calculate component density
                if "mean" in comp and "cov" in comp:
                    mean = comp["mean"][feature_idx]
                    std = np.sqrt(comp["cov"][feature_idx, feature_idx])

                    if self.distribution_type == DistType.STUDENTT and "dof" in comp:
                        from scipy.stats import t

                        dof = comp["dof"][feature_idx]
                        density = t.pdf(x, df=dof, loc=mean, scale=std)
                    elif self.distribution_type == DistType.LOGNORMAL:
                        from scipy.stats import lognorm

                        density = lognorm.pdf(x, s=std, scale=np.exp(mean))
                    else:  # Normal and others
                        density = norm.pdf(x, loc=mean, scale=std)

                    # Plot individual component
                    ax.plot(
                        x, weight * density, "--", alpha=0.5, label=f"Comp {comp_idx}"
                    )
                    total_density += weight * density

            # Plot total mixture density
            ax.plot(x, total_density, "k-", linewidth=2, label="Mixture")

            ax.set_title(f"State {state_idx}")
            ax.set_xlabel("Value")
            if state_idx == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"Mixture Distributions ({self.distribution_type.name}, {self.n_components} components)",
            fontsize=12,
            y=1.02,
        )
        plt.tight_layout()

        return fig, axes

    def regime_summary(self, X: Optional[np.ndarray] = None) -> str:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        params = self.get_mixture_parameters()
        trans_prob = self.transition_prob()

        lines = []
        lines.append("=" * 70)
        lines.append(f"Mixture HMM Regime Summary ({self.distribution_type.name})")
        lines.append("=" * 70)
        lines.append(f"Number of states: {self.n_states}")
        lines.append(f"Components per state: {self.n_components}")
        lines.append("")

        # State occupancy if data provided
        if X is not None:
            states = self.predict(X)
            lines.append("State Occupancy:")
            for i in range(self.n_states):
                count = np.sum(states == i)
                pct = 100 * count / len(states)
                lines.append(f"  State {i}: {count:6d} samples ({pct:5.2f}%)")
            lines.append("")

        # Transition probabilities
        lines.append("Transition Probabilities:")
        for i in range(self.n_states):
            probs = " ".join([f"{p:6.3f}" for p in trans_prob[i]])
            lines.append(f"  From State {i}: [{probs}]")
        lines.append("")

        # Component details per state
        for state_idx, state_params in enumerate(params):
            lines.append(f"State {state_idx} Mixture Components:")
            weights = state_params["weights"]

            for comp_idx, comp in enumerate(state_params["components"]):
                weight = (
                    weights[comp_idx]
                    if weights is not None
                    else 1.0 / self.n_components
                )
                lines.append(f"  Component {comp_idx} (weight={weight:.3f}):")

                if "mean" in comp:
                    mean_str = np.array2string(
                        comp["mean"], precision=4, suppress_small=True
                    )
                    lines.append(f"    Mean: {mean_str}")
                if "cov" in comp:
                    std = np.sqrt(np.diag(comp["cov"]))
                    std_str = np.array2string(std, precision=4, suppress_small=True)
                    lines.append(f"    Std:  {std_str}")
                if "dof" in comp:
                    dof_str = np.array2string(
                        comp["dof"], precision=2, suppress_small=True
                    )
                    lines.append(f"    DoF:  {dof_str}")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _student_t_covariance_config(self) -> tuple[str, float]:
        covariance_type = str(self.dist_kwargs.get("covariance_type", "diag")).lower()
        if covariance_type not in {"full", "diag", "sphere"}:
            covariance_type = "diag"
        if covariance_type == "full":
            raise ValueError(
                "covariance_type='full' is not supported for StudentT. "
                "pomegranate's StudentT.log_probability uses element-wise division by covs, "
                "which requires covs shape (d,) not (d, d). Use 'diag' or 'sphere' instead."
            )

        min_cov = self.dist_kwargs.get("min_cov")
        min_cov = 1e-6 if min_cov is None else float(min_cov)
        min_cov = max(min_cov, 1e-12)
        return covariance_type, min_cov

    def _student_t_component_init(self, state_idx: int, rng: np.random.Generator) -> tuple[np.ndarray | None, np.ndarray | None]:
        stats = getattr(self, "_kmeans_stats", None)
        if not isinstance(stats, dict):
            return None, None

        means = np.asarray(stats["centroids"][state_idx], dtype=np.float64).copy()
        covs_diag = np.asarray(stats["covs_diag"][state_idx], dtype=np.float64).copy()
        covs_full = np.asarray(stats["covs_full"][state_idx], dtype=np.float64).copy()
        covariance_type, min_cov = self._student_t_covariance_config()

        covs_diag = np.maximum(covs_diag, min_cov)
        jitter = rng.normal(0.0, np.sqrt(covs_diag) * 0.05, size=means.shape)
        means = means + jitter

        if covariance_type == "diag":
            return means, covs_diag
        if covariance_type == "sphere":
            return means, np.array([float(np.maximum(covs_diag.mean(), min_cov))], dtype=np.float64)

        covs_full = np.asarray(covs_full, dtype=np.float64)
        covs_full[np.diag_indices_from(covs_full)] = np.maximum(
            np.diag(covs_full), min_cov
        )
        return means, covs_full

    def _build_model(self) -> DenseHMM:
        """Construct the pomegranate HMM with mixture distributions."""
        distributions = []
        rng = np.random.default_rng(self.random_state)

        for state_idx in range(self.n_states):
            # Build mixture components
            component_dists = []
            for comp_idx in range(self.n_components):
                comp_dist = self._build_component_distribution(state_idx, comp_idx, rng)
                component_dists.append(comp_dist)

            # Create mixture model
            mixture = GeneralMixtureModel(distributions=component_dists)
            distributions.append(mixture)

        return DenseHMM(distributions=distributions, max_iter=self.max_iter,
            random_state=self.random_state,
            dtype=torch.float64)

    def _build_component_distribution(self, state_idx: int, comp_idx: int, rng: np.random.Generator):
        match self.distribution_type:
            case DistType.NORMAL:
                covariance_type = self.dist_kwargs.get("covariance_type", "full")
                min_cov = self.dist_kwargs.get("min_cov")
                return Normal(covariance_type=covariance_type, min_cov=min_cov, dtype=torch.float64)
            case DistType.STUDENTT:
                dofs = self.dist_kwargs.get("dofs", 3)
                covariance_type, min_cov = self._student_t_covariance_config()
                means, covs = self._student_t_component_init(state_idx, rng)
                return StudentT(
                    dofs=dofs,
                    means=means,
                    covs=covs,
                    covariance_type=covariance_type,
                    min_cov=min_cov,
                    dtype=torch.float64,
                )
            case DistType.LOGNORMAL:
                mu_start = rng.normal(0.0, 1.0)
                sigma_start = rng.uniform(0.2, 1.0)
                means = np.array([mu_start], dtype=np.float64)
                covs = np.array([[sigma_start**2]], dtype=np.float64)
                return LogNormal(means=means, covs=covs, dtype=torch.float64)
            case DistType.GAMMA:
                return Gamma(dtype=torch.float64)
            case DistType.EXPONENTIAL:
                return Exponential(dtype=torch.float64)
            case DistType.LAMDA:
                return Lambda(dtype=torch.float64)
            case _:
                raise ValueError(
                    f"Distribution type {self.distribution_type} not supported for mixture models"
                )

    def __repr__(self) -> str:
        return (
            f"PomegranateMixtureHMM("
            f"n_states={self.n_states}, "
            f"n_components={self.n_components}, "
            f"distribution={self.distribution_type.name}, "
            f"backend=DenseHMM)"
        )
