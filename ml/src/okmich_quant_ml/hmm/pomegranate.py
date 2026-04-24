import matplotlib.pyplot as plt
import numpy as np
import torch
from pomegranate.distributions import Bernoulli, Categorical, Exponential, Gamma, LogNormal, Poisson, Normal, StudentT
from pomegranate.hmm import DenseHMM
from typing import Optional, Sequence

from .base_pomegranate import BasePomegranateHMM
from .distribution import Lambda
from .util import InferenceMode, DistType


class PomegranateHMM(BasePomegranateHMM):
    """
    A convenience wrapper for pomegranate DenseHMM that allows switching the emission distribution at design
    time **and** exposes a unified API:
        fit / train / predict / predict_prob / transition_prob
    plus distribution–specific properties such as means, covs, etc.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, X: np.ndarray, lengths: Optional[Sequence[int]] = None, n_states_range: Sequence[int] = (2, 3, 4, 5), criterion: str = "aic") -> "PomegranateHMM":
        """Train model with automatic state selection via AIC or BIC."""
        _VALID_CRITERIA = {"aic", "bic"}
        if criterion not in _VALID_CRITERIA:
            raise ValueError(f"Unknown criterion '{criterion}'. Allowed values: {sorted(_VALID_CRITERIA)}")

        best_score, best_inst = np.inf, None
        for k in n_states_range:
            for inst_kwargs in self._iter_train_dist_kwargs():
                inst = self.__class__(self.distribution_type, n_states=k, random_state=self.random_state,
                                      max_iter=self.max_iter, inference_mode=self.inference_mode, **inst_kwargs)
                inst.fit(X, lengths)
                aic, bic = inst.get_aic_bic(X)
                score = aic if criterion == "aic" else bic
                if score < best_score:
                    best_score, best_inst = score, inst
        return best_inst

    def get_aic_bic(self, X: np.ndarray) -> tuple[float, float]:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        X = X.reshape(-1, 1) if X.ndim == 1 else X

        ll = float(self.log_likelihood(X))

        # Parameter counting
        n_features = X.shape[1]
        n_emission_params = self.n_states * (n_features + self._cov_param_count(n_features))
        n_trans_params = self.n_states * (self.n_states - 1)
        n_params = (self.n_states - 1) + n_trans_params + n_emission_params

        aic = -2 * ll + 2 * n_params
        bic = -2 * ll + n_params * np.log(X.shape[0])

        return float(aic), float(bic)

    # ------------------------------------------------------------------
    # Distribution–specific properties
    # ------------------------------------------------------------------
    @property
    def means(self):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.extract_property("mean")

    @property
    def covariances(self):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self.extract_property("cov")

    @property
    def parameters(self):
        """
        Returns a list of parameter dicts per state.
        Useful for StudentT (df, loc, scale) etc.
        """
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        results = []
        for d in self._model.distributions:
            if self.distribution_type == DistType.BERNOULLI:
                results.append({"probs": d.probs.detach().cpu().numpy()})
            elif self.distribution_type == DistType.EXPONENTIAL:
                results.append({"scales": d.scales.detach().cpu().numpy()})
            elif self.distribution_type == DistType.GAMMA:
                results.append(
                    {
                        "shapes": d.shapes.detach().cpu().numpy(),
                        "rates": d.rates.detach().cpu().numpy(),
                    }
                )
            elif self.distribution_type == DistType.LAMDA:
                results.append(
                    {
                        "means": d.mu.detach().cpu().numpy(),
                        "sigma": d.sigma.detach().cpu().numpy(),
                        "lambda": d.lambda_.detach().cpu().numpy(),
                    }
                )
            elif self.distribution_type == DistType.POISSON:
                results.append({"lambdas": d.lambdas.detach().cpu().numpy()})
            elif self.distribution_type in [
                DistType.LOGNORMAL,
                DistType.NORMAL,
                DistType.STUDENTT,
            ]:
                _params = {
                    "means": d.means.detach().cpu().numpy(),
                    "covs": d.covs.detach().cpu().numpy(),
                }
                if self.distribution_type == DistType.STUDENTT:
                    _params["dofs"] = d.dofs.detach().cpu().numpy()
                results.append(_params)
            else:
                results.append(d.parameters())
        return results

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

    def _student_t_init_params(self, state_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        stats = getattr(self, "_kmeans_stats", None)
        if not isinstance(stats, dict):
            return None, None

        means = np.asarray(stats["centroids"][state_idx], dtype=np.float64).copy()
        covs_diag = np.asarray(stats["covs_diag"][state_idx], dtype=np.float64).copy()
        covs_full = np.asarray(stats["covs_full"][state_idx], dtype=np.float64).copy()

        covariance_type, min_cov = self._student_t_covariance_config()
        covs_diag = np.maximum(covs_diag, min_cov)

        if covariance_type == "diag":
            return means, covs_diag
        if covariance_type == "sphere":
            return means, np.array([float(np.maximum(covs_diag.mean(), min_cov))], dtype=np.float64)

        # full covariance: floor diagonal to preserve positive definiteness
        covs_full = np.asarray(covs_full, dtype=np.float64)
        covs_full[np.diag_indices_from(covs_full)] = np.maximum(
            np.diag(covs_full), min_cov
        )
        return means, covs_full

    def _build_model(self) -> DenseHMM:
        """Construct the pomegranate HMM with the chosen distribution."""
        distributions = []
        for k in range(self.n_states):
            d = self._build_distribution(k)
            distributions.append(d)

        return DenseHMM(
            distributions=distributions,
            max_iter=self.max_iter,
            random_state=self.random_state,
            dtype=torch.float64,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_distribution(self, k: int):
        rng = np.random.default_rng()

        match self.distribution_type:
            case DistType.BERNOULLI:
                return Bernoulli(dtype=torch.float64)
            case DistType.CATEGORICAL:
                n_cats = self.dist_kwargs.get("n_categories", 3)
                probs = rng.dirichlet(np.ones(n_cats)).reshape(-1, 1)
                return Categorical(probs=probs, dtype=torch.float64)
            case DistType.EXPONENTIAL:
                return Exponential(dtype=torch.float64)
            case DistType.GAMMA:
                return Gamma(dtype=torch.float64)
            case DistType.LAMDA:
                return Lambda(dtype=torch.float64)
            case DistType.LOGNORMAL:
                mu_start = rng.normal(0.0, 1.0)
                sigma_start = rng.uniform(0.2, 1.0)
                means = np.array([mu_start], dtype=np.float64)
                _covs = np.array([[sigma_start**2]], dtype=np.float64)
                return LogNormal(means=means, covs=_covs, dtype=torch.float64)
            case DistType.NORMAL:
                covariance_type = self.dist_kwargs.get("covariance_type", "full")
                min_cov = self.dist_kwargs.get("min_cov")
                return Normal(covariance_type=covariance_type, min_cov=min_cov, dtype=torch.float64)
            case DistType.POISSON:
                return Poisson(dtype=torch.float64)
            case DistType.STUDENTT:
                covariance_type, min_cov = self._student_t_covariance_config()
                means, covs = self._student_t_init_params(k)
                return StudentT(
                    dofs=self.dist_kwargs.get("dofs", 3),
                    means=means,
                    covs=covs,
                    covariance_type=covariance_type,
                    min_cov=min_cov,
                    dtype=torch.float64,
                )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def extract_property(self, prop: str):
        """Generic extractor for distribution parameters."""
        values = []
        for d in self._model.distributions:
            if hasattr(d, prop):
                values.append(getattr(d, prop))
            elif hasattr(d, "distributions"):
                # GeneralMixtureModel or IndependentComponents
                values.append([getattr(sub, prop) for sub in d.distributions])
            else:
                values.append(None)
        return np.array(values, dtype=object)

    def regime_summary(self, X: Optional[np.ndarray] = None) -> str:
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        params = self.parameters
        trans_prob = self.transition_prob()

        lines = []
        lines.append("=" * 70)
        lines.append(f"HMM Regime Summary ({self.distribution_type.name})")
        lines.append("=" * 70)
        lines.append(f"Number of states: {self.n_states}")
        lines.append(f"Distribution type: {self.distribution_type.name}")
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

        # Distribution parameters per state
        lines.append("Distribution Parameters:")
        for state_idx, state_params in enumerate(params):
            lines.append(f"State {state_idx}:")

            # Display parameters
            for key, val in state_params.items():
                if isinstance(val, np.ndarray):
                    val_str = np.array2string(val, precision=4, suppress_small=True)
                else:
                    val_str = str(val)
                lines.append(f"  {key}: {val_str}")

            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def plot_distributions(self, X: Optional[np.ndarray] = None, feature_idx: int = 0, x_range: Optional[tuple] = None,
                           n_points: int = 200, figsize=(12, 4)):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        params = self.parameters

        # Determine x_range
        if x_range is None:
            if X is not None:
                x_min, x_max = X[:, feature_idx].min(), X[:, feature_idx].max()
                margin = (x_max - x_min) * 0.2
                x_range = (x_min - margin, x_max + margin)
            else:
                # Try to infer from means
                try:
                    means_array = self.means
                    if means_array is not None and len(means_array) > 0:
                        all_means = [
                            m[feature_idx] if hasattr(m, "__getitem__") else m
                            for m in means_array
                        ]
                        x_center = np.mean([m for m in all_means if m is not None])
                        x_range = (x_center - 5, x_center + 5)
                    else:
                        x_range = (-5, 5)
                except:
                    x_range = (-5, 5)

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

            # Plot distribution based on type
            if self.distribution_type in [
                DistType.NORMAL,
                DistType.STUDENTT,
                DistType.LOGNORMAL,
            ]:
                # Handle standard distributions with means and covs
                if "means" in state_params and "covs" in state_params:
                    mean = state_params["means"][feature_idx]
                    std = np.sqrt(state_params["covs"][feature_idx, feature_idx])

                    if (
                        self.distribution_type == DistType.STUDENTT
                        and "dofs" in state_params
                    ):
                        from scipy.stats import t

                        dof = state_params["dofs"][feature_idx]
                        density = t.pdf(x, df=dof, loc=mean, scale=std)
                    elif self.distribution_type == DistType.LOGNORMAL:
                        from scipy.stats import lognorm

                        density = lognorm.pdf(x, s=std, scale=np.exp(mean))
                    else:  # Normal
                        from scipy.stats import norm

                        density = norm.pdf(x, loc=mean, scale=std)

                    ax.plot(x, density, "b-", linewidth=2, label="Distribution")

            elif self.distribution_type == DistType.GAMMA:
                # Handle Gamma distribution
                if "shapes" in state_params and "rates" in state_params:
                    from scipy.stats import gamma

                    shape = state_params["shapes"][feature_idx]
                    rate = state_params["rates"][feature_idx]
                    # Gamma parameterization: scale = 1/rate
                    density = gamma.pdf(x, a=shape, scale=1 / rate)
                    ax.plot(x, density, "b-", linewidth=2, label="Distribution")

            elif self.distribution_type == DistType.EXPONENTIAL:
                # Handle Exponential distribution
                if "scales" in state_params:
                    from scipy.stats import expon

                    scale = state_params["scales"][feature_idx]
                    density = expon.pdf(x, scale=scale)
                    ax.plot(x, density, "b-", linewidth=2, label="Distribution")

            ax.set_title(f"State {state_idx}")
            ax.set_xlabel("Value")
            if state_idx == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

        fig.suptitle(
            f"State Distributions ({self.distribution_type.name})", fontsize=12, y=1.02
        )
        plt.tight_layout()

        return fig, axes

    def plot_transition_matrix(self, ax=None, figsize=(8, 6), cmap="Blues"):
        if self._model is None:
            raise RuntimeError("Model has not been fitted yet.")

        trans_prob = self.transition_prob()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        im = ax.imshow(trans_prob, cmap=cmap, aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.set_label("Probability", rotation=270, labelpad=20)

        # Add text annotations
        for i in range(self.n_states):
            for j in range(self.n_states):
                text = ax.text(
                    j,
                    i,
                    f"{trans_prob[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="black" if trans_prob[i, j] < 0.5 else "white",
                )

        ax.set_xticks(np.arange(self.n_states))
        ax.set_yticks(np.arange(self.n_states))
        ax.set_xticklabels([f"State {i}" for i in range(self.n_states)])
        ax.set_yticklabels([f"State {i}" for i in range(self.n_states)])
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        ax.set_title("Transition Probability Matrix")

        return ax

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"PomegranateHMM("
            f"n_states={self.n_states}, "
            f"distribution={str(self.distribution_type)}, "
            f"backend=DenseHMM)"
        )
