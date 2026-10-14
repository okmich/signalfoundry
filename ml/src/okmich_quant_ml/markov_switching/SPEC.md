# Markov-Switching Models — Specification

**Status:** ✅ Production-Ready
**Version:** 2.1
**Date:** 2026-10-14

---

## Overview

The `markov_switching` subpackage provides a family of regime-dependent time series models built on a shared
`BaseMarkovSwitching` base class with custom EM (Baum-Welch) implementations.

| Model | Class | Use Case |
|-------|-------|----------|
| MS-AR | `MarkovSwitchingAR` | Univariate AR with regime-switching mean and variance |
| MS-OU | `MarkovSwitchingOU` | Mean-reversion with OU reparametrisation |
| MS-GARCH | `MarkovSwitchingGARCH` | Regime-switching GARCH(1,1) conditional volatility |
| MS-VAR | `MarkovSwitchingVAR` | Multi-asset VAR with regime-switching covariance |

All models support:
- Baum-Welch E-step (filtered + smoothed regime probabilities)
- Closed-form or SLSQP M-step
- `forecast()` with regime uncertainty (law of total expectation/variance)
- Causal streaming `update()` that reproduces the batch filter bar for bar
- `predict_regime*()` defaulting to `causal=True` (filtered)
- `save()` / `load()` via joblib
- AIC / BIC

---

## Package Structure

```
projects/ml/src/okmich_quant_ml/markov_switching/
├── __init__.py          # Exports all model classes
├── base.py              # BaseMarkovSwitching
├── kernels.py           # Numba-JIT inner-loop kernels
├── ar.py                # MarkovSwitchingAR
├── ou.py                # MarkovSwitchingOU
├── garch.py             # MarkovSwitchingGARCH
└── var.py               # MarkovSwitchingVAR

projects/ml/tests/markov_switching/
├── test_ms_ar_*         # via tests/hmm/test_markov_switching_ar.py (36 tests)
├── test_ms_ou.py        # 24 tests
├── test_online_learning.py   # 22 tests  (L1 streaming)
├── test_sliding_window.py    # 17 tests  (L2 sliding-window refit)
├── test_ms_garch.py     # 31 tests
├── test_ms_var.py       # 41 tests
└── test_mean_form_and_streaming.py  # 25 tests (mean-form filter/forecast exactness, θ, GARCH/VAR streaming, guards)
```

---

## Import

```python
from okmich_quant_ml.markov_switching import (
    MarkovSwitchingAR,
    MarkovSwitchingOU,
    MarkovSwitchingGARCH,
    MarkovSwitchingVAR,
)
```

---

## Base Class

`BaseMarkovSwitching` provides the shared interface for all models.

### Shared attributes (set after `fit()`)

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `transition_matrix_` | (K, K) | Row-stochastic: P[i,j] = P(s_t=j \| s_{t-1}=i) |
| `filtered_probabilities_` | (T, K) | P(s_t \| y_{1:t}) — causal |
| `regime_probabilities_` | (T, K) | P(s_t \| y_{1:T}) — smoothed, look-ahead |
| `data_` | (T,) or (T,n) | Training data |
| `aic`, `bic` | float | Information criteria |
| `is_fitted` | bool | Guard flag |

### Shared methods

```python
model.get_transition_matrix()                 # (K, K) row-stochastic
model.forecast_regime_probabilities(steps=10) # (steps, K)
model.current_regime_proba()                  # (K,) filtered belief at the newest bar
model.current_regime()                        # int, argmax of above
model.get_aic_bic()                           # (aic, bic)
model.save("path.joblib")
model = ModelClass.load("path.joblib")
```

---

## Numba Kernels (`kernels.py`)

Performance-critical loops compiled with `@njit(cache=True)`.

| Kernel | Purpose |
|--------|---------|
| `_ms_ar_joint_forward_step` | One causal Hamilton-filter step of the mean-form MS-AR(p) on the joint path (s_t, …, s_{t−p}) |
| `_ms_ar_deviation_moments` | Regime-split moments of the deviation state Z_T = (y_T − μ_{s_T}, …) under a joint path distribution |
| `_ms_ar_forecast_moments` | Exact h-step MS-AR predictive moments, split by terminal regime |
| `_companion` | AR(p) companion matrix |
| `_ar_variance_kernel` | MA(∞) forecast variance via ψ-weight recursion |
| `_garch_filter_kernel` | GARCH(1,1) variance filter + log-likelihood sequence |
| `_propagate_regime_probs` | Forward regime probability propagation via transition matrix |

---

## MarkovSwitchingAR

Univariate regime-dependent AR(p) via statsmodels MLE (EM start + BFGS).

### Model — Hamilton (1989) mean form

```
y_t − μ_{s_t} = Σ_{l=1}^{p} φ_{s_t,l} (y_{t−l} − μ_{s_{t−l}}) + ε_t,   ε_t ~ N(0, σ²_{s_t})
```

- statsmodels' `const[k]` is the regime **mean** μ_k, not an intercept. Each lag is centred on the regime active at
  that lag, so the conditional mean depends on the last p+1 regimes. Filtering, `update()` and forecasting all run
  on the joint path (s_t, …, s_{t−p}) — K^(p+1) states.
- statsmodels 0.14.5 (the pinned version) keys the switching variance on s_{t−p+1} instead of s_t whenever order ≥ 2 (broadcasting
  slip in `_conditional_loglikelihoods`). The private `_HamiltonMarkovAutoregression` subclass fixes it; all fits go
  through it.
- The start-parameter search draws from NumPy's global RNG; `fit()` seeds it with `random_state` and restores it, so
  fits are reproducible.

### Construction & fitting

```python
ms_ar = MarkovSwitchingAR(n_regimes=2, order=2, switching_variance=True, random_state=42)
ms_ar.fit(returns, num_restarts=3)

# Auto hyperparameter search — adopts the winner's full state (streaming filter included)
ms_ar.train(returns, order_range=(1, 3), n_regimes_range=(2, 3), criterion='bic')

ms_ar.get_regime_parameters()
# columns: regime, mean (μ), intercept (μ(1−Σφ), valid while the regime persists), ar_L1..p,
#          variance, persistence, half_life
```

### Inference

```python
# causal=True (default) → filtered (no look-ahead), use for live trading / ML labels
# causal=False          → smoothed (look-ahead), use for diagnostics only
regimes = ms_ar.predict_regime()
probs   = ms_ar.predict_regime_proba()
```

### Forecasting

Exact predictive moments of the mean-form model (no stay-in-regime mixture approximation). With
m_h(j) = E[Z_{T+h} 1{s_{T+h}=j}] and S_h(j) its second moment, Z the deviation state and F_j the companion matrix:
`m_h(j) = F_j Σ_i P[i,j] m_{h−1}(i)`, `S_h(j) = F_j (Σ_i P[i,j] S_{h−1}(i)) F_j' + π_h(j) σ²_j e₁e₁'`.

```python
# Regime-mixed mean forecast
mean = ms_ar.forecast(steps=10)

result = ms_ar.forecast(steps=10, return_variance=True)
# result['mean']                (steps,)   E[y_{T+h} | y_{1:T}]
# result['variance']            (steps,)   Var[y_{T+h} | y_{1:T}]
# result['regime_probabilities'] (steps, K)
# result['regime_forecasts']    dict[int, (steps,)]   E[y_{T+h} | s_{T+h}=r, y_{1:T}]
#                               (mean == Σ_r regime_probabilities[:, r] · regime_forecasts[r])

# Stay-in-regime forecast: conditional on s_{T+1..T+h} = 0; the past path is re-weighted by P(s_T → 0)
mean_r0 = ms_ar.forecast(steps=10, regime=0)
```

Models fitted with `exog` raise `NotImplementedError` from `forecast()` and `update()`: the mean-form conditional
mean needs each new bar's exog values.

### Online Learning — L1 (streaming filter)

Processes new bars without refitting. Runs the same joint-path Hamilton filter statsmodels uses, so streaming
y[N:] after fitting on y[:N] reproduces `filter()` on y[:N+M] with the fitted params to machine precision.
O(K^(p+2)) per observation.

```python
ms_ar.update(y_new)                  # scalar or array
ms_ar.current_regime_proba()         # (K,) filtered at most recent bar
ms_ar.current_regime()               # int, argmax of above

# forecast() and forecast_regime_probabilities(causal=True) automatically
# project forward from the updated belief state after update() calls.
```

### Online Learning — L2 (sliding-window refit)

Maintains a rolling buffer; refits EM every `refit_every` bars.

```python
ms_ar.refit_window(y_new, window=252, refit_every=5, num_restarts=3)
# Between refits → L1 forward step keeps belief state current
# n_refits_ tracks refit count
```

---

## MarkovSwitchingOU

Subclass of `MarkovSwitchingAR` with `order=1` fixed. Reparametrises the AR(1)
coefficient into continuous-time OU parameters.

### OU reparametrisation

```
φ = AR(1) coefficient
κ = −ln(φ) / Δt          (speed of mean reversion)
θ = μ                     (the fitted regime mean — statsmodels' const; NOT c / (1 − φ))
σ²_cont = σ²_disc · 2κ / (1 − φ²)
t½ = ln(2) / κ            (half-life)
```

Requires 0 < φ < 1 for a valid OU interpretation. Before v2.1, θ was computed as const / (1 − φ), which inflated it
by 1/(1 − φ) (20× at φ = 0.95).

Fitting caveat: statsmodels starts both regimes at the pooled mean. When the regime levels are many stationary
standard deviations apart, EM can stay in a degenerate optimum (φ near ±1, equal θ). Standardise the series or check
`get_ou_parameters()` before trusting θ.

### Usage

```python
ms_ou = MarkovSwitchingOU(n_regimes=2, dt=1/252)
ms_ou.fit(spread, num_restarts=3)

params = ms_ou.get_ou_parameters()
# columns: regime, phi, kappa, theta, sigma, sigma_disc, half_life, is_mean_reverting

ms_ou.is_mean_reverting(regime=0)     # bool
ms_ou.interpret_regimes()             # {0: 'Fast MR t½=3.2 (low_vol)', ...}
fig = ms_ou.plot_mean_reversion()
```

---

## MarkovSwitchingGARCH

Custom EM (no statsmodels). GARCH(1,1) volatility per regime.

### Model

```
y_t = μ^(k) + Σ φ_j y_{t−j} + ε_t
ε_t = σ_t z_t,   z_t ~ N(0,1)
σ_t² = ω^(k) + α^(k) ε_{t−1}² + β^(k) σ_{t−1}²
```

### EM

- **E-step:** Per-regime GARCH filter → Baum-Welch forward-backward
- **M-step:** Closed-form transition matrix; SLSQP for AR+GARCH params
  - ω positivity via log-transform; α+β < 1 via inequality constraint

### Usage

```python
ms_g = MarkovSwitchingGARCH(n_regimes=2, order=1, random_state=0)
ms_g.fit(returns, num_restarts=3, maxiter=100)

result = ms_g.forecast(steps=10, return_variance=True)
# result['mean']                (steps,)
# result['variance']            (steps,)   GARCH-adjusted via MA(∞) impulse response
# result['regime_probabilities'] (steps, K)

params = ms_g.get_regime_parameters()
# columns: regime, intercept, ar_L1, omega, alpha, beta, garch_persistence, unconditional_variance

# Causal streaming filter: advances each regime's GARCH recursion and AR residual by one bar
ms_g.update(y_new)                  # scalar or 1-D array
ms_g.current_regime_proba()
```

The GARCH mean equation is intercept form (its own EM defines it so), unlike MS-AR. After EM, `fit()` re-runs the
E-step on the final parameters, so the stored probabilities, GARCH state and log-likelihood describe the stored model.

---

## MarkovSwitchingVAR

Custom EM for multi-asset (n ≥ 2) Vector AR(p).

### Model

```
Y_t = μ^(k) + A_1^(k) Y_{t−1} + … + A_p^(k) Y_{t−p} + ε_t
ε_t ~ N(0, Σ^(k))
```

### EM

- **E-step:** Multivariate Gaussian log-likelihood via Cholesky → Baum-Welch forward-backward
- **M-step (all closed-form):**
  - Transition matrix from two-slice marginals ξ
  - VAR params: weighted OLS `B_k = (X'WX + λI)⁻¹ X'WY`
  - Covariance: `Σ_k = Σ_t γ_t(k) ε_t ε_t' / Σ_t γ_t(k) + λI`
- Ridge regularisation (`ridge=1e-6`) prevents singularity at low regime occupancy

### Forecast

- **Mean:** Recursive VAR substitution
- **Covariance:** Impulse-response `Cov[Y_{T+h}|k] = Σ_{j=0}^{h-1} Ψ_j Σ^(k) Ψ_j'`
  where Ψ_0=I, Ψ_j = Σ_{m=1}^{min(j,p)} A_m Ψ_{j−m}
- Regime-weighted via law of total expectation / total covariance

### Usage

```python
ms_var = MarkovSwitchingVAR(n_regimes=2, order=1, ridge=1e-6, random_state=0)
ms_var.fit(returns_matrix, num_restarts=3)   # returns_matrix shape (T, n_assets)

# Mean forecast only
mean = ms_var.forecast(steps=5)              # (5, n_assets)

# With covariance (law of total covariance)
result = ms_var.forecast(steps=5, return_covariance=True)
# result['mean']                 (steps, n_assets)
# result['covariance']           (steps, n_assets, n_assets)   PSD, symmetric
# result['regime_probabilities'] (steps, K)
# result['regime_forecasts']     dict[int, (steps, n_assets)]

# Single-regime forecast
mean_r0 = ms_var.forecast(steps=5, regime=0)

# Parameter inspection
summary = ms_var.get_regime_parameters()
# columns: regime, spectral_radius, avg_volatility, min_volatility,
#          max_volatility, avg_correlation, log_det_Sigma

mats = ms_var.get_var_matrices(regime=0)
# keys: 'intercept' (n,), 'ar_coeffs' [A_1…A_p (n,n)], 'Sigma' (n,n)

# Causal streaming filter (same log-likelihood and forward arithmetic as the E-step)
ms_var.update(new_rows)             # (n_assets,) or (m, n_assets)
ms_var.current_regime_proba()
```

---

## Causality

| Setting | Probabilities used | Safe for live trading? |
|---------|--------------------|------------------------|
| `causal=True` | Filtered P(s_t \| y_{1:t}) | ✅ Yes |
| `causal=False` | Smoothed P(s_t \| y_{1:T}) | ❌ No — look-ahead bias |

Every `causal` argument defaults to `True`: `predict_regime()`, `predict_regime_proba()`, `forecast()` and
`forecast_regime_probabilities()` (before v2.1 the two `predict_regime*` methods defaulted to smoothed). After
`update()` calls, forecasts start from the streaming belief state; at the newest bar filtered and smoothed coincide.

In-sample filtered probabilities are still in-sample: the parameters were fitted on the same bars. For out-of-sample
regime labels, fit on a training window and `update()` through the test window.

---

## Testing

| File | Tests | Coverage |
|------|-------|----------|
| `test_markov_switching_ar.py` | 36 | MS-AR full API |
| `test_ms_ou.py` | 24 | OU reparametrisation, plotting |
| `test_online_learning.py` | 22 | L1 streaming filter |
| `test_sliding_window.py` | 17 | L2 sliding-window refit |
| `test_ms_garch.py` | 31 | GARCH filter, EM, forecasting |
| `test_ms_var.py` | 41 | Multivariate: VAR(1), VAR(2), 2-asset, PSD checks |
| `test_mean_form_and_streaming.py` | 25 | Streaming == batch filter (AR K=2 p=1,2,3 and K=3 p=1; GARCH; VAR p=1,2); non-finite bars rejected; forecasts == brute-force path enumeration; θ recovery; variance keyed on s_t; causal defaults; seeded fits |

---

## Roadmap

| Step | Feature | Status |
|------|---------|--------|
| 0 | `BaseMarkovSwitching` + `markov_switching/` subpackage | ✅ Done |
| 1 | MS-OU (mean-reversion reparametrisation) | ✅ Done |
| 2 | Online Learning L1 (streaming `update()`) | ✅ Done |
| 3 | Online Learning L2 (sliding-window `refit_window()`) | ✅ Done |
| 4 | MS-GARCH (custom EM with GARCH volatility) | ✅ Done |
| 5 | MS-VAR (multi-asset VAR with custom EM) | ✅ Done |
| 6 | GPU / parallel grid search | Deferred |

---

## Changelog

### Version 2.1 (2026-10-14) — correctness

- **MS-AR mean form.** `const[k]` is the regime mean. `update()` was an intercept-form K-state filter and diverged
  from the fit's own filter; it is now the exact joint-path Hamilton filter (matches statsmodels to ~1e-16).
  Forecasts are the exact predictive moments; `forecast(regime=r)` re-weights the past path by P(s_T → r).
  `get_regime_parameters()` gains `mean`; `intercept` is now μ(1−Σφ) (it used to hold μ under the wrong name).
- **statsmodels variance keying.** With order ≥ 2 and switching variance, upstream keyed σ² on s_{t−p+1}.
  Fixed via `_HamiltonMarkovAutoregression`. Fits with order ≥ 2 change.
- **MS-OU θ** = regime mean (was const / (1 − φ)).
- **Causal defaults.** `predict_regime()` / `predict_regime_proba()` default to `causal=True` on AR, OU, GARCH, VAR.
- **GARCH / VAR `update()`.** Causal out-of-sample filtering without refit; `fit()` re-runs the E-step on the final
  parameters so stored probabilities match them.
- **Forward-step underflow.** Every forward step (MS-AR joint filter, GARCH/VAR E-steps and `update()`) now
  normalises in log space over states with prior mass (`_scaled_update`). Shifting by the max log-likelihood let an
  impossible state set the scale, underflow every possible one and fall back to uniform — e.g. on level series
  whose regime means are tens of σ apart. MS-AR also floors transitions at 1e-20 like statsmodels.
- **One filter implementation.** GARCH/VAR E-steps and `update()` call the same numba kernels
  (`_scaled_forward_filter` / `_predict_regime_probs` + `_scaled_update`; `_garch_variance_step`, `_gaussian_loglik`),
  so streaming equals batch by construction. `predict_regime*` live once, in the base class.
- **Guards.** `update()` / `refit_window()` reject non-finite bars before touching state. Fitted MS-AR/OU models
  pickled before v2.1 are refused on load (`__setstate__`), as their θ, filter state and likelihood are wrong.
  `fit()` leaves the previous state intact if statsmodels raises.
- **Performance.** `MarkovSwitchingGARCH._compute_ar_residuals` is vectorised (bitwise-identical): a 3,000-bar,
  30-iteration fit drops from 12.9 s to 2.3 s. MS-VAR `update()` caches Σ Cholesky factors per fit. Mean-only
  MS-AR forecasts skip the second-moment recursion.
- `current_regime_proba()` / `current_regime()` moved to the base class (all families).
- `MarkovSwitchingAR.fit()` honours `random_state`; `train()` adopts the winner's streaming state.
- Removed kernels `_forward_step` and `_ar_forecast_kernel` (intercept form).

### Version 2.0 (2026-03-11)

- Migrated from `hmm/` to dedicated `markov_switching/` subpackage
- Added `BaseMarkovSwitching` shared base class
- Added Numba-JIT kernels (`kernels.py`)
- Added `MarkovSwitchingOU` with OU reparametrisation
- Added online learning: `update()` (L1) and `refit_window()` (L2)
- Added `MarkovSwitchingGARCH` with custom EM and GARCH(1,1) filter
- Added `MarkovSwitchingVAR` with closed-form weighted OLS M-step and impulse-response covariance
- 135 tests, all passing

### Version 1.0 (2026-01-03)

- Initial production release of `MarkovSwitchingAR`
- 36 tests