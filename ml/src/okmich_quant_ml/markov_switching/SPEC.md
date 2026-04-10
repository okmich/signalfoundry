# Markov-Switching Models — Specification

**Status:** ✅ Production-Ready
**Version:** 2.0
**Date:** 2026-03-11

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
└── test_ms_var.py       # 41 tests

Total: 135 tests — all passing
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
model.get_aic_bic()                           # (aic, bic)
model.save("path.joblib")
model = ModelClass.load("path.joblib")
```

---

## Numba Kernels (`kernels.py`)

Performance-critical loops compiled with `@njit(cache=True)`.

| Kernel | Purpose |
|--------|---------|
| `_ar_forecast_kernel` | Recursive multi-step AR point forecast |
| `_ar_variance_kernel` | MA(∞) forecast variance via ψ-weight recursion |
| `_garch_filter_kernel` | GARCH(1,1) variance filter + log-likelihood sequence |
| `_forward_step` | One causal HMM forward-pass update (log-sum-exp stable) |
| `_propagate_regime_probs` | Forward regime probability propagation via transition matrix |

---

## MarkovSwitchingAR

Univariate regime-dependent AR(p) via statsmodels EM.

### Construction & fitting

```python
ms_ar = MarkovSwitchingAR(n_regimes=2, order=2, switching_variance=True, random_state=42)
ms_ar.fit(returns, num_restarts=3)

# Auto hyperparameter search
ms_ar.train(returns, order_range=(1, 3), n_regimes_range=(2, 3), criterion='bic')
```

### Inference

```python
# causal=True  → filtered (no look-ahead), use for live trading / ML labels
# causal=False → smoothed (look-ahead), use for diagnostics only
regimes = ms_ar.predict_regime(causal=True)
probs   = ms_ar.predict_regime_proba(causal=True)
```

### Forecasting

```python
# Regime-weighted mean forecast
mean = ms_ar.forecast(steps=10)

# With uncertainty (law of total expectation + variance)
result = ms_ar.forecast(steps=10, return_variance=True)
# result['mean']                (steps,)
# result['variance']            (steps,)
# result['regime_probabilities'] (steps, K)
# result['regime_forecasts']    dict[int, (steps,)]

# Single-regime forecast
mean_r0 = ms_ar.forecast(steps=10, regime=0)
```

### Online Learning — L1 (streaming filter)

Processes new bars without refitting. O(K²) per observation.

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
θ = c / (1 − φ)           (long-run mean)
σ²_cont = σ²_disc · 2κ / (1 − φ²)
t½ = ln(2) / κ            (half-life)
```

Requires 0 < φ < 1 for a valid OU interpretation.

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
```

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
```

---

## Causality

| Setting | Probabilities used | Safe for live trading? |
|---------|--------------------|------------------------|
| `causal=True` | Filtered P(s_t \| y_{1:t}) | ✅ Yes |
| `causal=False` | Smoothed P(s_t \| y_{1:T}) | ❌ No — look-ahead bias |

`forecast()` and `forecast_regime_probabilities()` default to `causal=True`.
After `update()` calls, `causal=True` automatically propagates from the streaming belief state.

---

## Testing

```
135 tests across 6 test files — all passing
```

| File | Tests | Coverage |
|------|-------|----------|
| `test_markov_switching_ar.py` | 36 | MS-AR full API |
| `test_ms_ou.py` | 24 | OU reparametrisation, plotting |
| `test_online_learning.py` | 22 | L1 streaming filter |
| `test_sliding_window.py` | 17 | L2 sliding-window refit |
| `test_ms_garch.py` | 31 | GARCH filter, EM, forecasting |
| `test_ms_var.py` | 41 | Multivariate: VAR(1), VAR(2), 2-asset, PSD checks |

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