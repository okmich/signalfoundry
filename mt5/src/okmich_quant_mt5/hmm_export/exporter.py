"""Turn a fitted Gaussian HMM into ``HmmParams.mqh``.

The generated header carries the whole model *and its feature recipe*: state
priors, transition matrix, precomputed inverse covariances and normalising
constants, the MACD/ATR periods, and the ordered feature names. MQL5 never
inverts a matrix, and the periods cannot drift away from the ones used at fit
time because they are not user inputs on the MQL5 side.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

import numpy as np

from .features import FeatureSpec, Mql5Feature
from .kernel import Mql5HmmParams

#: Mirrors ``BasePomegranateHMM._extract_hmm_parameters._to_log_numpy``.
_PROB_FLOOR = 1e-300


class StateOrder(StrEnum):
    """How to canonicalise state indices at export time.

    EM is invariant to relabelling, so two refits of the same data can emit the
    same model with states permuted. Any EA whose rules are written against
    "state 0" then silently inverts. Fixing an ordering at export removes that.
    """

    NONE = "none"
    BY_FEATURE_MEAN = "by_feature_mean"


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _to_log(arr: np.ndarray) -> np.ndarray:
    """Probabilities -> logs, with the same floor Python applies internally.

    Detects already-logged input the same way the ml package does: a parameter
    block that is entirely non-negative must be probabilities.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if np.all(arr >= 0.0):
        arr = np.log(np.maximum(arr, _PROB_FLOOR))
    return arr


def _as_covariance_matrix(covs: np.ndarray, n_features: int) -> np.ndarray:
    """Accept full ``(D, D)`` or diagonal ``(D,)`` covariances, return ``(D, D)``."""
    covs = np.asarray(covs, dtype=np.float64)
    if covs.ndim == 1:
        if covs.shape[0] != n_features:
            raise ValueError(f"diagonal covariance has {covs.shape[0]} entries, expected {n_features}")
        return np.diag(covs)
    if covs.shape != (n_features, n_features):
        raise ValueError(f"covariance has shape {covs.shape}, expected ({n_features}, {n_features})")
    return covs


def _emission_constants(covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute ``Sigma^-1`` and ``-0.5 * (D*log(2pi) + logdet(Sigma))`` per state."""
    n_states, n_features = covs.shape[0], covs.shape[1]
    sigma_inv = np.empty_like(covs)
    log_norm = np.empty(n_states, dtype=np.float64)

    for k in range(n_states):
        cov = 0.5 * (covs[k] + covs[k].T)  # symmetrise away accumulated float drift
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"covariance for state {k} is not positive definite; the fit is degenerate or "
                "min_cov was set too low. Refit before exporting."
            ) from exc

        sign, logdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError(f"covariance for state {k} has non-positive determinant (sign={sign})")

        sigma_inv[k] = np.linalg.inv(cov)
        log_norm[k] = -0.5 * (n_features * math.log(2.0 * math.pi) + logdet)

    return sigma_inv, log_norm


def _permutation(order: StateOrder, mu: np.ndarray, order_feature: int) -> np.ndarray:
    if order == StateOrder.NONE:
        return np.arange(mu.shape[0])
    if not 0 <= order_feature < mu.shape[1]:
        raise ValueError(f"order_feature {order_feature} out of range for {mu.shape[1]} features")
    return np.argsort(mu[:, order_feature], kind="stable")


def build_params(*, model_id: str, spec: FeatureSpec, starts, edges, means, covariances,
                 feat_mean: np.ndarray | None = None, feat_sd: np.ndarray | None = None,
                 state_order: StateOrder = StateOrder.BY_FEATURE_MEAN, order_feature: int = 0) -> Mql5HmmParams:
    """Assemble exportable parameters from raw fitted arrays.

    ``starts`` / ``edges`` may be probabilities or logs - the same detection the
    ml package uses is applied. ``covariances`` may be full ``(K, D, D)`` or
    diagonal ``(K, D)``.
    """
    means = _to_numpy(means)
    if means.ndim != 2:
        raise ValueError(f"means must be (K, D), got shape {means.shape}")

    n_states, n_features = means.shape
    if n_features != spec.n_features:
        raise ValueError(f"means has {n_features} columns but spec declares {spec.n_features} features")

    covs_raw = _to_numpy(covariances)
    covs = np.stack([_as_covariance_matrix(covs_raw[k], n_features) for k in range(n_states)])

    log_pi = _to_log(_to_numpy(starts))
    log_a = _to_log(_to_numpy(edges))
    if log_pi.shape != (n_states,):
        raise ValueError(f"starts must have shape ({n_states},), got {log_pi.shape}")
    if log_a.shape != (n_states, n_states):
        raise ValueError(f"edges must have shape ({n_states}, {n_states}), got {log_a.shape}")

    perm = _permutation(state_order, means, order_feature)
    log_pi, log_a = log_pi[perm], log_a[np.ix_(perm, perm)]
    means, covs = means[perm], covs[perm]

    sigma_inv, log_norm = _emission_constants(covs)

    feat_mean = np.zeros(n_features) if feat_mean is None else np.asarray(feat_mean, dtype=np.float64)
    feat_sd = np.ones(n_features) if feat_sd is None else np.asarray(feat_sd, dtype=np.float64)

    return Mql5HmmParams(model_id=model_id, spec=spec, log_pi=log_pi, log_a=log_a, mu=means,
                         sigma_inv=sigma_inv, log_norm=log_norm, feat_mean=feat_mean, feat_sd=feat_sd)


def params_from_model(model, spec: FeatureSpec, model_id: str, **kwargs) -> Mql5HmmParams:
    """Extract exportable parameters from a fitted ``PomegranateHMM``.

    Duck-typed against the ml package so this module carries no import-time
    dependency on it. Reads the same attributes ``_compute_log_emissions`` and
    ``_extract_hmm_parameters`` read, so the exported model is the one that was
    actually scored in Python.
    """
    inner = getattr(model, "_model", None)
    if inner is None:
        raise TypeError(f"{type(model).__name__} has no fitted ._model; call fit() before exporting")

    dist_type = getattr(model, "distribution_type", None)
    dist_name = getattr(dist_type, "name", str(dist_type))
    if dist_name != "NORMAL":
        raise NotImplementedError(
            f"the MQL5 kernel implements Gaussian emissions only, got {dist_name}. "
            "Other distributions need a matching log-pdf in HmmCore.mqh."
        )

    dists = list(inner.distributions)
    means = np.stack([_to_numpy(d.means) for d in dists])
    covariances = np.stack([_as_covariance_matrix(_to_numpy(d.covs), means.shape[1]) for d in dists])

    return build_params(model_id=model_id, spec=spec, starts=inner.starts, edges=inner.edges, means=means,
                        covariances=covariances, **kwargs)


def _mql_double(value: float) -> str:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"cannot emit non-finite parameter {value!r} into MQL5")
    return repr(value)


def _mql_array(values, per_line: int = 4) -> str:
    items = [_mql_double(v) for v in np.asarray(values).ravel()]
    lines = [", ".join(items[i : i + per_line]) for i in range(0, len(items), per_line)]
    return ",\n      ".join(lines)


def params_hash(params: Mql5HmmParams) -> str:
    """Stable digest of the numeric block, so a dump can be tied to a model version."""
    digest = hashlib.sha256()
    for arr in (params.log_pi, params.log_a, params.mu, params.sigma_inv, params.log_norm,
                params.feat_mean, params.feat_sd):
        digest.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    digest.update("|".join(params.spec.names).encode())
    digest.update(f"{params.spec.macd_fast}/{params.spec.macd_slow}/{params.spec.macd_signal}".encode())
    digest.update(str(params.spec.atr_period).encode())
    return digest.hexdigest()[:16]


def write_mqh(params: Mql5HmmParams, path: str | Path) -> Path:
    """Write ``HmmParams.mqh``. Returns the path written."""
    path = Path(path)
    k, d = params.n_states, params.n_features

    if params.spec.price_scaled:
        names = ", ".join(params.spec.price_scaled)
        print(
            f"WARNING: exporting price-scaled features [{names}]. Emissions fitted at one price "
            "level are evaluated off-distribution at another, and the posterior will collapse onto "
            "a single state. Prefer macd_atr / atr_close / macd_close."
        )

    feature_names = ", ".join(f'"{name}"' for name in params.spec.names)
    body = f"""//+------------------------------------------------------------------+
//|                                                    HmmParams.mqh |
//|                                                                   |
//|  GENERATED BY okmich_quant_mt5.hmm_export.exporter - DO NOT EDIT. |
//|  Regenerate with write_mqh() whenever the model is refitted.      |
//|                                                                   |
//|  model id   : {params.model_id}
//|  generated  : {datetime.now(UTC).isoformat(timespec="seconds")}
//|  param hash : {params_hash(params)}
//|  states     : {k}
//|  features   : {", ".join(params.spec.names)}
//|  MACD       : {params.spec.macd_fast}/{params.spec.macd_slow}/{params.spec.macd_signal}
//|  ATR        : {params.spec.atr_period}
//+------------------------------------------------------------------+
#ifndef HMM_PARAMS_MQH
#define HMM_PARAMS_MQH

#define HMM_K {k}
#define HMM_D {d}

const string HMM_MODEL_ID   = "{params.model_id}";
const string HMM_PARAM_HASH = "{params_hash(params)}";

//--- Ordered feature recipe. Column order is part of the contract.
const string HMM_FEATURE_NAMES[HMM_D] = {{{feature_names}}};

//--- Feature periods travel with the model; they are NOT user inputs.
const int HMM_MACD_FAST   = {params.spec.macd_fast};
const int HMM_MACD_SLOW   = {params.spec.macd_slow};
const int HMM_MACD_SIGNAL = {params.spec.macd_signal};
const int HMM_ATR_PERIOD  = {params.spec.atr_period};

//--- log initial state probabilities, floored at log(1e-300)
const double HMM_LOG_PI[HMM_K] =
   {{
      {_mql_array(params.log_pi)}
   }};

//--- log transition matrix, row-major: HMM_LOG_A[i * HMM_K + j] = log P(j | i)
const double HMM_LOG_A[HMM_K * HMM_K] =
   {{
      {_mql_array(params.log_a)}
   }};

//--- state means, row-major: HMM_MU[k * HMM_D + d]
const double HMM_MU[HMM_K * HMM_D] =
   {{
      {_mql_array(params.mu)}
   }};

//--- inverse covariances, row-major: HMM_SIGINV[k * HMM_D * HMM_D + a * HMM_D + b]
const double HMM_SIGINV[HMM_K * HMM_D * HMM_D] =
   {{
      {_mql_array(params.sigma_inv)}
   }};

//--- -0.5 * (D * log(2pi) + logdet(Sigma_k))
const double HMM_LOGNORM[HMM_K] =
   {{
      {_mql_array(params.log_norm)}
   }};

//--- affine feature standardisation applied before the emission is evaluated
const double HMM_FEAT_MEAN[HMM_D] =
   {{
      {_mql_array(params.feat_mean)}
   }};

const double HMM_FEAT_SD[HMM_D] =
   {{
      {_mql_array(params.feat_sd)}
   }};

#endif // HMM_PARAMS_MQH
//+------------------------------------------------------------------+
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="ascii")
    return path


_DEFINE_RE = re.compile(r"#define\s+(HMM_\w+)\s+(\d+)")
_INT_RE = re.compile(r"const\s+int\s+(HMM_\w+)\s*=\s*(-?\d+)\s*;")
_STR_RE = re.compile(r'const\s+string\s+(HMM_\w+)\s*=\s*"([^"]*)"\s*;')
_STR_ARR_RE = re.compile(r"const\s+string\s+(HMM_\w+)\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}\s*;")
_DBL_ARR_RE = re.compile(r"const\s+double\s+(HMM_\w+)\s*\[[^\]]*\]\s*=\s*\{([^}]*)\}\s*;", re.DOTALL)


def parse_mqh(path: str | Path) -> dict:
    """Read a generated header back into plain Python. Used by the round-trip test."""
    text = Path(path).read_text(encoding="ascii")
    out: dict = {}
    out.update({m.group(1): int(m.group(2)) for m in _DEFINE_RE.finditer(text)})
    out.update({m.group(1): int(m.group(2)) for m in _INT_RE.finditer(text)})
    out.update({m.group(1): m.group(2) for m in _STR_RE.finditer(text)})

    for match in _STR_ARR_RE.finditer(text):
        out[match.group(1)] = [item.strip().strip('"') for item in match.group(2).split(",") if item.strip()]

    for match in _DBL_ARR_RE.finditer(text):
        out[match.group(1)] = np.array([float(v) for v in match.group(2).split(",") if v.strip()], dtype=np.float64)

    return out


def params_from_mqh(path: str | Path) -> Mql5HmmParams:
    """Reconstruct ``Mql5HmmParams`` from a generated header."""
    raw = parse_mqh(path)
    k, d = raw["HMM_K"], raw["HMM_D"]
    spec = FeatureSpec(names=tuple(Mql5Feature(n) for n in raw["HMM_FEATURE_NAMES"]), macd_fast=raw["HMM_MACD_FAST"],
                       macd_slow=raw["HMM_MACD_SLOW"], macd_signal=raw["HMM_MACD_SIGNAL"],
                       atr_period=raw["HMM_ATR_PERIOD"])

    return Mql5HmmParams(model_id=raw["HMM_MODEL_ID"], spec=spec, log_pi=raw["HMM_LOG_PI"],
                         log_a=raw["HMM_LOG_A"].reshape(k, k), mu=raw["HMM_MU"].reshape(k, d),
                         sigma_inv=raw["HMM_SIGINV"].reshape(k, d, d), log_norm=raw["HMM_LOGNORM"],
                         feat_mean=raw["HMM_FEAT_MEAN"], feat_sd=raw["HMM_FEAT_SD"])
