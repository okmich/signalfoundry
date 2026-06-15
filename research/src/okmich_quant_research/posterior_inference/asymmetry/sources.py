"""Posterior-stream sources for the asymmetry confirmer.

A *source* turns a model/data spec into the causal ``PosteriorStream`` the judge (``validate_stream``) consumes. Sources
are the only place that fit/load HMMs; the judge is source-agnostic. All streams here are **pure filtering** (causal) —
fixed-lag / smoothing is retired; the smoothed HMM is a teacher for labels, not a live signal.

  * ``walk_forward_filtered_posteriors`` — rolling/anchored refit of a frozen config, causal-filter each OOS segment,
    stitch into one stream with ``fold_ids`` for the per-fold stability test. (Promoted from the lab walk-forward script.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from okmich_quant_ml.hmm import DistType, InferenceMode, create_simple_hmm_instance
from okmich_quant_labelling.utils.label_util import map_regime_to_volatility_score

from .forward_axes import MarketAxis
from .validation import PosteriorStream


@dataclass(frozen=True)
class HmmFitSpec:
    """The frozen HMM configuration re-fit on each fold (only parameters are re-estimated, never K/variant/features)."""

    dist_type: DistType
    n_states: int = 3
    n_components: int | None = None
    is_mixture: bool = False
    random_state: int = 100
    max_iter: int = 100
    covariance_type: str = "diag"

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError(f"HmmFitSpec.n_states must be >= 2, got {self.n_states}.")
        if self.max_iter < 1:
            raise ValueError(f"HmmFitSpec.max_iter must be >= 1, got {self.max_iter}.")
        if self.is_mixture and (self.n_components is None or self.n_components < 2):
            raise ValueError(f"HmmFitSpec.is_mixture requires n_components >= 2 (a 1-component mixture is degenerate "
                             f"and rejected by create_simple_hmm_instance), got {self.n_components}.")


@dataclass(frozen=True)
class WalkForwardWindow:
    """Walk-forward geometry. ``step`` defaults to ``oos`` (contiguous, non-overlapping OOS — the stitchable case)."""

    train: int
    oos: int
    step: int | None = None
    lead_in: int = 2000
    anchored: bool = False

    def __post_init__(self) -> None:
        if self.train < 1:
            raise ValueError(f"WalkForwardWindow.train must be >= 1, got {self.train}.")
        if self.oos < 1:
            raise ValueError(f"WalkForwardWindow.oos must be >= 1, got {self.oos}.")
        if self.lead_in < 0:
            raise ValueError(f"WalkForwardWindow.lead_in must be >= 0, got {self.lead_in}.")
        if self.step is not None:
            if self.step < 1:
                raise ValueError(f"WalkForwardWindow.step must be >= 1, got {self.step}.")
            if self.step < self.oos:
                raise ValueError(f"WalkForwardWindow.step ({self.step}) < oos ({self.oos}) would overlap OOS "
                                 f"segments and double-count stitched rows.")

    def resolved_step(self) -> int:
        return self.oos if self.step is None else self.step


def _folds(n: int, window: WalkForwardWindow) -> list[tuple[int, int, int]]:
    """Yield ``(train_lo, oos_lo, oos_hi)`` per fold. Rolling train unless ``anchored`` (then train_lo=0, expanding).

    Window geometry (positivity, ``step >= oos``) is validated in ``WalkForwardWindow.__post_init__``, so ``oos_lo``
    always advances here.
    """
    step = window.resolved_step()
    folds: list[tuple[int, int, int]] = []
    oos_lo = window.train
    while oos_lo + window.oos <= n:
        train_lo = 0 if window.anchored else oos_lo - window.train
        folds.append((train_lo, oos_lo, oos_lo + window.oos))
        oos_lo += step
    return folds


def _vol_rank_order(train_close: np.ndarray, train_map: np.ndarray, n_states: int) -> list[int]:
    """Order raw state ids ascending by in-train realised volatility (|1-bar log return| median per state).

    Raises if the fold's state structure is degenerate — a state collapsed (never assigned in-train) or had too few
    observations to receive a distinct volatility bucket — so the ordering would be arbitrary. Failing loud here keeps a
    mislabeled column from silently corrupting the stitched stream's cross-fold state identity (and the stability verdict);
    in the funnel this surfaces as the candidate's ``error``.
    """
    abs_ret = np.abs(np.diff(np.log(train_close), prepend=np.nan))
    bucket = map_regime_to_volatility_score(pd.DataFrame({"regime": train_map, "abs_ret": abs_ret}),
                                            regime_col="regime", vol_proxy_col="abs_ret", method="median")
    buckets = [bucket.get(raw) for raw in range(n_states)]
    if any(b is None for b in buckets) or sorted(buckets) != list(range(n_states)):
        raise ValueError(
            f"degenerate fold: vol-rank could not assign a distinct volatility bucket to each of the {n_states} states "
            f"(a state collapsed or had too few observations). state->bucket={bucket}."
        )
    return sorted(range(n_states), key=lambda raw: bucket[raw])


def walk_forward_filtered_posteriors(data: pd.DataFrame, *, feature_columns: list[str], fit: HmmFitSpec,
                                     window: WalkForwardWindow, identity_axis: MarketAxis = MarketAxis.VOLATILITY,
                                     state_names: list[str] | None = None) -> PosteriorStream:
    """Walk-forward causal filtering posteriors as one stitched ``PosteriorStream``.

    Each fold refits ``fit`` on its train window, runs the *frozen* model in ``FILTERING`` mode over the OOS segment with
    a dropped lead-in burn-in, and ranks states by in-train volatility for a stable cross-fold identity (low→high). OOS
    segments are non-overlapping (``step >= oos``); the stream carries ``fold_ids`` for the validator's stability test.

    ``data`` needs a ``close`` column (for the vol-rank identity) and a time index (surfaced as ``PosteriorStream.index``).
    Feature warm-up NaNs must be trimmed before the first fold's fed range — non-finite fed features raise rather than
    silently producing NaN posteriors.
    """
    if identity_axis != MarketAxis.VOLATILITY:
        raise NotImplementedError(
            f"state identity for axis {identity_axis} not wired yet; only VOLATILITY (vol-rank) is supported. "
            f"Add the matching label_util mapper when another identity axis is needed."
        )
    if "close" not in data.columns:
        raise ValueError("walk_forward_filtered_posteriors: data must contain a 'close' column for vol-rank identity.")
    missing = [c for c in feature_columns if c not in data.columns]
    if missing:
        raise ValueError(f"walk_forward_filtered_posteriors: feature columns missing from data: {missing}")

    folds = _folds(len(data), window)
    if not folds:
        raise ValueError(f"walk_forward_filtered_posteriors: no folds fit train={window.train}+oos={window.oos} into {len(data)} rows.")

    feat = data[feature_columns].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    fold_probs, fold_positions, fold_ids = [], [], []

    for f, (train_lo, oos_lo, oos_hi) in enumerate(folds):
        x_train_raw = feat[train_lo:oos_lo]
        finite = np.isfinite(x_train_raw).all(axis=1)
        scaler = StandardScaler().fit(x_train_raw[finite])
        x_train_scaled = scaler.transform(x_train_raw[finite])

        model = create_simple_hmm_instance(dist_type=fit.dist_type, n_states=fit.n_states, n_components=fit.n_components,
                                           is_mixture_model=fit.is_mixture, random_state=fit.random_state,
                                           max_iter=fit.max_iter, inference_mode=InferenceMode.FILTERING,
                                           covariance_type=fit.covariance_type)
        model.fit(x_train_scaled)
        train_map = np.argmax(np.asarray(model.predict_proba(x_train_scaled)), axis=1)
        order = _vol_rank_order(close[train_lo:oos_lo][finite], train_map, fit.n_states)

        lead = max(0, oos_lo - window.lead_in)
        x_fed = scaler.transform(feat[lead:oos_hi])
        if not np.isfinite(x_fed).all():
            raise ValueError(f"walk_forward_filtered_posteriors: non-finite features in fold {f} fed range "
                             f"[{lead}:{oos_hi}] — trim feature warm-up before the first fold.")
        probs = np.asarray(model.predict_proba(x_fed), dtype=float)[oos_lo - lead:]
        fold_probs.append(probs[:, order])
        fold_positions.append(np.arange(oos_lo, oos_hi))
        fold_ids.append(np.full(oos_hi - oos_lo, f, dtype=int))

    positions = np.concatenate(fold_positions)
    names = state_names if state_names is not None else [f"vol_{i}" for i in range(fit.n_states)]
    return PosteriorStream(probs=np.vstack(fold_probs), state_names=names, index=data.index[positions],
                           fold_ids=np.concatenate(fold_ids))


def frozen_artifact_posteriors(artifact_dir: str, data: pd.DataFrame, *, feature_columns: list[str] | None = None,
                               oos_window: tuple[str, str] | None = None, lead_in: int = 2000,
                               state_names: list[str] | None = None) -> PosteriorStream:
    """Causal filtering posteriors from a frozen 3-file HMM artifact over its (or a given) OOS window — single fold.

    The 'validation-only' source: no refit, no screener. Loads the artifact via ``HmmModelWrapper``, forces ``FILTERING``,
    feeds ``[lead-in + OOS]`` of the shipped feature columns through the persisted transform pipeline, drops the warm-up,
    and labels states by the artifact's **shipped ``state_mapping``** (so we audit exactly what the live system would see).

    ``data`` must contain the feature columns and a time index whose tz matches the window timestamps. ``oos_window``
    defaults to the artifact metadata's window; ``feature_columns`` to the metadata's ``feature_columns``.
    """
    if lead_in < 0:
        raise ValueError(f"frozen_artifact_posteriors: lead_in must be >= 0, got {lead_in}.")
    from okmich_quant_ml.inference_model_wrappers import HmmModelWrapper

    wrapper = HmmModelWrapper({"type": "hmm", "model_path": str(artifact_dir)})
    wrapper.model.inference_mode = InferenceMode.FILTERING
    meta = wrapper.metadata
    cols = list(feature_columns) if feature_columns is not None else list(meta["feature_columns"])
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ValueError(f"frozen_artifact_posteriors: feature columns missing from data: {missing}")

    if oos_window is not None:
        start, end = pd.Timestamp(oos_window[0]), pd.Timestamp(oos_window[1])
    elif "oos_window" in meta:
        start, end = pd.Timestamp(meta["oos_window"]["start_ts"]), pd.Timestamp(meta["oos_window"]["end_ts"])
    else:
        start, end = data.index[0], data.index[-1]

    oos_pos = np.flatnonzero((data.index >= start) & (data.index <= end))
    if oos_pos.size == 0:
        raise ValueError(f"frozen_artifact_posteriors: no rows in oos_window [{start}, {end}].")
    first, last = int(oos_pos[0]), int(oos_pos[-1])

    lead_start = max(0, first - lead_in)
    probs, _ = wrapper.predict(data.iloc[lead_start:last + 1][cols])
    probs = np.asarray(probs, dtype=float)[first - lead_start:]

    if state_names is None:
        mapping = wrapper.state_mapping
        state_names = [str(mapping.get(i, i)) for i in range(probs.shape[1])]
    return PosteriorStream(probs=probs, state_names=state_names, index=data.index[first:last + 1], fold_ids=None)
