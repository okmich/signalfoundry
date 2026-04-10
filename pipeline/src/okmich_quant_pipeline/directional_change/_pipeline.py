"""
DCRegimePipeline — end-to-end DC regime detection pipeline.

Offline (fit):
  1. DC Parser        → parse_dc_events → completed trends with TMV, T, R
  2. Feature extract  → log_r per trend (HMM input)
  3. HMM labelling    → PomegranateHMM (Viterbi) → regime label per trend
  4. Normalisation    → MinMaxScaler fit on training TMV/T
  5. Classifier fit   → any sklearn-compatible classifier on (TMV_norm, T_norm, label)

Live (predict_proba):
  1. dc_live_features → per-bar current TMV / T from last confirmed EXT
  2. Normalise        → frozen training scaler
  3. Classify         → predict_proba → P(Regime1), P(Regime2) per bar

Artifacts (save/load):
  hmm.pkl, scaler.pkl, classifier.pkl, label_map.json

Reference: Chen & Tsang (2021), Chapters 3–6.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from okmich_quant_features.directional_change import dc_live_features, log_r, parse_dc_events
from okmich_quant_ml.hmm import PomegranateHMM

_LABEL_MAP_FILE = "label_map.json"
_HMM_FILE = "hmm.pkl"
_SCALER_FILE = "scaler.pkl"
_CLASSIFIER_FILE = "classifier.pkl"

# Minimum trends required for reliable fitting (book: ≥ 50 total, ≥ 20 per class)
_MIN_TRENDS = 50
_MIN_TRENDS_PER_CLASS = 20
# Maximum fraction of trends allowed to be dropped due to invalid features
_MAX_DROP_FRACTION = 0.05


class DCRegimePipeline:
    """
    End-to-end DC regime detection pipeline.

    The caller is responsible for constructing and configuring the HMM and classifier. The pipeline orchestrates fitting,
    normalisation, and artifact persistence. It is classifier-agnostic: any sklearn-compatible estimator that exposes
    fit / predict / predict_proba works.

    Parameters
    ----------
    theta : float
        DC threshold as a decimal fraction (e.g. 0.003 for 0.3%).
        Used for both offline trend extraction and live feature computation.
    hmm : PomegranateHMM
        Unfitted HMM instance. Must be configured for 2 states and
        InferenceMode.VITERBI for correct Viterbi-path label generation.
    classifier : sklearn-compatible estimator
        Unfitted classifier. Must expose fit(X, y), predict(X),
        predict_proba(X) → shape (n, 2).
    alpha : float, optional
        Attenuation coefficient for asymmetric downward DC threshold (Hu et al. 2022).
        Downward DC confirmed when price falls alpha*theta from the last peak.
        alpha=1.0 (default) → symmetric thresholds matching the original book.

    Examples
    --------
    >>> from okmich_quant_ml.hmm import PomegranateHMM, DistType, InferenceMode, create_simple_hmm_instance
    >>> from sklearn.naive_bayes import GaussianNB
    >>> hmm = create_simple_hmm_instance(DistType.NORMAL, n_states=2, inference_mode=InferenceMode.VITERBI)
    >>> pipeline = DCRegimePipeline(theta=0.003, hmm=hmm, classifier=GaussianNB())
    >>> pipeline.fit(prices_train)
    >>> pipeline.save("artifacts/v1/")
    """

    def __init__(self, theta: float, hmm: PomegranateHMM, classifier: Any, alpha: float = 1.0):
        self.theta = theta
        self.alpha = alpha
        self.hmm = hmm
        self.classifier = classifier
        self._scaler: MinMaxScaler | None = None
        self._label_map: dict | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Offline pipeline
    # ------------------------------------------------------------------

    def fit(self, prices: pd.Series) -> "DCRegimePipeline":
        """
        Fit the full pipeline on a training price series.

        Runs DC parsing → log(R) extraction → HMM labelling → MinMaxScaler fitting → classifier training.

        Parameters
        ----------
        prices : pd.Series
            Close price series for the training period, in chronological order.

        Returns
        -------
        self
        """
        # --- Stage 1: DC parsing ---
        trends = parse_dc_events(prices, self.theta, self.alpha)

        if len(trends) < _MIN_TRENDS:
            raise ValueError(f"Only {len(trends)} DC trends found — need >= {_MIN_TRENDS}. "
                             f"Lower theta or extend the training window.")

        # --- Stage 2: Feature extraction + quality filter ---
        lr = log_r(trends)
        valid = lr.notna() & np.isfinite(lr.values) & (trends["t"].values >= 2)
        n_dropped = int((~valid).sum())

        if n_dropped / len(trends) > _MAX_DROP_FRACTION:
            raise ValueError(f"{n_dropped}/{len(trends)} trends dropped due to invalid features "
                             f"(>{_MAX_DROP_FRACTION:.0%}). Check DC parser output.")

        trends = trends.loc[valid].reset_index(drop=True)
        lr_clean = lr.loc[valid].reset_index(drop=True)

        # --- Stage 3: HMM fitting and Viterbi label generation ---
        X_hmm = lr_clean.values.reshape(-1, 1)
        raw_labels = self.hmm.fit_predict(X_hmm)

        # Remap: Regime 1 = lower mean log(R) state, Regime 2 = higher mean log(R) state
        params = self.hmm.parameters
        state_means = np.array([p["means"][0] for p in params])
        regime1_state = int(np.argmin(state_means))
        regime2_state = int(np.argmax(state_means))
        labels = np.where(raw_labels == regime1_state, 1, 2)

        # Sanity checks
        r2_mean = trends.loc[labels == 2, "r"].mean()
        r1_mean = trends.loc[labels == 1, "r"].mean()
        if r2_mean <= r1_mean:
            warnings.warn("Regime 2 mean R is not higher than Regime 1 — check HMM state assignment.")

        regime2_pct = float((labels == 2).mean())
        if regime2_pct < 0.05 or regime2_pct > 0.50:
            warnings.warn(f"Regime 2 is {regime2_pct:.1%} of training data — "
                          "verify the training period covers a volatility event.")

        n_r1 = int((labels == 1).sum())
        n_r2 = int((labels == 2).sum())
        if n_r1 < _MIN_TRENDS_PER_CLASS or n_r2 < _MIN_TRENDS_PER_CLASS:
            warnings.warn(f"Regime class counts: R1={n_r1}, R2={n_r2}. "
                          f"Need >= {_MIN_TRENDS_PER_CLASS} per class for reliable classifier.")

        self._label_map = {
            "regime1_state": regime1_state,
            "regime2_state": regime2_state,
            "regime1_mean_log_r": float(state_means[regime1_state]),
            "regime2_mean_log_r": float(state_means[regime2_state]),
            "n_trends": len(trends),
            "n_regime1": n_r1,
            "n_regime2": n_r2,
            "regime2_pct": regime2_pct,
            "theta": self.theta,
            "alpha": self.alpha,
        }

        # --- Stage 4: Normalisation — fit on training TMV and T only ---
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        X_raw = trends[["tmv", "t"]].values.astype(np.float64)
        X_norm = self._scaler.fit_transform(X_raw)

        # --- Stage 5: Classifier training ---
        # sklearn convention: 0 = Regime 1 (normal), 1 = Regime 2 (abnormal)
        y = (labels == 2).astype(int)
        self.classifier.fit(X_norm, y)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Live inference
    # ------------------------------------------------------------------

    def predict_proba(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute per-bar regime probabilities on a price series.

        Uses dc_live_features (no look-ahead) and the frozen training scaler.
        Bars before the first DC event is confirmed are returned as NaN.
        Live TMV/T values outside the training range are clipped to [0, 1].

        Parameters
        ----------
        prices : pd.Series
            Close price series (training or test/live).

        Returns
        -------
        pd.DataFrame
            Same index as prices, columns:
            - p_regime1    : P(Regime 1 | TMV_current, T_current)
            - p_regime2    : P(Regime 2 | TMV_current, T_current)
            - direction    : +1.0 (uptrend) or -1.0 (downtrend), NaN before first DC
            - tmv_current  : raw current TMV
            - t_current    : raw current T (bars)
            - upward_dcc   : True on bar an upward DC is confirmed (price rose θ from trough)
            - downward_dcc : True on bar a downward DC is confirmed (price fell α×θ from peak)
        """
        self._check_fitted()

        live = dc_live_features(prices, self.theta, self.alpha)
        has_dc = live["direction"].notna()

        result = pd.DataFrame({
            "p_regime1": np.nan,
            "p_regime2": np.nan,
            "direction": live["direction"],
            "tmv_current": live["tmv_current"],
            "t_current": live["t_current"],
            "upward_dcc": live["upward_dcc"],
            "downward_dcc": live["downward_dcc"],
        }, index=prices.index)

        if not has_dc.any():
            return result

        X_raw = live.loc[has_dc, ["tmv_current", "t_current"]].values.astype(np.float64)
        X_transformed = self._scaler.transform(X_raw)

        # Warn only when raw values are genuinely outside the training range
        out_of_range = (X_transformed < 0.0) | (X_transformed > 1.0)
        if out_of_range.any():
            warnings.warn("Live TMV/T outside training range — clipping to [0, 1].")
        X_norm = np.clip(X_transformed, 0.0, 1.0)

        proba = self.classifier.predict_proba(X_norm)  # shape (n, 2): [P(R1), P(R2)]
        result.loc[has_dc, "p_regime1"] = proba[:, 0]
        result.loc[has_dc, "p_regime2"] = proba[:, 1]

        return result

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Save all fitted artifacts to a directory.

        Saves: hmm.pkl, scaler.pkl, classifier.pkl, label_map.json.

        Parameters
        ----------
        path : str or Path
            Directory to save artifacts. Created if it does not exist.
        """
        self._check_fitted()
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.hmm, out / _HMM_FILE)
        joblib.dump(self._scaler, out / _SCALER_FILE)
        joblib.dump(self.classifier, out / _CLASSIFIER_FILE)
        (out / _LABEL_MAP_FILE).write_text(json.dumps(self._label_map, indent=2))

    @classmethod
    def load(cls, path: str | Path, classifier: Any = None) -> "DCRegimePipeline":
        """
        Load a previously saved pipeline from a directory.

        Parameters
        ----------
        path : str or Path
            Directory containing hmm.pkl, scaler.pkl, classifier.pkl, label_map.json.
        classifier : sklearn-compatible estimator, optional
            If provided, overrides the loaded classifier.pkl (useful for swapping classifiers).

        Returns
        -------
        DCRegimePipeline
            Fully fitted pipeline ready for predict_proba.
        """
        src = Path(path)
        label_map = json.loads((src / _LABEL_MAP_FILE).read_text())
        hmm = joblib.load(src / _HMM_FILE)
        scaler = joblib.load(src / _SCALER_FILE)
        loaded_clf = joblib.load(src / _CLASSIFIER_FILE)

        instance = cls.__new__(cls)
        instance.theta = label_map["theta"]
        instance.alpha = label_map.get("alpha", 1.0)  # backwards compatible with older artifacts
        instance.hmm = hmm
        instance.classifier = classifier if classifier is not None else loaded_clf
        instance._scaler = scaler
        instance._label_map = label_map
        instance._is_fitted = True
        return instance

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def label_map(self) -> dict:
        self._check_fitted()
        return dict(self._label_map)

    @property
    def scaler_stats(self) -> dict:
        self._check_fitted()
        return {
            "tmv_min": float(self._scaler.data_min_[0]),
            "tmv_max": float(self._scaler.data_max_[0]),
            "t_min": float(self._scaler.data_min_[1]),
            "t_max": float(self._scaler.data_max_[1]),
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Pipeline is not fitted. Call fit() first.")

    def __repr__(self) -> str:
        state = "fitted" if self._is_fitted else "unfitted"
        return (f"DCRegimePipeline("
                f"theta={self.theta}, "
                f"alpha={self.alpha}, "
                f"hmm={self.hmm!r}, "
                f"classifier={type(self.classifier).__name__}, "
                f"state={state})")
