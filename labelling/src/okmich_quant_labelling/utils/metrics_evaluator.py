"""
Label Evaluation Toolkit for Market-Regime Labeling

Purpose
-------
Given candidate trend for a price series (e.g., regimes from HMM, clustering, triple-barrier outcomes, trend states), compute a consistent set of metrics to
rank and select trend algorithms for labelling regime classification.

Design
------
- Inputs: pandas DataFrame with datetime index.
  Required columns:
    - 'label' (int or str): candidate regime trend per timestamp.
  Optional columns:
    - 'ret' (float): forward or contemporaneous returns (e.g., r_{t+1}).
    - 'rv' (float): realized volatility measure.
    - Any feature columns (prefixed with 'feat_') for separability and predictive tests.

- Outputs: dict of metrics and a pandas.DataFrame summary for easy ranking.

Metric Families
---------------
1) Class/Temporal Structure
   - n_classes, class_counts, class_entropy
   - imbalance_ratio (max / min class share)
   - transition_matrix, self_transition_rate, mean_dwell, median_dwell
   - state_persistence_stats (detailed dwell statistics by label)

2) Separability (features → trend)
   - silhouette (using feature space), davies_bouldin
   - fisher_ratio (one-vs-rest across features)
   - separability_stats (per-label return stats + Kruskal-Wallis test)

3) Predictive Utility (simple, leakage-aware)
   - Using lagged features (t-1) to predict label at t
   - cross_val_AUC / F1 (binary) or macro-F1 (multiclass)
   - mutual_information( features_{t-1} ; label_t )
   - information_coefficient (Spearman correlation label_t vs ret_{t+1})

4) Economic Coherence
   - label_conditioned_returns: mean, t-stat, SR per class (using ret)
   - var(rv | label) and anova_p (differences in volatility across regimes)
   - lead_lag_corr(label_t, ret_{t+1}) and label_t with rv_{t+1}
   - naive_pnl_by_label (simple PnL calculation per label)

5) Robustness / Stability
   - bootstrap_jaccard under resampling & light noise
   - time_shift_stability (compare label vs label shifted by 1)

6) Coverage & Operational
   - coverage (non-NaN share), label_turnover, avg_label_latency (if provided)
   - compute_time_s (if used via context manager)

7) Composite Score (customizable weights)

Notes
-----
- For binary-vs-multi handling: Many metrics auto-adapt. Predictive utility uses LogisticRegression (binary/multiclass via multinomial) with time-series split.
- No look-ahead: predictive tests use X_{t-1} → y_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway as scipy_f_oneway
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    f1_score,
    roc_auc_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler


@dataclass
class LabelEvalConfig:
    feature_prefix: str = "feat_"
    label_col: str = "label"
    min_class_support: int = 30
    n_splits: int = 5  # TimeSeriesSplit for predictive tests
    random_state: int = 10
    robustness_noise_sigma: float = 0.01
    vol_window: int = 20  # rolling window for realized volatility
    composite_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "structure": 0.15,
            "separability": 0.2,
            "predictive": 0.3,
            "economic": 0.25,
            "robustness": 0.1,
        }
    )


class LabelEvaluator:
    def __init__(self, df: pd.DataFrame, config: Optional[LabelEvalConfig] = None):
        self.df = df.copy()
        if config is None:
            config = LabelEvalConfig()
        self.cfg = config
        if self.cfg.label_col not in self.df.columns:
            raise ValueError(
                f"DataFrame must contain the label column: '{self.cfg.label_col}'"
            )
        self._prep()

    # ------------------ Preparation ------------------
    def _prep(self):
        label_series = self.df[self.cfg.label_col]

        # Cast trend to a compact integer space while preserving mapping
        cat_labels = pd.Categorical(label_series)
        self.label_map = {lab: i for i, lab in enumerate(cat_labels.categories)}
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.df["label_id"] = label_series.map(self.label_map)

        # Compute returns if not present
        if "ret" not in self.df.columns:
            if "close" not in self.df.columns:
                raise ValueError(
                    "Cannot compute returns: no 'ret' or 'close' column found"
                )
            self.df["ret"] = np.log(self.df["close"]).diff()

        # Compute realized volatility if not present
        if "rv" not in self.df.columns:
            self.df["rv"] = self.df["ret"].rolling(self.cfg.vol_window).std()

        # Features
        self.feat_cols = [
            c for c in self.df.columns if c.startswith(self.cfg.feature_prefix)
        ]

        # Drop first row if lagging created NaNs
        # Drop rows with NaNs in required columns
        req = ["label_id"]
        if "ret" in self.df.columns:
            req.append("ret")
        if "rv" in self.df.columns:
            req.append("rv")
        if self.feat_cols:
            req += self.feat_cols
        self.df = self.df.dropna(subset=req).copy()

    # ------------------ Structure Metrics ------------------
    def structure_metrics(self) -> Dict[str, object]:
        y = self.df["label_id"].to_numpy()
        vals, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        class_entropy = -np.sum(probs * np.log(probs + 1e-12))
        imbalance_ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0
        n_classes = len(vals)
        class_counts = {
            self.inv_label_map[int(v)]: int(c) for v, c in zip(vals, counts)
        }

        tm = self._transition_matrix(y, n_classes)
        self_transition_rate = float(np.trace(tm))
        # bring in dwell stats
        dwell_stats = self.dwell_statistics()
        # add state persistence stats
        state_pers_stats = self.state_persistence_stats()

        return {
            "n_classes": int(n_classes),
            "class_counts": class_counts,
            "class_entropy": float(class_entropy),
            "imbalance_ratio": float(imbalance_ratio),
            "transition_matrix": tm,
            "self_transition_rate": self_transition_rate,
            **dwell_stats,
            **state_pers_stats,
        }

    def dwell_statistics(self) -> Dict[str, float]:
        """Calculate comprehensive dwell time statistics"""
        y = self.df["label_id"].to_numpy()
        dwell_times = self._dwell_times(y)

        if len(dwell_times) == 0:
            return {
                "min_dwell": np.nan,
                "max_dwell": np.nan,
                "mean_dwell": np.nan,
                "median_dwell": np.nan,
                "q25_dwell": np.nan,
                "q75_dwell": np.nan,
                "std_dwell": np.nan,
            }

        dwell_array = np.array(dwell_times)
        return {
            "min_dwell": float(np.min(dwell_array)),
            "max_dwell": float(np.max(dwell_array)),
            "mean_dwell": float(np.mean(dwell_array)),
            "median_dwell": float(np.median(dwell_array)),
            "q25_dwell": float(np.percentile(dwell_array, 25)),
            "q75_dwell": float(np.percentile(dwell_array, 75)),
            "std_dwell": float(np.std(dwell_array)),
        }

    def state_persistence_stats(self) -> Dict[str, object]:
        """Detailed state persistence statistics by label"""
        y = self.df["label_id"].to_numpy()
        if len(y) == 0:
            return {"state_persistence_by_label": pd.DataFrame()}

        vals = []
        cur = None
        run_len = 0
        for v in y:
            if cur is None:
                cur = v
                run_len = 1
            elif v == cur:
                run_len += 1
            else:
                vals.append((cur, run_len))
                cur = v
                run_len = 1
        if cur is not None:
            vals.append((cur, run_len))

        if not vals:
            return {"state_persistence_by_label": pd.DataFrame()}

        df = pd.DataFrame(vals, columns=["label_id", "length"])
        # Map back to original trend
        df["label"] = df["label_id"].map(self.inv_label_map)
        out = (
            df.groupby("label")["length"]
            .agg(["count", "mean", "median", "std", "max"])
            .reset_index()
            .sort_values("label")
        )
        return {"state_persistence_by_label": out}

    @staticmethod
    def _transition_matrix(y: np.ndarray, n_classes: int) -> np.ndarray:
        mat = np.zeros((n_classes, n_classes), dtype=float)
        for i in range(len(y) - 1):
            mat[int(y[i]), int(y[i + 1])] += 1
        row_sums = mat.sum(axis=1, keepdims=True) + 1e-12
        return mat / row_sums

    @staticmethod
    def _dwell_times(y: np.ndarray) -> List[int]:
        if len(y) == 0:
            return []
        dwell = []
        run = 1
        for i in range(1, len(y)):
            if y[i] == y[i - 1]:
                run += 1
            else:
                dwell.append(run)
                run = 1
        dwell.append(run)
        return dwell

    # ------------------ Separability Metrics ------------------
    def separability_metrics(self) -> Dict[str, float]:
        out = {}

        # Original separability metrics
        if not self.feat_cols:
            out.update(
                {
                    "silhouette": np.nan,
                    "davies_bouldin": np.nan,
                    "calinski_harabasz": np.nan,
                    "fisher_ratio": np.nan,
                }
            )
        else:
            X = self.df[self.feat_cols].to_numpy()
            y = self.df["label_id"].to_numpy()

            # Guard for empty data after dropna
            if len(X) == 0 or len(y) == 0:
                out.update(
                    {
                        "silhouette": np.nan,
                        "davies_bouldin": np.nan,
                        "calinski_harabasz": np.nan,
                        "fisher_ratio": np.nan,
                    }
                )
            else:
                Xs = StandardScaler().fit_transform(X)
                labels_unique = np.unique(y)
                if len(labels_unique) < 2:
                    out.update(
                        {
                            "silhouette": np.nan,
                            "davies_bouldin": np.nan,
                            "calinski_harabasz": np.nan,
                            "fisher_ratio": np.nan,
                        }
                    )
                else:
                    try:
                        sil = float(silhouette_score(Xs, y))
                    except (ValueError, RuntimeError):
                        sil = np.nan
                    try:
                        db = float(davies_bouldin_score(Xs, y))
                    except (ValueError, RuntimeError):
                        db = np.nan
                    try:
                        ch = float(calinski_harabasz_score(Xs, y))
                    except (ValueError, RuntimeError):
                        ch = np.nan
                    fisher = float(self._fisher_ratio(Xs, y))
                    out.update(
                        {
                            "silhouette": sil,
                            "davies_bouldin": db,
                            "calinski_harabasz": ch,
                            "fisher_ratio": fisher,
                        }
                    )

        # Add separability stats using returns if available (no look-ahead - uses concurrent data for descriptive stats)
        sep_stats = self.separability_stats_with_returns()
        out.update(sep_stats)

        return out

    def separability_stats_with_returns(self) -> Dict[str, object]:
        """Separability statistics using returns - Kruskal-Wallis test and per-label stats"""
        if "ret" not in self.df.columns:
            return {
                "return_separability_stats": pd.DataFrame(),
                "kruskal_wallis_pval": np.nan,
                "kruskal_wallis_stat": np.nan,
            }

        tmp = self.df[["label_id", "ret"]].dropna().copy()
        if tmp.empty:
            return {
                "return_separability_stats": pd.DataFrame(),
                "kruskal_wallis_pval": np.nan,
                "kruskal_wallis_stat": np.nan,
            }

        groups = tmp.groupby("label_id")["ret"].apply(list)
        stats_rows = []
        for label_id, vals in groups.items():
            arr = np.array(vals)
            original_label = self.inv_label_map.get(int(label_id), label_id)
            stats_rows.append(
                {
                    "label": original_label,
                    "label_id": label_id,
                    "n": len(arr),
                    "mean": float(np.nanmean(arr)),
                    "std": float(np.nanstd(arr, ddof=1)),
                }
            )

        try:
            kw = stats.kruskal(*[np.array(v) for v in groups.values])
            pval = float(kw.pvalue)
            stat = float(kw.statistic)
        except (ValueError, TypeError, IndexError):
            pval = float("nan")
            stat = float("nan")

        out_df = pd.DataFrame(stats_rows).sort_values("label")
        return {
            "return_separability_stats": out_df,
            "kruskal_wallis_pval": pval,
            "kruskal_wallis_stat": stat,
        }

    @staticmethod
    def _fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
        classes = np.unique(y)
        if len(classes) < 2:
            return np.nan
        mu = X.mean(axis=0)
        num = 0.0
        den = 0.0
        for k in classes:
            Xk = X[y == k]
            pk = Xk.shape[0] / X.shape[0]
            mk = Xk.mean(axis=0)
            sk2 = Xk.var(axis=0) + 1e-12
            num += pk * ((mk - mu) ** 2)
            den += pk * sk2
        fr = np.mean(num / (den + 1e-12))
        return float(fr)

    # ------------------ Predictive Utility ------------------
    def predictive_metrics(self) -> Dict[str, float]:
        out = {}

        # Original predictive metrics
        if not self.feat_cols:
            out.update({"cv_macro_f1": np.nan, "cv_auc": np.nan, "mutual_info": np.nan})
        else:
            # Create lagged features dynamically
            lag_cols = []
            for c in self.feat_cols:
                lag_col = f"{c}_lag1"
                if lag_col not in self.df.columns:
                    self.df[lag_col] = self.df[c].shift(1)
                lag_cols.append(lag_col)

            # Drop rows with NaNs caused by lagging
            df_clean = self.df.dropna(subset=["label_id"] + lag_cols)
            X = df_clean[lag_cols].to_numpy()
            y = df_clean["label_id"].to_numpy()

            # Guard for single class (after dropna) - LogisticRegression requires at least 2 classes
            labels_unique = np.unique(y)
            if len(labels_unique) < 2:
                out.update(
                    {"cv_macro_f1": np.nan, "cv_auc": np.nan, "mutual_info": np.nan}
                )
            else:
                Xs = StandardScaler().fit_transform(X)

                # Guard for small samples relative to n_splits
                max_splits = min(self.cfg.n_splits, max(len(y) - 1, 0))
                if max_splits < 2:
                    mi = (
                        float(
                            np.mean(mutual_info_classif(Xs, y, discrete_features=False))
                        )
                        if Xs.shape[1] > 0
                        else np.nan
                    )
                    out.update(
                        {"cv_macro_f1": np.nan, "cv_auc": np.nan, "mutual_info": mi}
                    )
                else:
                    # TimeSeriesSplit CV
                    tscv = TimeSeriesSplit(n_splits=max_splits)
                    f1s: List[float] = []
                    aucs: List[float] = []

                    for train, test in tscv.split(Xs):
                        clf = LogisticRegression(
                            max_iter=1000, random_state=self.cfg.random_state
                        )
                        clf.fit(Xs[train], y[train])
                        y_pred = clf.predict(Xs[test])
                        f1s.append(f1_score(y[test], y_pred, average="macro"))

                        try:
                            proba = clf.predict_proba(Xs[test])
                            classes = np.unique(y)
                            if len(classes) == 2:
                                aucs.append(roc_auc_score(y[test], proba[:, 1]))
                            else:
                                aucs.append(
                                    np.mean(
                                        [
                                            roc_auc_score(
                                                (y[test] == c).astype(int), proba[:, i]
                                            )
                                            for i, c in enumerate(classes)
                                        ]
                                    )
                                )
                        except Exception:
                            aucs.append(np.nan)

                    # Mutual information
                    try:
                        mi_vals = mutual_info_classif(
                            Xs,
                            y,
                            discrete_features=False,
                            random_state=self.cfg.random_state,
                        )
                        mi = float(np.mean(mi_vals))
                    except Exception:
                        mi = np.nan

                    out.update(
                        {
                            "cv_macro_f1": float(np.nanmean(f1s)),
                            "cv_auc": float(np.nanmean(aucs)),
                            "mutual_info": mi,
                        }
                    )

        # Add information coefficient (fixed for no look-ahead)
        ic = self.information_coefficient()
        out.update({"information_coefficient": ic})

        return out

    def information_coefficient(self) -> float:
        """Spearman correlation between trend and forward returns (no look-ahead)"""
        if "ret" not in self.df.columns:
            return np.nan

        # Shift returns forward by 1 to avoid look-ahead bias
        df = self.df[["label_id", "ret"]].dropna()
        df_shifted = df.copy()
        df_shifted["ret"] = df_shifted["ret"].shift(-1)  # label_t predicts ret_{t+1}

        # Drop the last row since we shifted returns forward
        df_shifted = df_shifted.iloc[:-1]

        if df_shifted.shape[0] < 10:
            return float("nan")
        return float(df_shifted["label_id"].corr(df_shifted["ret"], method="spearman"))

    # ------------------ Economic Coherence ------------------
    def economic_metrics(self) -> Dict[str, float]:
        out = {
            "sr_overall": np.nan,
            "anova_p_ret": np.nan,
            "anova_p_rv": np.nan,
            "corr_label_future_ret": np.nan,
            "corr_label_future_rv": np.nan,
        }
        y = self.df["label_id"].to_numpy()

        # ---------- Returns ----------
        if "ret" in self.df.columns:
            r = self.df["ret"].to_numpy()
            valid_idx = np.isfinite(r) & np.isfinite(y)
            if valid_idx.sum() > 10:
                r_valid = r[valid_idx]
                y_valid = y[valid_idx]
                mu = float(np.nanmean(r_valid))
                sd = float(np.nanstd(r_valid) + 1e-12)
                out["sr_overall"] = mu / sd
                out["anova_p_ret"] = self._one_way_anova_p(r_valid, y_valid)
                # Shift for proper lead-lag relationship (label_t vs ret_{t+1})
                r_shift = r_valid[1:]
                y_shift = y_valid[:-1]
                if len(r_shift) > 1:
                    out["corr_label_future_ret"] = float(
                        np.corrcoef(y_shift, r_shift)[0, 1]
                    )

        # ---------- Realized volatility ----------
        if "rv" in self.df.columns:
            v = self.df["rv"].to_numpy()
            valid_idx = np.isfinite(v) & np.isfinite(y)
            if valid_idx.sum() > 10:
                v_valid = v[valid_idx]
                y_valid = y[valid_idx]
                out["anova_p_rv"] = self._one_way_anova_p(v_valid, y_valid)
                # Shift for proper lead-lag relationship (label_t vs rv_{t+1})
                v_shift = v_valid[1:]
                y_shift = y_valid[:-1]
                if len(v_shift) > 1:
                    out["corr_label_future_rv"] = float(
                        np.corrcoef(y_shift, v_shift)[0, 1]
                    )

        # Add naive PnL by label (fixed for no look-ahead)
        pnl_stats = self.naive_pnl_by_label()
        out.update(pnl_stats)

        return out

    def naive_pnl_by_label(self) -> Dict[str, object]:
        """Simple PnL calculation by label (no look-ahead: label_t predicts ret_{t+1})"""
        if "ret" not in self.df.columns:
            return {"pnl_by_label": pd.DataFrame()}

        # Create proper lead-lag structure: label_t predicts ret_{t+1}
        df = self.df[["label_id", "ret"]].dropna()
        if len(df) < 2:
            return {"pnl_by_label": pd.DataFrame()}

        # Shift returns forward by 1 (label at time t predicts return at time t+1)
        df_lagged = df.iloc[:-1].copy()  # All but last row for trend
        returns_forward = df["ret"].iloc[1:].values  # Returns shifted forward by 1
        df_lagged["fwd_ret"] = returns_forward

        if df_lagged.empty:
            return {"pnl_by_label": pd.DataFrame()}

        out_rows = []
        for lab_id, grp in df_lagged.groupby("label_id"):
            # Use sign of label as position signal to predict next period return
            s = np.sign(grp["label_id"]).fillna(0.0)
            pnl = (
                (s * grp["fwd_ret"]).sum() if len(grp) > 0 else 0.0
            )  # Sum of pnl, not cumsum
            original_label = self.inv_label_map.get(int(lab_id), lab_id)
            out_rows.append(
                {
                    "label": original_label,
                    "label_id": lab_id,
                    "n": len(grp),
                    "pnl": float(pnl),
                }
            )
        pnl_df = pd.DataFrame(out_rows).sort_values("label")
        return {"pnl_by_label": pnl_df}

    @staticmethod
    def _one_way_anova_p(x: np.ndarray, y: np.ndarray) -> float:
        groups = [x[y == k] for k in np.unique(y)]
        groups = [g[np.isfinite(g)] for g in groups if len(g) > 1]
        if len(groups) < 2:
            return np.nan

        try:
            _, p = scipy_f_oneway(*groups)
            return float(p)
        except (ValueError, TypeError):
            pass

        # Fallback: return an F-like statistic (not a true p-value), for ranking only
        k = len(groups)
        n = sum(len(g) for g in groups)
        grand_mean = np.mean(np.concatenate(groups))
        ssb = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
        ssw = sum(((g - np.mean(g)) ** 2).sum() for g in groups)
        msb = ssb / (max(k - 1, 1))
        msw = ssw / (max(n - k, 1))
        F = msb / (msw + 1e-12)
        # Map F to (0,1] monotonically for ranking
        return float(1.0 / (1.0 + F))

    # ------------------ Robustness ------------------
    def robustness_metrics(
            self, n_boot: int = 20, frac: float = 0.8
    ) -> Dict[str, float]:
        y = self.df["label_id"].to_numpy()
        n = len(y)
        if n < 10:
            return {"bootstrap_jaccard": np.nan, "time_shift_stability": np.nan}

        rng = np.random.default_rng(self.cfg.random_state)
        js = []

        valid_idx = np.where(np.isfinite(y))[0]
        if self.feat_cols:
            X = self.df[self.feat_cols].to_numpy()
            valid_features = np.all(np.isfinite(X), axis=1)
            valid_idx = np.intersect1d(valid_idx, np.where(valid_features)[0])
            if len(valid_idx) == 0:
                return {"bootstrap_jaccard": np.nan, "time_shift_stability": np.nan}
            Xs = StandardScaler().fit_transform(X[valid_idx])
            y_valid = y[valid_idx]
            classes = np.unique(y_valid)

            centroids = {}
            for c in classes:
                Xc = Xs[y_valid == c]
                if len(Xc) == 0:
                    continue
                centroids[c] = Xc.mean(axis=0)

            if len(centroids) < 2:
                boot_jaccard = np.nan
            else:
                C = np.stack([centroids[c] for c in classes], axis=0)
                for _ in range(n_boot):
                    m = max(int(len(y_valid) * frac), 1)
                    idx = np.sort(rng.choice(len(y_valid), m, replace=False))
                    Xb = Xs[idx] + rng.normal(
                        0, self.cfg.robustness_noise_sigma, size=(m, Xs.shape[1])
                    )
                    d2 = ((Xb[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
                    y_hat = classes[np.argmin(d2, axis=1)]
                    js.append(self._jaccard(y_valid[idx], y_hat))
                boot_jaccard = float(np.nanmean(js)) if len(js) else np.nan
        else:
            boot_jaccard = np.nan

        agree = (
            float(np.mean(y[valid_idx][1:] == y[valid_idx][:-1]))
            if len(valid_idx) > 1
            else np.nan
        )
        return {"bootstrap_jaccard": boot_jaccard, "time_shift_stability": agree}

    @staticmethod
    def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
        # For label sequences on same index set, use simple accuracy as Jaccard proxy
        return float(np.mean(a == b))

    # ------------------ Coverage & Ops ------------------
    def ops_metrics(self) -> Dict[str, float]:
        y = self.df["label_id"].to_numpy()
        coverage = float(np.isfinite(y).mean())
        turnover = float(np.mean(y[1:] != y[:-1])) if len(y) > 1 else np.nan
        return {"coverage": coverage, "label_turnover": turnover}

    # ------------------ Composite Score ------------------
    def composite_score(self, m: Dict[str, Dict[str, float]]) -> float:
        def safe_mean(values: list[float]) -> float:
            """Mean of non-NaN values, or 0.0 if all are NaN."""
            valid = [x for x in values if not np.isnan(x)]
            return float(np.mean(valid)) if valid else 0.0

        # Separability
        sep = m["separability"]
        sep_score = safe_mean(
            [
                self._z_pos(sep.get("silhouette")),
                self._z_neg(sep.get("davies_bouldin")),
                self._z_pos(
                    self._rescale_log(sep.get("calinski_harabasz"))
                ),  # CH grows large; log-rescale
                self._z_pos(sep.get("fisher_ratio")),
            ]
        )

        # Predictive
        pred = m["predictive"]
        pred_score = safe_mean(
            [
                self._z_pos(pred.get("cv_macro_f1")),
                self._z_pos(pred.get("cv_auc")),
                self._z_pos(pred.get("mutual_info")),
                self._z_pos(pred.get("information_coefficient")),
            ]
        )

        # Economic
        econ = m["economic"]
        econ_score = safe_mean(
            [
                self._z_pos(econ.get("sr_overall")),
                self._z_neg(econ.get("anova_p_ret")),
                self._z_neg(econ.get("anova_p_rv")),
                self._z_pos(econ.get("corr_label_future_ret")),
                self._z_pos(econ.get("corr_label_future_rv")),
            ]
        )

        # Structure
        struct = m["structure"]
        struct_score = safe_mean(
            [
                self._z_pos(struct.get("self_transition_rate")),
                self._z_pos(struct.get("mean_dwell")),
                self._z_pos(struct.get("class_entropy")),
                self._z_neg(struct.get("imbalance_ratio")),
            ]
        )

        # Robustness
        rob = m["robustness"]
        rob_score = safe_mean(
            [
                self._z_pos(rob.get("bootstrap_jaccard")),
                self._z_pos(rob.get("time_shift_stability")),
            ]
        )

        # Weighted composite
        w = self.cfg.composite_weights
        comp = (
                w["structure"] * struct_score
                + w["separability"] * sep_score
                + w["predictive"] * pred_score
                + w["economic"] * econ_score
                + w["robustness"] * rob_score
        )
        return float(comp)

    @staticmethod
    def _rescale_log(x: Optional[float]) -> Optional[float]:
        if x is None or not np.isfinite(x):
            return x
        return np.log1p(max(x, 0.0))

    @staticmethod
    def _z_pos(x: Optional[float]) -> float:
        # Logistic squash after clipping to a broad range to avoid extreme compression
        if x is None or not np.isfinite(x):
            return np.nan
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _z_neg(x: Optional[float]) -> float:
        if x is None or not np.isfinite(x):
            return np.nan
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(x))  # lower is better

    # ------------------ Master Evaluate ------------------
    def evaluate(self) -> Dict[str, Dict[str, object]]:
        out = {
            "structure": self.structure_metrics(),
            "separability": self.separability_metrics(),
            "predictive": self.predictive_metrics(),
            "economic": self.economic_metrics(),
            "robustness": self.robustness_metrics(),
            "ops": self.ops_metrics(),
        }
        out["composite_score"] = {"score": self.composite_score(out)}
        return out

    def get_evaluation_summary(self) -> pd.DataFrame:
        m = self.evaluate()
        rows = []
        for group, md in m.items():
            if group == "composite_score":
                rows.append(
                    {"metric": "composite_score", self.cfg.label_col: md["score"]}
                )
                continue
            for k, v in md.items():
                # Handle DataFrame values specially
                if isinstance(v, pd.DataFrame):
                    # For DataFrames, we could serialize or skip - here we'll skip for simplicity
                    continue
                rows.append({"metric": f"{group}.{k}", self.cfg.label_col: v})
        return pd.DataFrame(rows)
