from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ----------------------------- Utilities -----------------------------
def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure df is indexed by DatetimeIndex (or convert from 'Date' column)."""
    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()
    if "Date" in df.columns:
        out = df.copy()
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.set_index("Date").sort_index()
        return out
    raise ValueError("DataFrame must have a DatetimeIndex or a 'Date' column.")


# ----------------------------- Returns -----------------------------


def prepare_returns(
        df: pd.DataFrame,
        price_col: str = "close",
        log_returns: bool = True,
        horizons: List[int] = [10],
) -> pd.DataFrame:
    """
    Add 'ret' (1-step) and forward returns 'fwd_ret_H' for each horizon H.
    """
    df = _ensure_datetime_index(df.copy())
    px = df[price_col].astype(float)

    if log_returns:
        df["ret"] = np.log(px).diff()
        for h in horizons:
            df[f"fwd_ret_{h}"] = np.log(px.shift(-h) / px)
    else:
        df["ret"] = px.pct_change()
        for h in horizons:
            df[f"fwd_ret_{h}"] = px.shift(-h).div(px).sub(1.0)

    return df


# ----------------------------- Evaluation -----------------------------
@dataclass
class LabelEvalResult:
    label: str
    horizon: int
    corr: float
    mi: float
    auc: float
    pnl: float
    coverage: float


def _naive_pnl(label: pd.Series, fwd_ret: pd.Series) -> Tuple[float, float]:
    # Align the series by index
    combined = pd.DataFrame({"label": label, "fwd_ret": fwd_ret}).dropna()
    if len(combined) == 0:
        return 0.0, 0.0

    s = np.sign(
        combined["label"].replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0.0)
    )
    coverage = float((s != 0).mean())
    pnl = (s * fwd_ret).replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0.0).cumsum()

    return float(pnl.iloc[-1]) if len(pnl) else 0.0, coverage


def _mutual_info(label: pd.Series, fwd_ret: pd.Series) -> float:
    """
    Mutual information between t label and t forward return to avoid look-ahead bias.
    """
    combined = pd.DataFrame({"label": label, "fwd_ret": fwd_ret}).dropna()
    if len(combined) < 10:
        return float("nan")

    y = combined["fwd_ret"].replace({np.inf: np.nan, -np.inf: np.nan}).dropna()
    if len(y) < 10:
        return float("nan")

    # Use previous bar's label
    x = label.reindex(y.index).replace({np.inf: np.nan, -np.inf: np.nan}).fillna(0.0)
    if len(y) < 100:
        return float("nan")

    mi = mutual_info_regression(
        x.values.reshape(-1, 1), y.values, discrete_features=False, random_state=42
    )
    return float(mi[0])


def _auc_walkforward(
        label: pd.Series, fwd_ret: pd.Series, walk_len: int = 1000, test_len: int = 250
) -> float:
    """
    Walk-forward AUC using t label values to predict t forward returns.
    """

    combined = pd.DataFrame({"label": label, "fwd_ret": fwd_ret}).dropna()
    if len(combined) < 10:
        return float("nan")

    y_full = (combined["fwd_ret"] > 0).astype(int)
    x_full = (
        combined["label"]
        .replace({np.inf: np.nan, -np.inf: np.nan})
        .fillna(0.0)
        .values.reshape(-1, 1)
    )

    n = len(y_full)
    if n < (walk_len + test_len + 50):
        return float("nan")

    aucs: List[float] = []
    start = 0
    while start + walk_len + test_len <= n:
        tr = slice(start, start + walk_len)
        te = slice(start + walk_len, start + walk_len + test_len)

        Xtr, ytr = x_full[tr], y_full.values[tr]
        Xte, yte = x_full[te], y_full.values[te]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr_s, ytr)
        proba = clf.predict_proba(Xte_s)[:, 1]
        try:
            aucs.append(roc_auc_score(yte, proba))
        except ValueError:
            pass

        start += test_len

    return float(np.nanmean(aucs)) if len(aucs) else float("nan")


def evaluate_labels(
        df: pd.DataFrame,
        label_cols: List[str],
        return_col: str,
        walkforward: Dict[str, int] = {"train": 1000, "test": 250},
) -> pd.DataFrame:
    """
    Evaluate label predictiveness: corr, MI, walk-forward AUC, PnL, coverage.
    Assumes `fwd_ret` is already a forward return.
    """
    if return_col not in df.columns:
        raise ValueError(f"{return_col} not found. Run prepare_returns first.")

    try:
        horizon = int(return_col.split("_")[-1])
    except Exception:
        horizon = np.nan

    fwd_ret = df[return_col]
    rows: List[LabelEvalResult] = []

    for col in label_cols:
        if col not in df.columns:
            continue
        lbl = df[col]

        corr = float(lbl.corr(fwd_ret))
        mi = _mutual_info(lbl, fwd_ret)
        auc = _auc_walkforward(
            lbl, fwd_ret, walk_len=walkforward["train"], test_len=walkforward["test"]
        )
        pnl, cov = _naive_pnl(lbl, fwd_ret)

        rows.append(LabelEvalResult(col, horizon, corr, mi, auc, pnl, cov))

    out = pd.DataFrame([r.__dict__ for r in rows])
    return out.sort_values(by=["auc", "mi", "corr"], ascending=False)


def classify_labels(
        eval_table: pd.DataFrame, auc_thresh: float = 0.62, corr_thresh: float = 0.08
) -> Dict[str, List[str]]:
    """Split trend into predictive vs descriptive by thresholds."""
    tbl = eval_table.copy()
    tbl["abs_corr"] = tbl["corr"].abs()
    mask = (tbl["auc"] >= auc_thresh) & (tbl["abs_corr"] >= corr_thresh)
    return {
        "predictive": tbl.loc[mask, "label"].tolist(),
        "descriptive": tbl.loc[~mask, "label"].tolist(),
    }


def classify_label(data, label_col, horizon=1):
    results = {}

    df = data.copy()

    returns_lbl = f"fwd_ret_{horizon}"
    px = df["close"].astype(float)
    df[returns_lbl] = np.log(px).shift(-horizon) - np.log(px)
    # Drop rows where either the label or the forward-return is missing; preserve both columns.
    df = df.dropna(subset=[label_col, returns_lbl])

    # 1. Correlation
    results["corr"] = df[label_col].corr(df[returns_lbl])
    # 2. Mutual Information
    results["mi"] = _mutual_info(df[label_col], df[returns_lbl])
    # 3. Predictive AUC
    results["auc"] = _auc_walkforward(df[label_col], df[returns_lbl])
    # 4. Naive Strategy PnL — unpack scalar pnl and coverage separately
    results["pnl"], results["coverage"] = _naive_pnl(df[label_col], df[returns_lbl])

    # Classification
    if results["auc"] > 0.65 and results["corr"] > 0.1:
        results["type"] = "Predictive"
    else:
        results["type"] = "Descriptive"
    return results


# ----------------------------- Meta-signal -----------------------------
def _walk_slices(n: int, train: int, test: int):
    """
    Generator that creates train/test slice indices for walk-forward analysis

    :param n:
    :param train:
    :param test:
    :return:
    """
    start = 0
    while start + train + test <= n:
        yield slice(start, start + train), slice(start + train, start + train + test)
        start += test


def build_meta_signal(
        df: pd.DataFrame,
        predictive_cols: List[str],
        method: str = "logit",
        target_col: str = "fwd_ret_10",
        walkforward: Dict[str, int] = {"train": 1000, "test": 250},
) -> pd.Series:
    """
    Creates an ensemble signal from multiple predictive trend.
    Create an out-of-sample meta signal from predictive trend.

    - 'logit': logistic regression on all predictive trend
    - 'weighted': 1D AUC weights on each label, sign of weighted score - Weighted combination based on individual AUC scores

    Output is in {-1, 0, +1} via 0.5 threshold for logit / sign(score) for weighted.
    """
    if not predictive_cols:
        return pd.Series(index=df.index, dtype=float, name="meta_signal")

    y_full = df[target_col].copy()
    X_full = (
        df[predictive_cols]
        .replace({np.inf: np.nan, -np.inf: np.nan})
        .fillna(0.0)
        .values
    )
    n = len(df)
    signal = np.zeros(n, dtype=float)

    for tr, te in _walk_slices(n, walkforward["train"], walkforward["test"]):
        ytr = (y_full.iloc[tr] > 0).astype(int).values
        Xtr = X_full[tr]
        Xte = X_full[te]

        if method == "logit":
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            clf = LogisticRegression(max_iter=1000)
            clf.fit(Xtr_s, ytr)
            proba = clf.predict_proba(Xte_s)[:, 1]
            s = np.sign(proba - 0.5)
        elif method == "weighted":
            aucs = []
            for j in range(Xtr.shape[1]):
                try:
                    p = (
                        LogisticRegression(max_iter=1000)
                        .fit(Xtr[:, [j]], ytr)
                        .predict_proba(Xtr[:, [j]])[:, 1]
                    )
                    aucs.append(roc_auc_score(ytr, p))
                except Exception:
                    aucs.append(0.5)
            w = np.array(aucs, dtype=float)
            w = np.clip(w - 0.5, 0, None)  # keep only positive edge
            if w.sum() == 0:
                s = np.zeros(len(Xte))
            else:
                score = (Xte * w).sum(axis=1) / (np.abs(w).sum() + 1e-9)
                s = np.sign(score)
        else:
            raise ValueError("method must be 'logit' or 'weighted'")
        signal[te] = s

    return pd.Series(signal, index=df.index, name="meta_signal")


# ----------------------------- Filters -----------------------------
def apply_filters(
        signal: pd.Series, df: pd.DataFrame, filters: Dict[str, Dict]
) -> pd.Series:
    """
    Gate a signal by descriptive filters, i.e. Applies descriptive filters to gate quant signals.
    Supports categorical filters (e.g., market regimes) and threshold filters (e.g., volatility bounds).
    Sets signal to 0 when filters are not met

    filters spec example:
      {'vol_regime': {'type':'categorical','allow_values':[0]},
       'ATR': {'type':'threshold','min':0.001,'max':0.01}}
    """
    sig = signal.copy().astype(float)
    mask = pd.Series(True, index=df.index)
    for col, spec in (filters or {}).items():
        if col not in df.columns:
            continue
        if spec.get("type") == "categorical":
            allow = set(spec.get("allow_values", []))
            mask &= df[col].isin(allow)
        elif spec.get("type") == "threshold":
            mn, mx = spec.get("min", -np.inf), spec.get("max", np.inf)
            mask &= df[col].between(mn, mx)
    sig[~mask] = 0.0
    return sig


# ----------------------------- Backtest -----------------------------
@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    stats: Dict[str, float]


def backtest_walkforward(
        df: pd.DataFrame,
        signal_col: str,
        return_col: str = "ret",
        fees_bps: float = 0.0,
        slippage_bps: float = 0.0,
        size: float = 1.0,
) -> BacktestResult:
    """
    Vectorized next-bar execution:
      position = signal.shift(1)
      strat_ret = position * returns - transaction_costs_on_changes
    """
    s = df[signal_col].fillna(0.0).astype(float)
    r = df[return_col].fillna(0.0).astype(float)

    pos = s.shift(1).fillna(0.0)
    dpos = pos.diff().fillna(pos)
    cost_per_change = (fees_bps + slippage_bps) / 1e4
    costs = np.abs(dpos) * cost_per_change

    strat_ret = pos * r * size - costs
    equity = (1.0 + strat_ret).cumprod()
    trades = pd.DataFrame(
        {
            "position": pos,
            "ret": r,
            "costs": costs,
            "strat_ret": strat_ret,
            "equity": equity,
        },
        index=df.index,
    )

    # Stats
    def _cagr(eq: pd.Series) -> float:
        if len(eq) < 2:
            return float("nan")
        if isinstance(eq.index, pd.DatetimeIndex):
            years = (eq.index[-1] - eq.index[0]).days / 365.25
        else:
            years = len(eq) / 252.0
        return float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else float("nan")

    def _maxdd(eq: pd.Series) -> float:
        roll = eq.cummax()
        dd = (eq / roll) - 1.0
        return float(dd.min())

    def _sharpe(x: pd.Series, ann: float = 252.0) -> float:
        mu = x.mean() * ann
        sd = x.std(ddof=1) * (ann ** 0.5)
        return float(mu / sd) if sd > 1e-12 else float("nan")

    stats = {
        "CAGR": _cagr(equity),
        "MaxDD": _maxdd(equity),
        "Sharpe": _sharpe(strat_ret),
        "AvgDailyRet": float(strat_ret.mean()),
        "StdDailyRet": float(strat_ret.std(ddof=1)),
        "Turnover": float(np.abs(dpos).mean()),
    }

    return BacktestResult(equity=equity, trades=trades, stats=stats)


def summarize(bt: BacktestResult) -> pd.DataFrame:
    """Small helper to pretty-print backtest stats as a one-row DataFrame."""
    return pd.DataFrame([bt.stats])


# ----------------------------- Auto pipeline -----------------------------
def auto_pipeline(
        df: pd.DataFrame,
        price_col: str = "close",
        label_cols: Optional[List[str]] = None,
        horizon: int = 10,
        walkforward: Dict[str, int] = {"train": 1000, "test": 250},
        auc_thresh: float = 0.62,
        corr_thresh: float = 0.08,
        meta_method: str = "logit",
        filters: Optional[Dict[str, Dict]] = None,
        fees_bps: float = 0.0,
        slippage_bps: float = 0.0,
) -> Dict[str, object]:
    """
    One-call pipeline:
      1) compute forward return at `horizon`
      2) evaluate each label
      3) classify predictive vs descriptive
      4) build meta-signal on predictive trend (walk-forward)
      5) optional: apply filters
      6) backtest & stats
    Returns dict with: df, evaluation, classes, backtest
    """
    df = prepare_returns(df, price_col=price_col, log_returns=True, horizons=[horizon])
    return_col = f"fwd_ret_{horizon}"

    if label_cols is None:
        ignore = {price_col, "ret", return_col}
        label_cols = [c for c in df.columns if c not in ignore]

    eval_tbl = evaluate_labels(df, label_cols, return_col, walkforward=walkforward)
    classes = classify_labels(eval_tbl, auc_thresh=auc_thresh, corr_thresh=corr_thresh)
    eval_tbl["lbl_class"] = eval_tbl.label.map(
        lambda x: (
            "pred"
            if x in classes["predictive"]
            else "desc" if x in classes["descriptive"] else np.nan
        )
    )

    df["meta_signal"] = build_meta_signal(
        df,
        predictive_cols=classes["predictive"],
        method=meta_method,
        target_col=return_col,
        walkforward=walkforward,
    )

    sig_col = "meta_signal"
    if filters:
        df["meta_signal_filtered"] = apply_filters(df["meta_signal"], df, filters)
        sig_col = "meta_signal_filtered"

    bt = backtest_walkforward(
        df,
        signal_col=sig_col,
        return_col="ret",
        fees_bps=fees_bps,
        slippage_bps=slippage_bps,
    )
    return {"df": df, "evaluation": eval_tbl, "backtest": bt}
