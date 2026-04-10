"""
C+GP+TS — Training pipeline and live prediction for the αDC/βDC classifier
and GP overshoot regressor.

train_cgpts_model   — full offline training: GP first (perfect foresight),
                      then classifier on X1–X5 features.
predict_trend_end   — online inference: classify then optionally predict OS length.

The GP regression runs first (Section 3.1 of the paper) because running
classification first would contaminate the GP training data with
misclassification noise.  The classifier is trained on the clean
X1–X6 features regardless of GP quality.

Reference: Adegboye & Kampouridis (2020) Sections 3.1–3.2; Algorithms 2–3.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from okmich_quant_features.directional_change import (
    extract_dc_classification_features,
    label_alpha_beta_dc,
    parse_dc_events,
)

from ._cgp_gp import run_gp_regression


def train_cgpts_model(prices: pd.Series, theta: float, alpha: float = 1.0, n_generations: int = 37, population_size: int = 500, cx_prob: float = 0.98, mut_prob: float = 0.02, elitism_fraction: float = 0.10, random_seed: int = 42) -> dict:
    """
    Train the full C+GP+TS model on a price series.

    Execution order (Adegboye & Kampouridis 2020 Section 3.1):
      1. Parse DC trends with perfect foresight on training data.
      2. Label αDC / βDC (has_os column).
      3. Run GP symbolic regression on αDC trends only → GP model.
      4. Extract X1–X5 features for all trends (except last, unlabelled).
      5. Train αDC/βDC classifier on features + labels.

    Parameters
    ----------
    prices : pd.Series
        Training price series (close prices).
    theta : float
        DC threshold.  Should come from ITA's Bayesian optimisation when
        integrating with the ITA regime system.
    alpha : float
        Asymmetric attenuation coefficient for downward DC.  Default 1.0.
    n_generations, population_size, cx_prob, mut_prob, elitism_fraction, random_seed
        GP hyperparameters passed to run_gp_regression().

    Returns
    -------
    dict with keys:
        gp_func        : callable — best GP model: gp_func(dc_l) -> predicted os_l (bars).
        gp_rmse        : float   — RMSE of best GP model on αDC training trends.
        gp_equation    : str     — string form of the evolved GP equation.
        classifier     : sklearn Pipeline — fitted StandardScaler + GradientBoostingClassifier.
        alpha_rate     : float   — fraction of training trends that are αDC.
        n_alpha        : int     — number of αDC training trends.
        n_beta         : int     — number of βDC training trends.
        n_trends_total : int     — total labelled training trends (last row excluded).
        theta          : float   — DC threshold used.
        alpha_param    : float   — alpha parameter used.

    Raises
    ------
    ValueError
        If fewer than 10 αDC trends are found in the training data (insufficient
        for meaningful GP regression).
    """
    trends_raw = parse_dc_events(prices, theta, alpha)
    if len(trends_raw) < 4:
        raise ValueError(f"train_cgpts_model: only {len(trends_raw)} DC trends parsed. "
                         "Need at least 4. Use a longer training window or smaller theta.")

    trends = label_alpha_beta_dc(trends_raw)
    # Drop last row (has_os=NaN — no subsequent trend to label from)
    labelled = trends.dropna(subset=['has_os'])

    alpha_mask = labelled['has_os'] == True  # noqa: E712
    beta_mask = labelled['has_os'] == False  # noqa: E712
    n_alpha = int(alpha_mask.sum())
    n_beta = int(beta_mask.sum())

    if n_alpha < 10:
        raise ValueError(f"train_cgpts_model: only {n_alpha} αDC trends in training set. "
                         "Need at least 10 for GP regression. Use a longer training window.")
    if n_beta < 1:
        raise ValueError("train_cgpts_model: all trends are αDC — no βDC trends to train a "
                         "binary classifier. Use a more volatile training series or larger theta.")

    # Step 1 — GP regression on αDC trends only (perfect foresight)
    alpha_rows = labelled[alpha_mask]
    dc_lengths = alpha_rows['dc_length'].values.astype(float)
    os_lengths = alpha_rows['os_length'].values.astype(float)
    gp_func, gp_rmse, gp_equation = run_gp_regression(
        dc_lengths, os_lengths,
        n_generations=n_generations, population_size=population_size,
        cx_prob=cx_prob, mut_prob=mut_prob,
        elitism_fraction=elitism_fraction, random_seed=random_seed,
    )

    # Step 2 — Extract X1–X5 features for all labelled trends
    features_df = extract_dc_classification_features(labelled)
    X_train = features_df.values.astype(float)
    y_train = labelled['has_os'].astype(int).values  # 1=αDC, 0=βDC

    # Replace NaN in X4 (first row has no previous DCC) with 0
    X_train = np.nan_to_num(X_train, nan=0.0)

    # Step 3 — Train classifier
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=random_seed)),
    ])
    clf.fit(X_train, y_train)

    return {
        'gp_func': gp_func,
        'gp_rmse': gp_rmse,
        'gp_equation': gp_equation,
        'classifier': clf,
        'alpha_rate': n_alpha / (n_alpha + n_beta),
        'n_alpha': n_alpha,
        'n_beta': n_beta,
        'n_trends_total': n_alpha + n_beta,
        'theta': theta,
        'alpha_param': alpha,
    }


def predict_trend_end(dc_length: int, dcc_price: float, ext_end_price: float, prev_dcc_price: float | None, prev_has_os: bool, cgpts_model: dict) -> dict:
    """
    Predict whether the DC event that just fired will be followed by an OS
    (αDC) and, if so, estimate how many bars the OS will last.

    Called live at each DC confirmation bar with features computable at that point.

    Parameters
    ----------
    dc_length : int
        Bars of the DC event that just fired (dcc_pos - ext_end_pos).
    dcc_price : float
        Price at the DC confirmation point.
    ext_end_price : float
        Price at the extreme that the DC was measured from (EXT point).
    prev_dcc_price : float or None
        Price at the preceding DC confirmation point.  Pass None if this is
        the first trend.
    prev_has_os : bool
        Whether the preceding trend was αDC.  Pass False if no prior trend.
    cgpts_model : dict
        Output of train_cgpts_model().

    Returns
    -------
    dict with keys:
        trend_type          : str   — 'alpha_dc' or 'beta_dc'.
        p_alpha             : float — classifier probability of αDC.
        p_beta              : float — classifier probability of βDC.
        predicted_os_bars   : float — predicted OS length in bars (0 if βDC).
        estimated_dce_offset: int   — predicted bars from DCC to trend end.
                              Use: exit_bar = dcc_bar + estimated_dce_offset.
    """
    clf = cgpts_model['classifier']
    gp_func = cgpts_model['gp_func']

    x1 = abs(dcc_price - ext_end_price)
    x2 = max(dc_length, 1)
    x3 = x1 / x2
    x4 = prev_dcc_price if prev_dcc_price is not None else 0.0
    x5 = 1 if prev_has_os else 0

    features = np.array([[x1, x2, x3, x4, x5]], dtype=float)
    prediction = int(clf.predict(features)[0])
    proba = clf.predict_proba(features)[0]

    if prediction == 1:
        try:
            os_pred = float(gp_func(float(dc_length)))
            os_pred = max(0.0, os_pred) if np.isfinite(os_pred) else 0.0
        except Exception:
            os_pred = 0.0
        return {
            'trend_type': 'alpha_dc',
            'p_alpha': float(proba[1]),
            'p_beta': float(proba[0]),
            'predicted_os_bars': os_pred,
            'estimated_dce_offset': int(round(os_pred)),
        }
    return {
        'trend_type': 'beta_dc',
        'p_alpha': float(proba[1]),
        'p_beta': float(proba[0]),
        'predicted_os_bars': 0.0,
        'estimated_dce_offset': 0,
    }
