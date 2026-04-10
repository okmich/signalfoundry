"""
TSFDC — BBTheta binary classifier training and inference.

train_bbtheta_classifier  — LightGBM classifier on (TMV, T, OSV, COP) features.
predict_bbtheta           — single-event inference (conservative NaN handling).

Reference: Bakhach, Tsang & Jalalian (IEEE CIFEr, 2016) Section 3;
           Bakhach, Tsang & Chinthalapati (ISAFM, 2018) Section 3.
"""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

FEATURE_COLS = ['TMV', 'T', 'OSV', 'COP']


def train_bbtheta_classifier(features_df: pd.DataFrame, random_seed: int = 42) -> Pipeline:
    """
    Train BBTheta binary classifier on (TMV, T, OSV, COP) features.

    Drops rows with NaN in any feature or label column before training.
    Uses LightGBM with is_unbalance=True to handle class imbalance
    (most STheta events don't reach BTheta magnitude).

    Parameters
    ----------
    features_df : pd.DataFrame
        Feature matrix with columns [TMV, T, OSV, COP, bbtheta].
        Typically built by extract_tsfdc_features() + label_bbtheta().
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted StandardScaler + LGBMClassifier pipeline.

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    ValueError
        If fewer than 20 clean training samples remain after NaN removal,
        or if only one class is present in the labels.
    """
    if not _LGBM_AVAILABLE:
        raise ImportError("lightgbm is required for BBTheta classifier. Install with: pip install lightgbm")

    required = set(FEATURE_COLS) | {'bbtheta'}
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df missing columns: {missing}")

    df = features_df.dropna(subset=FEATURE_COLS + ['bbtheta'])
    if len(df) < 20:
        raise ValueError(
            f"train_bbtheta_classifier: need at least 20 clean samples, got {len(df)}. "
            "Use a longer training window or smaller stheta."
        )

    X = df[FEATURE_COLS].values.astype(float)
    y = df['bbtheta'].astype(int).values

    if len(np.unique(y)) < 2:
        raise ValueError(
            "train_bbtheta_classifier: only one class in labels. "
            "Need both True and False BBTheta samples. "
            "Use a longer training window or adjust btheta/stheta ratio."
        )

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, is_unbalance=True, random_state=random_seed, verbose=-1)),
    ])
    clf.fit(X, y)
    return clf


def predict_bbtheta(clf: Pipeline, tmv: float, t: float, osv: float, cop: float) -> bool:
    """
    Predict BBTheta for a single STheta DC event.

    Returns False (conservative) if cop is NaN — this means no BTheta DCC has
    fired yet in the current window, making COP undefined.  The conservative
    default routes these events through Rule 1 (immediate entry) rather than
    waiting for BTheta to confirm.

    Parameters
    ----------
    clf : sklearn.pipeline.Pipeline
        Fitted classifier from train_bbtheta_classifier().
    tmv, t, osv, cop : float
        Feature values for the current STheta DC event.

    Returns
    -------
    bool
        True  — trend predicted to reach BTheta magnitude (Rule 2 delayed entry).
        False — trend predicted to reverse before BTheta (Rule 1 immediate entry).
    """
    if np.isnan(cop):
        return False
    X = np.array([[tmv, t, osv, cop]], dtype=float)
    return bool(clf.predict(X)[0])
