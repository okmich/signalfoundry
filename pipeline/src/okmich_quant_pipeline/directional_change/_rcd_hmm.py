"""
RCD-HMM — Regime Change Detection via Hidden Markov Model.

Implements the HMM utilities for ITA Algorithm 1 (Wu & Han 2023, Section 3.2):
  - fit_rcd_hmm        : fit 2-state Gaussian HMM on RDC sequence
  - s1_state_index     : identify which pomegranate state index is S1 (normal)
  - assign_regime_labels : Viterbi-decode full RDC sequence to regime labels
  - get_current_regime : query HMM at a single DC confirmation point (live mode)

Input: RDC values from idc_parse() — filter rows where upturn_dc or downturn_dc
       is True and rdc is not NaN, then pass the rdc column.

Reference: Wu & Han arXiv:2309.15383 (2023), Section 3.2, Equations 4–5.
           Chen & Tsang CRC Press (2021), Section 2.3.1.
"""
import numpy as np
import pandas as pd
import torch

from okmich_quant_ml.hmm import DistType, InferenceMode, PomegranateHMM

_MIN_RDC_VALUES = 20


def fit_rcd_hmm(rdc: pd.Series | np.ndarray, random_state: int = 42) -> PomegranateHMM:
    """
    Fit 2-state Gaussian HMM on the RDC sequence from completed DC trends.

    Uses Baum-Welch (EM) internally via the pomegranate backend.

    Parameters
    ----------
    rdc : pd.Series or np.ndarray
        RDC values at DC confirmation bars. Obtain from idc_parse() output:
            idc = idc_parse(prices, theta, alpha)
            rdc = idc.loc[idc['rdc'].notna(), 'rdc']
        Must contain at least 20 finite values. No NaN/inf allowed.
    random_state : int
        Seed for reproducible k-means initialisation.

    Returns
    -------
    PomegranateHMM
        Fitted model. State ordering is arbitrary — call s1_state_index()
        to identify which index maps to S1 (normal/low-volatility regime).

    Raises
    ------
    ValueError
        If fewer than 20 values are provided or if values contain NaN/inf.
    RuntimeError
        If the model log-likelihood is not finite after fitting.

    Notes
    -----
    Wu & Han (2023) use raw RDC (not log-transformed) as input.
    If convergence is poor, consider log_r() from okmich_quant_features.
    """
    arr = np.asarray(rdc, dtype=np.float64).reshape(-1, 1)

    if len(arr) < _MIN_RDC_VALUES:
        raise ValueError(f"Need at least {_MIN_RDC_VALUES} RDC values to fit HMM, got {len(arr)}")
    if not np.isfinite(arr).all():
        raise ValueError("RDC sequence contains NaN or infinite values")

    model = PomegranateHMM(distribution_type=DistType.NORMAL, n_states=2, covariance_type='full', max_iter=1000, inference_mode=InferenceMode.VITERBI, random_state=random_state)
    model.fit(arr)

    if not np.isfinite(model.log_likelihood(arr)):
        raise RuntimeError(
            "HMM log-likelihood is not finite after fitting. "
            "Try: (1) more training data, (2) different theta, (3) log(RDC) input."
        )
    return model


def s1_state_index(model: PomegranateHMM) -> int:
    """
    Identify which pomegranate state index corresponds to S1 (normal regime).

    S1 is defined as the state with the lower mean RDC — lower volatility,
    mean-reverting behaviour. This mapping must be established once after
    fitting and reused consistently for all label generation and live queries.

    Parameters
    ----------
    model : PomegranateHMM
        Fitted 2-state HMM from fit_rcd_hmm().

    Returns
    -------
    int
        State index (0 or 1) of the S1 (normal) regime.

    Notes
    -----
    Wu & Han (2023) Section 3.2; Chen & Tsang (2021) Section 5.
    """
    params = model.parameters
    state_means = np.array([p["means"][0] for p in params], dtype=np.float64)
    return int(np.argmin(state_means))


def assign_regime_labels(model: PomegranateHMM, rdc: pd.Series | np.ndarray, s1_idx: int) -> pd.Series:
    """
    Decode regime sequence using Viterbi on the full training RDC sequence.

    Maps pomegranate state indices to regime labels: 1 = S1 (normal), 2 = S2 (abnormal).

    CRITICAL: Always call with the complete training sequence so that Viterbi
    can use full temporal context. Do not call this bar-by-bar.

    Parameters
    ----------
    model : PomegranateHMM
        Fitted model from fit_rcd_hmm(), must have InferenceMode.VITERBI.
    rdc : pd.Series or np.ndarray
        Same RDC sequence used to fit the model. If a Series, the returned
        labels will share the same index.
    s1_idx : int
        S1 state index from s1_state_index().

    Returns
    -------
    pd.Series
        Integer regime labels (1=normal, 2=abnormal) with the same index as
        rdc if a Series was supplied, or a RangeIndex otherwise.

    Raises
    ------
    ValueError
        If all trends are assigned to a single regime after decoding.

    Notes
    -----
    Wu & Han (2023) Section 3.2 — Viterbi decoding.
    Chen & Tsang (2021) Section 5 — state labelling convention.
    """
    index = rdc.index if isinstance(rdc, pd.Series) else None
    arr = np.asarray(rdc, dtype=np.float64).reshape(-1, 1)

    raw_states = model.predict(arr)
    labels = np.where(raw_states == s1_idx, 1, 2).astype(np.int8)

    n_s1 = int((labels == 1).sum())
    n_s2 = int((labels == 2).sum())

    if n_s2 == 0:
        raise ValueError(
            "HMM assigned all trends to S1. Training window may lack a volatility "
            "event. Check data period or reduce theta."
        )
    if n_s1 == 0:
        raise ValueError(
            "HMM assigned all trends to S2. Check HMM convergence and "
            "state assignment logic."
        )

    return pd.Series(labels, index=index, dtype=np.int8)


def get_current_regime(model: PomegranateHMM, rdc_value: float, s1_idx: int) -> int:
    """
    Query HMM for regime at the current DC confirmation bar (live / Strategy A).

    Uses a single RDC value without temporal context. Intended for live use at
    each upward DC confirmation when deciding whether to enter a position.

    Parameters
    ----------
    model : PomegranateHMM
        Fitted model from fit_rcd_hmm().
    rdc_value : float
        RDC at the current DC confirmation bar.
    s1_idx : int
        S1 state index from s1_state_index().

    Returns
    -------
    int
        1 if current regime is S1 (normal — trade normally).
        2 if current regime is S2 (abnormal — suspend trading).

    Notes
    -----
    Wu & Han (2023) Algorithm 1, line 27.
    For temporal-context-aware live queries, use Strategy B or C instead
    (RollingViterbiLabeller / PosteriorThresholdLabeller in the spec).
    """
    # Pomegranate Viterbi / forward-backward requires >= 2 observations.
    # For a single-point query we compute log P(x | state_k) per state and
    # take the argmax — equivalent to MAP classification with uniform priors.
    # NOTE: accesses model._model.distributions (pomegranate internal) — fragile
    # against PomegranateHMM API changes; should be replaced with a public method.
    x = torch.tensor([[rdc_value]], dtype=torch.float64)
    log_probs = np.array([
        d.log_probability(x).item()
        for d in model._model.distributions
    ])
    state = int(np.argmax(log_probs))
    return 1 if state == s1_idx else 2
