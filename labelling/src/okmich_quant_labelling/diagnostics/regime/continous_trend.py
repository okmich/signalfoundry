from typing import Any, Dict

import numpy as np
import pandas as pd
from pandas import Series
from tstrends.label_tuning import RemainingValueTuner, LinearWeightedAverage
from tstrends.optimization import Optimizer
from tstrends.returns_estimation import ReturnsEstimatorWithFees, FeesConfig
from tstrends.trend_labelling import BinaryCTL, TernaryCTL


def bi_ctl_label(
    series: pd.Series, omega: float, smoothing_window: int = 36
) -> (pd.Series, pd.Series):
    prices = series.to_list()
    labeller = BinaryCTL(omega=omega)
    labelled_prices = labeller.get_labels(prices)
    tuner = RemainingValueTuner()
    linear_smoother = LinearWeightedAverage(
        window_size=smoothing_window, direction="centered"
    )
    tuned_labels_with_linear_smoothing = np.array(
        tuner.tune(
            prices,
            labelled_prices,
            normalize_over_interval=True,
            smoother=linear_smoother,
            enforce_monotonicity=False,
        )
    )

    return (
        pd.Series(labelled_prices, index=series.index, name="bi_ctl_label"),
        pd.Series(
            tuned_labels_with_linear_smoothing,
            index=series.index,
            name="bi_ctl_rem_score",
        ),
    )


def tri_ctl_label(
    series: pd.Series,
    marginal_change_thres: float,
    window_size: int,
    smoothing_window: int = 36,
) -> tuple[Series, Series]:
    prices = series.to_list()

    labeller = TernaryCTL(
        marginal_change_thres=marginal_change_thres, window_size=window_size
    )
    labelled_prices = labeller.get_labels(prices)

    tuner = RemainingValueTuner()
    linear_smoother = LinearWeightedAverage(
        window_size=smoothing_window, direction="centered"
    )
    tuned_labels_with_linear_smoothing = np.array(
        tuner.tune(
            prices,
            labelled_prices,
            normalize_over_interval=True,
            smoother=linear_smoother,
            enforce_monotonicity=False,
        )
    )
    return (
        pd.Series(labelled_prices, index=series.index, name="tri_ctl_label"),
        pd.Series(
            tuned_labels_with_linear_smoothing,
            index=series.index,
            name="tri_ctl_rem_score",
        ),
    )


##############################################################################################################
########################################## OPTIMIZATION FUNCTIONS ############################################
##############################################################################################################


def optimize_bi_ctl_label(
    series: pd.Series, omega_bounds: tuple[float, float] = (0, 0.1), fees: float = 0.0
) -> Dict[str, Any]:
    fees_config = FeesConfig(lp_transaction_fees=fees, sp_transaction_fees=fees)
    custom_bounds = {
        "omega": omega_bounds,
    }
    optimizer = Optimizer(
        ReturnsEstimatorWithFees(fees_config), initial_points=5, nb_iter=100
    )
    optimization_results = optimizer.optimize(
        BinaryCTL, series.to_list(), bounds=custom_bounds, verbose=1
    )
    return optimization_results["params"]


def optimize_tri_ctl_label(
    series: pd.Series,
    marginal_change_thres_values: tuple[float, float],
    window_size_values: tuple[int, int],
    holding_fees: float = 0.0,
    txn_fees: float = 0.0,
) -> Dict[str, Any]:
    fees_config = FeesConfig(
        lp_transaction_fees=txn_fees,
        sp_transaction_fees=txn_fees,
        lp_holding_fees=holding_fees,
        sp_holding_fees=holding_fees,
    )
    custom_bounds = {
        "marginal_change_thres": marginal_change_thres_values,
        "window_size": window_size_values,
    }
    optimizer = Optimizer(
        ReturnsEstimatorWithFees(fees_config), initial_points=10, nb_iter=50
    )
    optimization_results = optimizer.optimize(
        TernaryCTL, series.to_list(), custom_bounds, verbose=1
    )
    return optimization_results["params"]
