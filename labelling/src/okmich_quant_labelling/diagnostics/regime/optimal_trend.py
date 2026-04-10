from typing import List, Dict, Any

import pandas as pd
from pandas import Series
from tstrends.label_tuning import RemainingValueTuner, LinearWeightedAverage
from tstrends.optimization import Optimizer
from tstrends.returns_estimation import FeesConfig, ReturnsEstimatorWithFees
from tstrends.trend_labelling import (
    OracleBinaryTrendLabeller,
    OracleTernaryTrendLabeller,
)


def bi_oracle_label(series: pd.Series, txn_costs: float, smoothing_window: int = 36) -> tuple[Series, Series]:
    prices = series.to_list()
    labeller = OracleBinaryTrendLabeller(transaction_cost=txn_costs)
    labelled_prices = labeller.get_labels(prices)

    tuner = RemainingValueTuner()
    linear_smoother = LinearWeightedAverage(
        window_size=smoothing_window, direction="centered"
    )
    tuned_labels_with_rem = tuner.tune(
        prices,
        labelled_prices,
        normalize_over_interval=True,
        smoother=linear_smoother,
        enforce_monotonicity=False,
    )
    return (
        pd.Series(labelled_prices, index=series.index, name="bi_oracle_label"),
        pd.Series(
            tuned_labels_with_rem, index=series.index, name="bi_oracle_rem_score"
        ),
    )


def tri_oracle_label(
    series: pd.Series,
    transaction_cost: float,
    neutral_reward_factor: float,
    smoothing_window: int = 36,
) -> tuple[Series, Series]:
    prices = series.to_list()
    labeller = OracleTernaryTrendLabeller(
        transaction_cost=transaction_cost, neutral_reward_factor=neutral_reward_factor
    )
    labelled_prices = labeller.get_labels(prices)

    tuner = RemainingValueTuner()
    linear_smoother = LinearWeightedAverage(
        window_size=smoothing_window, direction="centered"
    )
    tuned_labels_with_rem = tuner.tune(
        prices,
        labelled_prices,
        normalize_over_interval=True,
        smoother=linear_smoother,
        enforce_monotonicity=False,
    )
    return (
        pd.Series(labelled_prices, index=series.index, name="tri_oracle_label"),
        pd.Series(
            tuned_labels_with_rem, index=series.index, name="tri_oracle_rem_score"
        ),
    )


##############################################################################################################
########################################## OPTIMIZATION FUNCTIONS ############################################
##############################################################################################################


def optimize_bi_oracle_label(series: pd.Series, txn_cost_values: tuple[float, float], fees: float = 0.0) -> Dict[str, Any]:
    fees_config = FeesConfig(lp_transaction_fees=fees, sp_transaction_fees=fees)
    custom_bounds = {
        "transaction_cost": txn_cost_values,
    }
    optimizer = Optimizer(ReturnsEstimatorWithFees(fees_config), initial_points=5, nb_iter=100)
    opt_result = optimizer.optimize(OracleBinaryTrendLabeller, series.to_list(), bounds=custom_bounds, verbose=True)
    return opt_result["params"]


def optimize_tri_oracle_label(series: pd.Series, transaction_costs_values: List[float],
                              neutral_reward_factor_values: List[float], holding_fees: float = 0.0,
                              txn_fees: float = 0.0,) -> Dict[str, Any]:
    fees_config = FeesConfig(
        lp_transaction_fees=txn_fees,
        sp_transaction_fees=txn_fees,
        lp_holding_fees=holding_fees,
        sp_holding_fees=holding_fees,
    )
    custom_bounds = {
        "transaction_cost": transaction_costs_values,
        "neutral_reward_factor": neutral_reward_factor_values,
    }
    optimizer = Optimizer(ReturnsEstimatorWithFees(fees_config), initial_points=10, nb_iter=50)
    optimization_results = optimizer.optimize(OracleTernaryTrendLabeller, series.to_list(), custom_bounds, verbose=1)
    return optimization_results["params"]
