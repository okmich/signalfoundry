"""
MTDC — Genetic Algorithm for weight optimisation.

train_ga_weights  — evolve real-valued weights [w₁…wₖ] maximising Sharpe (or return)
                    of the MTDC multi-threshold strategy on training data.

GA parameters match Adegboye, Kampouridis & Otero (2023) Section 3.2:
  population=500, generations=50, tournament=7, cx=0.90, mut=0.10, elitism=10%.

Population is seeded with single-threshold specialists (one per STDC slot) to
guarantee the worst-case MTDC performance is at least as good as the best STDC.

Reference: Adegboye, Kampouridis & Otero (AI Review, 2023) Sections 3.2, 4.
"""
import numpy as np

from okmich_quant_features.directional_change import idc_parse

from ._cgpts_model import predict_trend_end
from ._mtdc_thresholds import FitnessMode


# ---------------------------------------------------------------------------
# Signal precomputation
# ---------------------------------------------------------------------------

def _precompute_stdc_signals(prices, stdc_model: dict, theta: float, alpha: float = 1.0) -> list:
    """
    Precompute per-bar STDC signals for all DC events in prices.

    Called once per threshold before GA evolution to avoid re-running the
    idc_parse + C+GP prediction for each of 25,000 fitness evaluations.

    Returns
    -------
    list of dict
        Each dict: {bar: int, action: 'buy'|'sell'|'hold', reversal_bar: int}
        'buy'  — upturn DC confirmed + αDC predicted (enter long, exit at reversal_bar).
        'sell' — downturn DC confirmed (exit any open long).
        'hold' — upturn DC confirmed + βDC predicted (skip entry).
    """
    idc = idc_parse(prices, theta, alpha)
    price_arr = prices.values.astype(np.float64)
    n = len(price_arr)

    upturn_dc = idc['upturn_dc'].values
    downturn_dc = idc['downturn_dc'].values
    pl = idc['pl'].values
    ph = idc['ph'].values  # noqa: F841 (available for future short-side extension)
    t_dc0 = idc['t_dc0'].values

    signals = []
    prev_dcc_price = None
    prev_has_os = False

    for i in range(n):
        p = price_arr[i]
        if upturn_dc[i]:
            trough_bar = int(t_dc0[i])
            trough_price = float(pl[i])
            dc_len = i - trough_bar
            pred = predict_trend_end(
                dc_length=dc_len, dcc_price=p, ext_end_price=trough_price,
                prev_dcc_price=prev_dcc_price, prev_has_os=prev_has_os,
                cgpts_model=stdc_model,
            )
            if pred['trend_type'] == 'alpha_dc':
                signals.append({
                    'bar': i,
                    'action': 'buy',
                    'reversal_bar': i + pred['estimated_dce_offset'],
                })
            else:
                signals.append({'bar': i, 'action': 'hold', 'reversal_bar': i})
            prev_dcc_price = p
            prev_has_os = pred['trend_type'] == 'alpha_dc'

        elif downturn_dc[i]:
            signals.append({'bar': i, 'action': 'sell', 'reversal_bar': i})
            prev_dcc_price = p
            prev_has_os = False

    return signals


# ---------------------------------------------------------------------------
# MTDC simulation for GA fitness
# ---------------------------------------------------------------------------

def _simulate_mtdc_for_fitness(all_signals: list, weights: np.ndarray, price_arr: np.ndarray, tx_cost_rate: float = 0.0) -> list:
    """
    Simulate MTDC strategy (long + short) using precomputed per-threshold signals.

    Applies the weighted-sum consensus rule:
      buy  if weight_buy > weight_sell → long entry (or short exit).
      sell if weight_sell > weight_buy → short entry (or long exit).
    Does NOT apply the ITA regime gate (gate is test-time only).

    Long exit: C+GP target bar reached OR sell consensus.
    Short exit: buy consensus only (no C+GP target — STDC models trained on
    upturn events; short exit defaults to consensus reversal).

    Parameters
    ----------
    all_signals : list of list
        all_signals[i] = list of signal dicts for threshold i.
    weights : np.ndarray
        Shape (k,), weights in [0, 1].
    price_arr : np.ndarray
        Price array corresponding to the training window.
    tx_cost_rate : float
        One-way transaction cost (deducted on entry and exit).

    Returns
    -------
    list of float
        Per-trade returns (decimal, not percentage).
    """
    k = len(weights)

    # Build bar → {threshold_idx: signal} lookup
    timeline: dict = {}
    for i in range(k):
        for sig in all_signals[i]:
            b = sig['bar']
            if b not in timeline:
                timeline[b] = {}
            timeline[b][i] = sig

    sorted_bars = sorted(timeline)
    position = 0   # 0=flat, 1=long, -1=short
    entry_price = 0.0
    target_exit_bar = 0
    trade_returns = []

    for bar in sorted_bars:
        if bar >= len(price_arr):
            break
        p = price_arr[bar]
        bar_sigs = timeline[bar]

        buy_w = sum(weights[i] for i, s in bar_sigs.items() if s['action'] == 'buy')
        sell_w = sum(weights[i] for i, s in bar_sigs.items() if s['action'] == 'sell')

        # ── Exit checks ───────────────────────────────────────────────────────
        if position == 1:
            if bar >= target_exit_bar:
                # C+GP estimated DCE reached
                pnl = (p - entry_price) / entry_price - tx_cost_rate
                trade_returns.append(pnl)
                position = 0
            elif sell_w > buy_w:
                pnl = (p - entry_price) / entry_price - tx_cost_rate
                trade_returns.append(pnl)
                position = 0

        elif position == -1:
            if buy_w > sell_w:
                pnl = (entry_price - p) / entry_price - tx_cost_rate
                trade_returns.append(pnl)
                position = 0

        # ── Entry check ───────────────────────────────────────────────────────
        if position == 0:
            if buy_w > sell_w and buy_w > 0:
                buy_sigs = {i: s for i, s in bar_sigs.items() if s['action'] == 'buy'}
                weighted_rev = sum(weights[i] * s['reversal_bar'] for i, s in buy_sigs.items()) / buy_w
                entry_price = p * (1.0 + tx_cost_rate)
                target_exit_bar = int(round(weighted_rev))
                position = 1
            elif sell_w > buy_w and sell_w > 0:
                entry_price = p * (1.0 + tx_cost_rate)
                position = -1

    return trade_returns


def _compute_sharpe(returns: list, risk_free_rate: float = 0.0) -> float:
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=float)
    std = arr.std()
    if std < 1e-10:
        return 0.0
    return float((arr.mean() - risk_free_rate) / std)


def _compute_total_return(returns: list) -> float:
    if not returns:
        return 0.0
    capital = 1.0
    for r in returns:
        capital *= 1.0 + r
    return (capital - 1.0) * 100.0


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

def _uniform_crossover(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Uniform crossover: each gene swapped with p=0.5; one child selected randomly."""
    child1, child2 = p1.copy(), p2.copy()
    mask = np.random.random(len(p1)) < 0.5
    child1[mask], child2[mask] = p2[mask], p1[mask]
    return child1 if np.random.randint(2) == 0 else child2


def _uniform_mutation(individual: np.ndarray) -> np.ndarray:
    """Uniform mutation: each gene resampled from U(0,1) with p=0.5."""
    mutant = individual.copy()
    mask = np.random.random(len(mutant)) < 0.5
    mutant[mask] = np.random.uniform(0.0, 1.0, mask.sum())
    return mutant


# ---------------------------------------------------------------------------
# Main GA training function
# ---------------------------------------------------------------------------

def train_ga_weights(prices, stdc_models: list, thetas: list, alpha: float = 1.0, population_size: int = 500, n_generations: int = 50, tournament_size: int = 7, cx_prob: float = 0.90, mut_prob: float = 0.10, elitism_fraction: float = 0.10, fitness_mode: FitnessMode | str = FitnessMode.SHARPE, random_seed: int = 42, tx_cost_rate: float = 0.0) -> np.ndarray:
    """
    Evolve GA weight vector [w₁…wₖ] for K STDC models.

    Fitness function is Sharpe ratio (default) or cumulative return % of the
    simulated MTDC strategy on the training price series.  Transaction costs
    are not included by default (matching the paper's assumption).

    The initial population is seeded with K single-threshold specialists —
    chromosomes where wᵢ=1 and all others=0 — guaranteeing that the evolved
    multi-threshold solution is at least as good as the best single-threshold.

    Parameters
    ----------
    prices : pd.Series
        Training price series (same data used to train the STDC models).
    stdc_models : list of dict
        Trained C+GP+TS models from train_cgpts_model(), one per theta.
    thetas : list of float
        DC thresholds corresponding to stdc_models.
    alpha : float
        Asymmetric attenuation coefficient.
    population_size : int
        GA population size.  Paper default: 500.
    n_generations : int
        Number of GA generations.  Paper default: 50.
    tournament_size : int
        Tournament selection size.  Paper default: 7.
    cx_prob : float
        Probability of crossover being applied.  Paper default: 0.90.
    mut_prob : float
        Probability of mutation being applied.  Paper default: 0.10.
    elitism_fraction : float
        Fraction of best individuals preserved per generation.  Default: 0.10.
    fitness_mode : FitnessMode or str
        FitnessMode.SHARPE — maximise Sharpe ratio (default, reduces MDD).
        FitnessMode.RETURN — maximise cumulative return %.
    random_seed : int
        NumPy random seed.
    tx_cost_rate : float
        One-way transaction cost rate for fitness simulation.

    Returns
    -------
    np.ndarray
        Shape (k,), best evolved weight vector.  Weights ∈ [0, 1], not normalised.

    Raises
    ------
    ValueError
        If stdc_models and thetas have different lengths, or fitness_mode is invalid.
    """
    if len(stdc_models) != len(thetas):
        raise ValueError(f"stdc_models ({len(stdc_models)}) and thetas ({len(thetas)}) must have the same length.")
    fitness_mode = FitnessMode(fitness_mode)

    k = len(stdc_models)
    np.random.seed(random_seed)

    # Precompute signals once — avoids re-running C+GP inference for each fitness eval
    all_signals = [
        _precompute_stdc_signals(prices, stdc_models[i], thetas[i], alpha)
        for i in range(k)
    ]
    price_arr = prices.values.astype(np.float64)
    elite_size = max(1, int(population_size * elitism_fraction))

    # ── Initialise population with specialists + random ───────────────────────
    population = []
    for i in range(k):
        specialist = np.zeros(k)
        specialist[i] = 1.0
        population.append(specialist)
    while len(population) < population_size:
        population.append(np.random.uniform(0.0, 1.0, k))

    def _fitness(w):
        returns = _simulate_mtdc_for_fitness(all_signals, w, price_arr, tx_cost_rate)
        if fitness_mode == FitnessMode.SHARPE:
            return _compute_sharpe(returns)
        return _compute_total_return(returns)

    best_weights = population[0].copy()
    best_fitness = _fitness(best_weights)

    for _ in range(n_generations):
        fitnesses = np.array([_fitness(ind) for ind in population])

        # Track global best
        gen_best_idx = int(np.argmax(fitnesses))
        if fitnesses[gen_best_idx] > best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_weights = population[gen_best_idx].copy()

        # Elitism: preserve top individuals
        elite_idx = np.argsort(fitnesses)[::-1][:elite_size]
        elite = [population[i].copy() for i in elite_idx]

        # Selection + variation → offspring
        offspring = []
        n_pop = len(population)
        while len(offspring) < population_size - elite_size:
            # Tournament selection for parent 1
            t1_idx = np.random.choice(n_pop, min(tournament_size, n_pop), replace=False)
            p1 = population[t1_idx[int(np.argmax(fitnesses[t1_idx]))]]
            # Tournament selection for parent 2
            t2_idx = np.random.choice(n_pop, min(tournament_size, n_pop), replace=False)
            p2 = population[t2_idx[int(np.argmax(fitnesses[t2_idx]))]]

            child = _uniform_crossover(p1, p2) if np.random.random() < cx_prob else p1.copy()
            if np.random.random() < mut_prob:
                child = _uniform_mutation(child)
            offspring.append(np.clip(child, 0.0, 1.0))

        population = elite + offspring

    return best_weights
