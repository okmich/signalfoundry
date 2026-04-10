"""
C+GP+TS — Genetic Programming symbolic regression for OS length prediction.

Evolves a non-linear equation OS_l = f(DC_l) using DEAP, trained exclusively
on αDC trends (those confirmed to have an overshoot).  This is Step 1 of the
C+GP pipeline and runs BEFORE the classifier is trained.

GP parameters match Adegboye & Kampouridis (2020) Table 3 (I/F-Race tuned):
  population=500, generations=37, tournament=3, cx=0.98, mut=0.02,
  elitism=10%, max_depth=3.

Reference: Adegboye & Kampouridis (2020) Section 3.1; Table 3.
"""
import operator
import random

import numpy as np

try:
    from deap import algorithms, base, creator, gp, tools
    _DEAP_AVAILABLE = True
except ImportError:
    _DEAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Protected math primitives
# ---------------------------------------------------------------------------

def _protected_div(a: float, b: float) -> float:
    return 1.0 if abs(b) < 1e-10 else a / b


def _protected_log(x: float) -> float:
    return 0.0 if x <= 0.0 else float(np.log(x))


def _protected_exp(x: float) -> float:
    try:
        result = float(np.exp(np.clip(x, -500.0, 500.0)))
        return result if np.isfinite(result) else 0.0
    except Exception:
        return 0.0


def _protected_pow(a: float, b: float) -> float:
    try:
        result = float(a ** b)
        return result if np.isfinite(result) else 0.0
    except Exception:
        return 0.0


def _protected_sqrt(x: float) -> float:
    return 0.0 if x < 0.0 else float(np.sqrt(x))


# ---------------------------------------------------------------------------
# Toolbox factory
# ---------------------------------------------------------------------------

def build_gp_toolbox() -> tuple:
    """
    Build DEAP GP toolbox matching Adegboye & Kampouridis (2020) configuration.

    Must be called before run_gp_regression().  Uses a single input variable
    DC_l (the DC event length in bars).

    Returns
    -------
    tuple[base.Toolbox, gp.PrimitiveSet]
        (toolbox, pset) — toolbox is ready for evolution; pset is needed for
        compiling individuals into callable functions.

    Raises
    ------
    ImportError
        If DEAP is not installed.  Install with: pip install deap
    """
    if not _DEAP_AVAILABLE:
        raise ImportError("DEAP is required for GP regression. Install with: pip install deap")

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.renameArguments(ARG0='DC_l')

    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(_protected_div, 2)
    pset.addPrimitive(_protected_pow, 2)
    pset.addPrimitive(np.sin, 1)
    pset.addPrimitive(np.cos, 1)
    pset.addPrimitive(_protected_log, 1)
    pset.addPrimitive(_protected_exp, 1)
    pset.addPrimitive(_protected_sqrt, 1)

    pset.addEphemeralConstant("ERC", lambda: float(np.random.uniform(-10.0, 10.0)))

    if not hasattr(creator, 'CGPFitnessMin'):
        creator.create("CGPFitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, 'CGPIndividual'):
        creator.create("CGPIndividual", gp.PrimitiveTree, fitness=creator.CGPFitnessMin, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.CGPIndividual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=3))

    return toolbox, pset


# ---------------------------------------------------------------------------
# Evolution
# ---------------------------------------------------------------------------

def run_gp_regression(dc_lengths: np.ndarray, os_lengths: np.ndarray, n_generations: int = 37, population_size: int = 500, cx_prob: float = 0.98, mut_prob: float = 0.02, elitism_fraction: float = 0.10, random_seed: int = 42) -> tuple:  # noqa: E501
    """
    Evolve a symbolic regression model predicting OS_l from DC_l.

    Trained exclusively on αDC trends.  Penalises individuals that produce
    only constants (no DC_l dependency), negative predictions, or non-finite
    output.  Secondary selection criterion: smaller tree depth breaks ties.

    Parameters
    ----------
    dc_lengths : np.ndarray
        DC event lengths (bars) for αDC training trends.  Shape (n,).
    os_lengths : np.ndarray
        Corresponding OS event lengths (bars).  Shape (n,).
    n_generations : int
        Number of GP generations.  Paper default: 37.
    population_size : int
        Population size.  Paper default: 500.
    cx_prob : float
        Crossover probability.  Paper default: 0.98.
    mut_prob : float
        Mutation probability.  Paper default: 0.02.
    elitism_fraction : float
        Fraction of best individuals preserved unchanged each generation.
        Paper default: 0.10 (top 10%).
    random_seed : int
        NumPy random seed for reproducibility.

    Returns
    -------
    tuple[callable, float, str]
        (best_func, best_rmse, equation_str)
        best_func    — compiled callable: best_func(dc_l) -> predicted os_l.
        best_rmse    — RMSE of best individual on training αDC trends.
        equation_str — string representation of the best GP tree.

    Raises
    ------
    ValueError
        If fewer than 5 αDC training samples are provided.
    ImportError
        If DEAP is not installed.
    """
    if not _DEAP_AVAILABLE:
        raise ImportError("DEAP is required for GP regression. Install with: pip install deap")
    if len(dc_lengths) < 5:
        raise ValueError(f"run_gp_regression: need at least 5 αDC samples, got {len(dc_lengths)}.")

    np.random.seed(random_seed)
    random.seed(random_seed)
    dc_arr = dc_lengths.astype(float)
    os_arr = os_lengths.astype(float)

    toolbox, pset = build_gp_toolbox()
    elite_size = max(1, int(population_size * elitism_fraction))

    def _evaluate(individual):
        func = toolbox.compile(expr=individual)
        # Penalise pure-constant individuals (no DC_l dependency)
        tree_str = str(individual)
        if 'DC_l' not in tree_str:
            return (float('inf'),)
        preds = []
        for dc_l in dc_arr:
            try:
                val = float(func(dc_l))
                val = val if np.isfinite(val) else 0.0
                val = max(0.0, val)
            except Exception:
                val = 0.0
            preds.append(val)
        rmse = float(np.sqrt(np.mean((os_arr - np.array(preds)) ** 2)))
        return (rmse,)

    toolbox.register("evaluate", _evaluate)

    population = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1, similar=lambda a, b: str(a) == str(b))

    for gen in range(n_generations):
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        hof.update(population)

        elite = tools.selBest(population, elite_size)
        offspring = toolbox.select(population, len(population) - elite_size)
        offspring = algorithms.varAnd(offspring, toolbox, cx_prob, mut_prob)

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        population[:] = elite + offspring

    best = hof[0]
    best_func = toolbox.compile(expr=best)
    best_rmse = float(best.fitness.values[0])

    return best_func, best_rmse, str(best)
