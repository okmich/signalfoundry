import pytest

from okmich_quant_research.features.hmm_screener import ParetoStatus, classify_pareto


def test_classify_pareto_empty_input() -> None:
    assert classify_pareto([], honesty_trap_rate=0.4) == []


def test_classify_pareto_single_measurement_is_keeper() -> None:
    out = classify_pareto([(0.5, 0.2)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.KEEPER]


def test_classify_pareto_single_trap_when_honesty_exceeds_threshold() -> None:
    out = classify_pareto([(0.5, 0.6)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.TRAP]


def test_classify_pareto_strictly_dominated_subset_marked_dominated() -> None:
    # Two non-trap points; second has higher sep AND lower honesty -> dominates first.
    out = classify_pareto([(0.3, 0.2), (0.5, 0.1)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.DOMINATED, ParetoStatus.KEEPER]


def test_classify_pareto_pareto_frontier_with_three_points() -> None:
    # Three non-trap points on a tradeoff:
    # A: (0.1, 0.05) - low sep, very honest
    # B: (0.3, 0.20) - mid sep, mid honest
    # C: (0.5, 0.30) - high sep, less honest
    # None dominates any other strictly.
    out = classify_pareto([(0.1, 0.05), (0.3, 0.20), (0.5, 0.30)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.KEEPER, ParetoStatus.KEEPER, ParetoStatus.KEEPER]


def test_classify_pareto_trap_supersedes_pareto_optimal() -> None:
    # The (0.7, 0.5) point would Pareto-dominate (0.3, 0.2) on sep,
    # but its honesty is above the trap rate, so it's classified TRAP regardless.
    out = classify_pareto([(0.3, 0.2), (0.7, 0.5)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.KEEPER, ParetoStatus.TRAP]


def test_classify_pareto_trap_points_excluded_from_keeper_dominance_check() -> None:
    # If A dominates B but A is a trap, B should remain a keeper (not dominated by a trap).
    # A: (0.9, 0.6) - trap
    # B: (0.3, 0.2) - non-trap, no other non-trap dominates it
    out = classify_pareto([(0.9, 0.6), (0.3, 0.2)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.TRAP, ParetoStatus.KEEPER]


def test_classify_pareto_ties_remain_keepers() -> None:
    # Identical points are not strictly dominated by each other.
    out = classify_pareto([(0.5, 0.2), (0.5, 0.2)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.KEEPER, ParetoStatus.KEEPER]


def test_classify_pareto_threshold_at_exact_boundary() -> None:
    # Honesty exactly equal to trap_rate is NOT a trap (strict greater-than rule).
    out = classify_pareto([(0.5, 0.4)], honesty_trap_rate=0.4)
    assert out == [ParetoStatus.KEEPER]
