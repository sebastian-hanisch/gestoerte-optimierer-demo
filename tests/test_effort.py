"""Löseraufwand: der zählende Löser zählt Zeilen, die Zählung stimmt mit den Formeln überein; die Zeitmessung ist live und nie festgeschrieben.

Mini-Instanz von Hand: n = 3 Fälle, 10 Schritte, M = 5 Störungen: 10 · 5 · 3 = 150 gelöste Instanzen in der Schleife und einmalig 3 vorab; SPO+: 10 · 3 = 30."""

import numpy as np
import pytest

import gso_effort as E
from gso_fyrun import expected_counts, run_fy, run_spo_plus
from gso_grid import Grid
from gso_perturb import train_fy
from gso_rng import SplitMix64
from gso_training import train_spo_plus
from gso_world import WorldSpec, make_dataset


@pytest.fixture(scope="module")
def mini():
    grid = Grid(5)
    data = make_dataset(WorldSpec(noise=0.5), grid.d, 3, 10, 4)
    return grid, data


def test_counting_solver_counts_rows_and_calls(grid5):
    counter = E.CountingSolver(grid5)
    counter.solve(np.ones((7, 40)))
    counter.solve(np.ones(40))                     # eine Zeile als Vektor
    assert (counter.rows, counter.calls) == (8, 2) and counter.d == 40
    W, z = counter.solve(np.ones((2, 40)))
    assert W.shape == (2, 40) and z == pytest.approx([8.0, 8.0])


def test_fy_loop_count_by_hand_150_plus_3_upfront(mini):
    grid, data = mini
    counter = E.CountingSolver(grid)
    train_fy(counter, data.X_train, data.C_train, 1.0, 5, 0, steps=10)
    assert counter.rows == 3 + 10 * 5 * 3 == 153 and counter.calls == 1 + 10


def test_spo_plus_loop_count_by_hand_30_plus_3_upfront(mini):
    grid, data = mini
    counter = E.CountingSolver(grid)
    train_spo_plus(counter, data.X_train, data.C_train, steps=10)
    assert counter.rows == 3 + 10 * 3 == 33


def test_formulas_for_the_documented_cell():
    """n = 1000, 300 Schritte: SPO+ und M = 1: 300 000; M = 5: 1 500 000; Black-Box (zwei Aufrufe): 600 000; KQ: 0."""
    assert E.instances_in_loop("spo+", 1000, 300) == 300_000
    assert E.instances_in_loop("fy", 1000, 300, 1) == 300_000
    assert E.instances_in_loop("fy", 1000, 300, 5) == 1_500_000
    assert E.instances_in_loop("bb", 1000, 300) == 600_000
    assert E.instances_in_loop("kq", 1000, 300) == 0
    assert E.instances_in_loop("spo+", 1000, 25) == 25_000
    assert E.instances_upfront("fy", 1000) == 1000 and E.instances_upfront("kq", 1000) == 0


def test_counted_runs_equal_the_formulas(mini):
    grid, data = mini
    spo = run_spo_plus(grid, data, 3, steps=10, snapshot_steps=(0, 10))
    fy = run_fy(grid, data, 3, 1.0, 5, 0, steps=10, snapshot_steps=(0, 10))
    exp = expected_counts(3, 10, 5)
    assert (spo.loop_instances, spo.upfront_instances) == (exp["spo+"], exp["upfront"]) == (30, 3)
    assert (fy.loop_instances, fy.upfront_instances) == (exp["fy"], exp["upfront"]) == (150, 3)


def test_steps_must_be_snapshot(mini):
    grid, data = mini
    with pytest.raises(ValueError):
        run_spo_plus(grid, data, 3, steps=10, snapshot_steps=(0, 5))


def test_timing_uses_the_clock_and_takes_the_best_of_the_repeats():
    ticks = iter([0.0, 5.0, 10.0, 10.5, 20.0, 29.0, 30.0, 30.25])        # vier Läufe: Dauer 5, 0,5, 9, 0,25 -> kleinste 0,25
    assert E.time_solver(5, n_instances=10, repeats=4, clock=lambda: next(ticks)) == pytest.approx(0.25)


def test_timing_returns_a_positive_time_and_does_not_depend_on_a_stored_value():
    t = E.time_solver(5, n_instances=50, repeats=2)
    assert 0 < t < 5.0


def test_random_costs_are_reproducible_and_in_range():
    a = E.random_costs(10, 40, 3)
    assert np.array_equal(a, E.random_costs(10, 40, 3)) and a.min() >= 0.5 and a.max() < 3.0
    assert np.array_equal(a.ravel(), 0.5 + 2.5 * SplitMix64(3).uniform(400))
