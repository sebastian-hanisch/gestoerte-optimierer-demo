"""Randfälle, an denen Kernroutinen erfahrungsgemäß brechen (DEMO-PLAYBOOK §12): σ = 0, gleiche Vorhersagen (Gleichstand), M = 1, n = 1 und n = 2 Trainingsfälle,
kleinstes Gitter, negative und sehr große σ, Rand der Regler."""

import numpy as np
import pytest

import gso_evaluation as ev
import gso_perturb as P
from gso_fyrun import bias_ratios, run_methods
from gso_grid import Grid, two_paths
from gso_rng import SplitMix64


def test_sigma_zero_is_hard_and_ties_go_to_path_one():
    assert P.two_path_weights(2.0, 2.0, 0.0) == (1.0, 0.0)
    solver = two_paths()
    z = P.draw_noise(SplitMix64(1), 5, 1, 2)
    assert np.array_equal(P.expected_solution(solver, np.array([[2.0, 2.0]]), 0.0, z), np.array([[1.0, 0.0]]))


def test_equal_predictions_with_noise_split_evenly_in_the_sample_too():
    solver = two_paths()
    z = P.draw_noise(SplitMix64(2), 6000, 1, 2)
    w = P.expected_solution(solver, np.array([[2.0, 2.0]]), 1.0, z)[0]
    assert w == pytest.approx((0.5, 0.5), abs=0.03)          # 6000 Störungen: Standardfehler 0,0065


def test_training_with_one_and_two_cases_runs_and_is_finite():
    grid = Grid(5)
    for n in (1, 2):
        X = SplitMix64(n).normal(n * 5).reshape(n, 5)
        C = 0.5 + 2.5 * SplitMix64(n + 10).uniform(n * 40).reshape(n, 40)
        res = P.train_fy(grid, X, C, 1.0, 1, 3, steps=5)
        assert res.coef.shape == (6, 40) and np.all(np.isfinite(res.coef))


def test_smallest_grid_two_by_two_has_two_paths():
    grid = Grid(2)
    z = P.draw_noise(SplitMix64(3), 3000, 1, grid.d)
    mean = P.expected_solution(grid, np.array([[1.0, 1.0, 1.0, 1.0]]), 1.0, z)[0]
    assert mean.sum() == pytest.approx(2.0) and mean.min() > 0           # beide Wege kommen vor (jeder hat 2 Bögen)
    assert mean[0] == pytest.approx(0.5, abs=0.04)                       # gleiche Kosten: Weg "rechts, unten" (Bogen 1 und 3) und Weg "unten, rechts" gleich häufig; 3000 Störungen, Standardfehler 0,009


def test_extreme_sigma_does_not_break_the_weights():
    for sigma in (1e-12, 1e6):
        w = P.two_path_weights(4.0, 3.0, sigma)
        assert all(0.0 <= v <= 1.0 for v in w) and sum(w) == pytest.approx(1.0)


def test_hand_numbers_at_the_slider_edges():
    for c1, c2 in ((0.0, 0.0), (8.0, 0.0), (0.0, 8.0), (8.0, 8.0), (4.0, 3.0)):
        for s in (0.1, 5.0):
            h = ev.hand_numbers(c1, c2, s)
            assert sum(h["weights"]) == pytest.approx(1.0) and sum(h["mc"]) == pytest.approx(1.0)
            assert abs(h["mc"][0] - h["weights"][0]) < 0.06      # 2000 Störungen: Standardfehler höchstens 0,0112, 5 Standardfehler


def test_no_effect_case_does_not_report_an_effect():
    """σ = 0,1 bei ĉ = (4, 3): die Störung ändert die Wahl nicht, das Verdikt sagt es."""
    kind, text = ev.verdict_hand(ev.hand_numbers(4.0, 3.0, 0.1), 0.1)
    assert kind == "warning" and "zu klein" in text


def test_world_with_one_noise_draw_and_bias_without_noise_has_no_division_by_zero():
    r = bias_ratios(0.0, 0, sigmas=(0.1,), nx=3, ndraw=4)
    assert r["SPO+"] == 0.0 and np.isfinite(r["FY σ = 0.1"])


def test_run_methods_small_n_and_steps_zero_snapshot():
    w = run_methods(0.5, 0, configs=((1.0, 1),), n=5, steps=3, snapshot_steps=(0, 3))
    assert set(w.spo.regret) == {0, 3} and set(w.fy[(1.0, 1)].regret) == {0, 3}
    assert w.spo.regret[0] == pytest.approx(w.regret_kq)          # Schritt 0 ist KQ
    assert w.fy[(1.0, 1)].regret[0] == pytest.approx(w.regret_kq)
