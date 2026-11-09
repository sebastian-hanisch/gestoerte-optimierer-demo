"""Gestörte Lösungen von Hand: Handbeispiel (zwei Wege, σ = 0,1 / 1 / 3), Grenzfälle σ → 0 und σ → ∞, Eigenschaften, Randfälle.

Handbeispiel: zwei Wege mit je einem Bogen, wahre Kosten c = (3, 5), Vorhersage ĉ = (4, 3); Weg 1 wird bei Störung gewählt mit Wahrscheinlichkeit
Φ((ĉ₂ − ĉ₁) / (σ √2)) = Φ(−1 / (σ √2)). Werte von Hand (Standardnormalverteilung, Tafelwerte): σ = 1: Φ(−0,7071) = 0,2398; σ = 3: Φ(−0,2357) = 0,4068;
σ = 0,1: Φ(−7,07) = 1,2·10⁻¹² (praktisch 0)."""

import numpy as np
import pytest

import gso_perturb as P
from gso_grid import Grid, two_paths
from gso_rng import SplitMix64


def test_hand_example_weights_sigma_01_1_3():
    w01 = P.two_path_weights(4.0, 3.0, 0.1)
    w1 = P.two_path_weights(4.0, 3.0, 1.0)
    w3 = P.two_path_weights(4.0, 3.0, 3.0)
    assert w01[0] == pytest.approx(0.0, abs=1e-10) and w01[1] == pytest.approx(1.0)
    assert w1 == pytest.approx((0.2398, 0.7602), abs=5e-5)
    assert w3 == pytest.approx((0.4068, 0.5932), abs=5e-5)


def test_hand_example_fy_subgradient_is_y_minus_mean_solution():
    """y = w*(c) = (1, 0); Teilgradient y − E[w]: σ = 0,1 → (1, −1); σ = 1 → (0,7602; −0,7602); σ = 3 → (0,5932; −0,5932)."""
    y = np.array([1.0, 0.0])
    for sigma, expected in ((0.1, 1.0), (1.0, 0.7602), (3.0, 0.5932)):
        p1, p2 = P.two_path_weights(4.0, 3.0, sigma)
        g = y - np.array([p1, p2])
        assert g == pytest.approx((expected, -expected), abs=5e-5), sigma


def test_stronger_noise_means_smaller_but_never_zero_gradient():
    gs = [1.0 - P.two_path_weights(4.0, 3.0, s)[0] for s in (0.5, 1.0, 2.0, 3.0, 10.0, 100.0)]
    assert all(a > b for a, b in zip(gs, gs[1:]))
    assert all(g > 0.5 for g in gs)          # E[w1] < 0,5, weil Weg 2 vorhergesagt billiger ist: der Teilgradient bleibt über 0,5


def test_sigma_zero_is_the_hard_solver_and_sigma_infinity_is_one_half():
    assert P.two_path_weights(4.0, 3.0, 0.0) == (0.0, 1.0) and P.two_path_weights(3.0, 4.0, 0.0) == (1.0, 0.0)
    assert P.two_path_weights(3.0, 3.0, 0.0) == (1.0, 0.0)                       # Gleichstand: Weg 1, wie PathSet
    assert P.two_path_weights(4.0, 3.0, 1e9) == pytest.approx((0.5, 0.5), abs=1e-6)


def test_equal_predictions_give_one_half_for_every_sigma():
    for sigma in (0.1, 1.0, 5.0):
        assert P.two_path_weights(2.0, 2.0, sigma) == pytest.approx((0.5, 0.5))


def test_weights_sum_to_one_and_negative_sigma_is_rejected():
    for c1, c2, s in ((0.0, 8.0, 0.3), (5.0, 1.0, 2.0), (3.5, 3.5, 0.7)):
        assert sum(P.two_path_weights(c1, c2, s)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        P.two_path_weights(1.0, 2.0, -0.1)


def test_scaling_sigma_and_prediction_together_does_not_change_the_choice(grid5):
    """w(λ ĉ + λ σ z) = w(ĉ + σ z) für λ > 0 (der Löser minimiert ein Vielfaches derselben Kosten)."""
    rng = SplitMix64(5)
    Chat = 0.5 + 2.5 * rng.uniform(20 * 40).reshape(20, 40)
    z = P.draw_noise(SplitMix64(6), 3, 20, 40)
    for lam in (0.01, 3.0, 250.0):
        a = P.perturbed_solutions(grid5, Chat, 0.8, z)
        b = P.perturbed_solutions(grid5, lam * Chat, lam * 0.8, z)
        assert np.array_equal(a, b), lam


def test_perturbed_solutions_are_paths_and_the_mean_sums_to_the_path_length(grid5):
    Chat = 0.5 + 2.5 * SplitMix64(1).uniform(8 * 40).reshape(8, 40)
    z = P.draw_noise(SplitMix64(2), 4, 8, 40)
    W = P.perturbed_solutions(grid5, Chat, 1.0, z)
    assert W.shape == (4, 8, 40) and set(np.unique(W)) <= {0.0, 1.0}
    assert np.all(W.sum(axis=2) == 8)                                        # jede Lösung ist ein Weg mit 8 Bögen
    mean = P.expected_solution(grid5, Chat, 1.0, z)
    assert mean.sum(axis=1) == pytest.approx(np.full(8, 8.0))               # der Mittelwert hat dieselbe Bogensumme (Gewichte von Wegen)
    assert mean.min() >= 0 and mean.max() <= 1


def test_gradient_is_zero_when_y_equals_the_mean_solution(grid5):
    """Gradient null bei y = E[w]: der Teilgradient ist y − E[w] (Sonderfall: kleine σ, dann ist E[w] = w(ĉ) und y = w(ĉ))."""
    C = 0.5 + 2.5 * SplitMix64(3).uniform(6 * 40).reshape(6, 40)
    z = P.draw_noise(SplitMix64(4), 3, 6, 40)
    mean = P.expected_solution(grid5, C, 1.0, z)
    assert P.fy_subgradient(grid5, mean, C, 1.0, z) == pytest.approx(np.zeros((6, 40)), abs=1e-12)
    Wstar, _ = grid5.solve(C)
    assert P.fy_subgradient(grid5, Wstar, C, 1e-9, z) == pytest.approx(np.zeros((6, 40)), abs=1e-12)    # σ → 0: harter Löser, y = w(c) = w(ĉ)


def test_m_equal_one_and_shape_errors(grid5):
    C = 0.5 + 2.5 * SplitMix64(7).uniform(2 * 40).reshape(2, 40)
    z = P.draw_noise(SplitMix64(8), 1, 2, 40)
    W = P.perturbed_solutions(grid5, C, 1.0, z)
    assert W.shape == (1, 2, 40)
    assert np.array_equal(P.expected_solution(grid5, C, 1.0, z), W[0])
    with pytest.raises(ValueError):
        P.perturbed_solutions(grid5, C[:1], 1.0, z)


def test_sigma_zero_equals_the_hard_solution_in_every_draw(grid5):
    C = 0.5 + 2.5 * SplitMix64(9).uniform(5 * 40).reshape(5, 40)
    z = P.draw_noise(SplitMix64(10), 4, 5, 40)
    W = P.perturbed_solutions(grid5, C, 0.0, z)
    hard, _ = grid5.solve(C)
    assert all(np.array_equal(W[m], hard) for m in range(4))


def test_two_path_solver_through_the_perturbation_machinery():
    """Dieselbe Rechnung wie die geschlossene Form, aber über `perturbed_solutions` mit dem PathSet-Löser: ein Strom, M = 4000 Störungen (Band ± 4 Standardfehler)."""
    solver = two_paths()
    m = 4000
    z = P.draw_noise(SplitMix64(11), m, 1, 2)
    for sigma in (0.3, 1.0, 3.0):
        est = P.expected_solution(solver, np.array([[4.0, 3.0]]), sigma, z)[0]
        exact = P.two_path_weights(4.0, 3.0, sigma)
        tol = 4 * np.sqrt(exact[0] * exact[1] / m) + 1e-3
        assert est == pytest.approx(exact, abs=tol), sigma


def test_noise_streams_are_reproducible_and_depend_on_the_seed():
    a, b = P.draw_noise(SplitMix64(1), 2, 3, 4), P.draw_noise(SplitMix64(1), 2, 3, 4)
    assert np.array_equal(a, b) and not np.array_equal(a, P.draw_noise(SplitMix64(2), 2, 3, 4))
    big = P.draw_noise(SplitMix64(3), 1, 1, 200_000).ravel()
    assert abs(big.mean()) < 0.02 and abs(big.std() - 1.0) < 0.02       # standardnormal (200 000 Werte: Standardfehler 0,002)


def test_train_fy_is_deterministic_and_needs_at_least_one_perturbation(grid5):
    X = SplitMix64(1).normal(30 * 5).reshape(30, 5)
    C = 0.5 + 2.5 * SplitMix64(2).uniform(30 * 40).reshape(30, 40)
    a = P.train_fy(grid5, X, C, 1.0, 2, 7, steps=5)
    b = P.train_fy(grid5, X, C, 1.0, 2, 7, steps=5)
    assert np.array_equal(a.coef, b.coef)
    assert not np.array_equal(a.coef, P.train_fy(grid5, X, C, 1.0, 2, 8, steps=5).coef)
    with pytest.raises(ValueError):
        P.train_fy(grid5, X, C, 1.0, 0, 7, steps=1)


def test_adam_step_by_hand():
    """Ein Adam-Schritt bei Gradient g = 2 (konstant) auf coef = 0: m̂ = g, v̂ = g², Schritt = lr · g / (|g| + ε) ≈ lr: coef = −0,05."""
    r = P.adam_descend(lambda coef, t: np.array([2.0]), np.array([0.0]), 1, lr=0.05, snapshot_steps=(0, 1))
    assert r.coef == pytest.approx([-0.05], abs=1e-8)
    assert r.snapshots[0] == pytest.approx([0.0]) and r.snapshots[1] == pytest.approx([-0.05], abs=1e-8)
    r2 = P.adam_descend(lambda coef, t: np.array([2.0]), np.array([0.0]), 2, lr=0.05)
    assert r2.coef == pytest.approx([-0.10], abs=1e-6)       # konstanter Gradient: jeder Schritt ≈ lr
