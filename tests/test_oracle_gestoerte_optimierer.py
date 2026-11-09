"""Orakel-Tests (DEMO-PLAYBOOK §12, ORACLE-PLAYBOOK §5): Löser, SPO+ und Teilgradient gegen unabhängige Rechenwege.

1. Der Löser (Rückwärtsrechnung) gegen die Aufzählung der 70 Wege. Die Wege werden in `conftest.independent_paths` rekursiv erzeugt (andere Konstruktion als
   `Grid.paths`), die Kosten je Weg als Summe der Bogenkosten gerechnet; Gleichstände (ganzzahlige Kosten) folgen derselben Regel (erster kleinster in der Reihenfolge R vor D).
2. SPO+ gegen das Brute-Force-Maximum über alle 70 Wege.
3. Teilgradient gegen den Differenzenquotienten in Richtungen ohne Knick (SPO+ ist stückweise linear in ch: der zentrale Differenzenquotient ist dort exakt).
4. Gesamt-Teilgradient des Trainings (Kettenregel über das lineare Modell) gegen den Differenzenquotienten in den Parametern.
5. KQ gegen die Normalengleichungen, SplitMix64 steht in test_rng.py (Python-Ganzzahlreferenz).

Im Scratch-Lauf (nicht im Repo) mit 300 Instanzen für den Löser, 2000 für SPO+ und Teilgradient wiederholt, jeweils ohne Abweichung."""

import numpy as np
import pytest

import gso_losses as L
from conftest import independent_paths, independent_solve
from gso_training import add_intercept, fit_least_squares, subgradient_of_mean_loss


def test_solver_equals_enumeration_on_random_instances(grid5):
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 5.0, size=(60, 40))
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.abs(z - zo).max() < 1e-12
    assert np.array_equal(W, Wo)                      # stetige Kosten: der beste Weg ist eindeutig


def test_solver_equals_enumeration_with_ties_on_integer_costs(grid5):
    rng = np.random.default_rng(1)
    C = rng.integers(0, 3, size=(60, 40)).astype(float)
    W, z = grid5.solve(C)
    Wo, zo = independent_solve(grid5, C)
    assert np.array_equal(z, zo) and np.array_equal(W, Wo)   # auch die gewählte Wahl bei Gleichstand (R vor D) stimmt überein


def test_solver_equals_enumeration_on_smaller_grids(grid3, grid2):
    rng = np.random.default_rng(2)
    for grid in (grid3, grid2):
        C = rng.uniform(0, 5, size=(40, grid.d))
        W, z = grid.solve(C)
        Wo, zo = independent_solve(grid, C)
        assert np.abs(z - zo).max() < 1e-12 and np.array_equal(W, Wo)


def test_independent_enumeration_has_70_paths_and_matches_the_hand_count(grid5):
    paths = independent_paths(grid5)
    assert len(paths) == 70 and all(len(arcs) == 8 for _, arcs in paths)
    keys = ["".join(m).replace("R", "0").replace("D", "1") for m, _ in paths]
    assert keys == sorted(keys) and len(set(keys)) == 70      # lexikographisch mit R vor D, alle verschieden


def test_spo_plus_equals_brute_force_maximum_over_all_paths(grid5):
    rng = np.random.default_rng(3)
    P = grid5.paths()
    C = rng.uniform(0.5, 4, size=(80, 40))
    Chat = C + rng.normal(0, 1.2, size=C.shape)
    _, zstar = independent_solve(grid5, C)
    Wstar, _ = independent_solve(grid5, C)
    brute = np.max((C - 2 * Chat) @ P.T, axis=1) + 2 * np.sum(Chat * Wstar, axis=1) - zstar
    assert np.abs(L.spo_plus_loss(grid5, C, Chat) - brute).max() < 1e-10


def test_spo_loss_equals_brute_force_regret(grid5):
    rng = np.random.default_rng(4)
    P = grid5.paths()
    C = rng.uniform(0.5, 4, size=(80, 40))
    Chat = C + rng.normal(0, 1.0, size=C.shape)
    chosen = np.argmin(Chat @ P.T, axis=1)
    brute = np.sum(C * P[chosen], axis=1) - (C @ P.T).min(axis=1)
    assert np.abs(L.spo_loss(grid5, C, Chat) - brute).max() < 1e-12


def test_subgradient_equals_central_difference_where_there_is_no_kink(grid5):
    rng = np.random.default_rng(5)
    n, eps = 400, 1e-6
    C = rng.uniform(0.5, 4, size=(n, 40))
    Chat = rng.normal(2, 1.2, size=(n, 40))
    V = rng.normal(size=(n, 40))                      # eine Richtung je Fall
    plus = L.spo_plus_loss(grid5, C, Chat + eps * V)
    minus = L.spo_plus_loss(grid5, C, Chat - eps * V)
    numeric = (plus - minus) / (2 * eps)
    analytic = np.sum(L.spo_plus_subgradient(grid5, C, Chat) * V, axis=1)
    # dieselbe Wegwahl vor und nach dem Schritt: kein Knick im Intervall
    same = np.all(grid5.solve(2 * (Chat + eps * V) - C)[0] == grid5.solve(2 * (Chat - eps * V) - C)[0], axis=1)
    assert same.sum() >= 0.95 * n
    assert np.abs(numeric[same] - analytic[same]).max() < 1e-6


def test_training_subgradient_equals_central_difference_in_the_parameters(grid5):
    rng = np.random.default_rng(6)
    n = 120
    X = rng.normal(size=(n, 5))
    C = rng.uniform(0.5, 4, size=(n, 40))
    coef = rng.normal(0.3, 0.2, size=(6, 40))
    V = rng.normal(size=coef.shape)
    Xa = add_intercept(X)
    eps = 1e-7

    def mean_loss(c):
        return float(np.mean(L.spo_plus_loss(grid5, C, Xa @ c)))

    numeric = (mean_loss(coef + eps * V) - mean_loss(coef - eps * V)) / (2 * eps)
    analytic = float(np.sum(subgradient_of_mean_loss(grid5, X, C, coef) * V))
    assert numeric == pytest.approx(analytic, abs=1e-5)


def test_least_squares_equals_the_normal_equations():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(60, 5))
    C = rng.normal(size=(60, 7))
    Xa = add_intercept(X)
    expected = np.linalg.solve(Xa.T @ Xa, Xa.T @ C)
    assert np.abs(fit_least_squares(X, C) - expected).max() < 1e-10


# --- Gestörte Optimierer (gso_perturb): geschlossene Form, Aufzählung, Differenzenquotient -------------------------------------------------------
# Neu in dieser Demo; die Tests oberhalb sind aus spo-kuerzester-weg-demo kopiert (Löser, SPO+, KQ).

import gso_perturb as P  # noqa: E402
from gso_grid import PathSet, two_paths  # noqa: E402
from gso_rng import SplitMix64  # noqa: E402


def test_closed_form_for_two_paths_against_monte_carlo_with_an_independent_generator():
    """Orakel 1: Φ((ĉ₂ − ĉ₁) / (σ √2)) gegen den Anteil gewählter Wege bei Störungen aus numpy.random.default_rng (anderer Zufallsstrom, andere Normalverteilung),
    je 200 000 Ziehungen: Standardfehler höchstens 0,0012, Toleranz 0,006."""
    rng = np.random.default_rng(2024)
    m = 200_000
    for c1, c2, sigma in ((4.0, 3.0, 0.5), (4.0, 3.0, 1.0), (4.0, 3.0, 3.0), (1.0, 6.0, 2.0), (2.5, 2.5, 1.0), (0.0, 8.0, 5.0)):
        z = rng.standard_normal((m, 2))
        chosen_1 = np.mean(c1 + sigma * z[:, 0] < c2 + sigma * z[:, 1])
        assert chosen_1 == pytest.approx(P.two_path_weights(c1, c2, sigma)[0], abs=0.006), (c1, c2, sigma)


def test_perturbed_solver_equals_enumeration_of_the_70_paths_on_the_same_noise(grid5):
    """Orakel 2: dieselben gestörten Kosten, einmal durch die Rückwärtsrechnung, einmal durch die rekursiv erzeugte Aufzählung aller 70 Wege (conftest)."""
    Chat = 0.5 + 2.5 * SplitMix64(21).uniform(30 * 40).reshape(30, 40)
    z = P.draw_noise(SplitMix64(22), 3, 30, 40)
    W = P.perturbed_solutions(grid5, Chat, 1.0, z)
    for m in range(3):
        W_enum, _ = independent_solve(grid5, Chat + 1.0 * z[m])
        assert np.array_equal(W[m], W_enum), m


def test_mean_perturbed_solution_on_the_grid_against_enumeration_with_many_independent_draws(grid5):
    """Orakel 3: E[w] auf dem Gitter für zwei Vorhersagen: Schätzer aus 150 SplitMix64-Störungen gegen die Aufzählung (PathSet über alle 70 Wege) mit 6000 Störungen
    aus numpy.random.default_rng. Beide Schätzer haben je Bogen höchstens Standardfehler 0,5 / sqrt(M); Band: 5 Standardfehler der Differenz."""
    all_paths = PathSet(grid5.paths())
    Chat = 0.5 + 2.5 * SplitMix64(31).uniform(2 * 40).reshape(2, 40)
    m_a, m_b = 150, 6000
    est_a = P.expected_solution(grid5, Chat, 1.0, P.draw_noise(SplitMix64(32), m_a, 2, 40))
    z_b = np.random.default_rng(33).standard_normal((m_b, 2, 40))
    est_b = P.expected_solution(all_paths, Chat, 1.0, z_b)
    tol = 5 * 0.5 * np.sqrt(1 / m_a + 1 / m_b)
    assert np.abs(est_a - est_b).max() < tol


def test_fy_subgradient_equals_the_difference_quotient_of_the_linear_surrogate(grid5):
    """Orakel 4: bei festen Störungen ist ĉ ↦ ĉᵀ w*(c) − (1/M) Σ_m min_w (ĉ + σ z_m)ᵀ w eine Funktion, deren Gradient y − w̄ ist (Ableitung eines Minimums linearer
    Funktionen). Zentraler Differenzenquotient in zufälligen Richtungen, Minimum über alle 70 Wege aus der Aufzählung; nur Richtungen ohne Wegwechsel im Intervall."""
    rng = np.random.default_rng(41)
    paths = grid5.paths()
    C = rng.uniform(0.5, 3, size=(25, 40))
    Chat = rng.uniform(0.5, 3, size=(25, 40))
    z = rng.standard_normal((3, 25, 40))
    sigma, eps = 0.7, 1e-7
    Wstar, _ = grid5.solve(C)
    V = rng.standard_normal((25, 40))

    def surrogate(ch):
        mins = np.mean([np.min((ch + sigma * z[m]) @ paths.T, axis=1) for m in range(3)], axis=0)
        return np.sum(ch * Wstar, axis=1) - mins

    numeric = (surrogate(Chat + eps * V) - surrogate(Chat - eps * V)) / (2 * eps)
    analytic = np.sum(P.fy_subgradient(grid5, Wstar, Chat, sigma, z) * V, axis=1)
    assert np.abs(numeric - analytic).max() < 1e-5


def test_training_gradient_of_fy_through_the_linear_model_equals_the_chain_rule_difference_quotient(grid5):
    """Orakel 5: Kettenregel über das lineare Modell (Xaᵀ · Teilgradient / n) gegen den Differenzenquotienten des Surrogats in den Parametern."""
    rng = np.random.default_rng(51)
    n = 40
    X = rng.normal(size=(n, 5))
    C = rng.uniform(0.5, 3, size=(n, 40))
    coef = rng.normal(2.0, 0.3, size=(6, 40))
    z = rng.standard_normal((2, n, 40))
    sigma, eps = 1.0, 1e-7
    paths = grid5.paths()
    Wstar, _ = grid5.solve(C)
    Xa = add_intercept(X)

    def surrogate_mean(c):
        ch = Xa @ c
        mins = np.mean([np.min((ch + sigma * z[m]) @ paths.T, axis=1) for m in range(2)], axis=0)
        return float(np.mean(np.sum(ch * Wstar, axis=1) - mins))

    V = rng.normal(size=coef.shape)
    numeric = (surrogate_mean(coef + eps * V) - surrogate_mean(coef - eps * V)) / (2 * eps)
    analytic = float(np.sum(Xa.T @ P.fy_subgradient(grid5, Wstar, Xa @ coef, sigma, z) / n * V))
    assert numeric == pytest.approx(analytic, abs=1e-5)


def test_handbeispiel_through_the_two_path_solver_gives_the_hand_gradient():
    """Handbeispiel über die Maschinerie: y = w*(c) mit c = (3, 5) ist (1, 0); Teilgradient bei σ = 1 mit 20 000 Störungen ≈ (0,7602; −0,7602)."""
    solver = two_paths()
    Wstar, _ = solver.solve(np.array([[3.0, 5.0]]))
    z = P.draw_noise(SplitMix64(61), 20_000, 1, 2)
    g = P.fy_subgradient(solver, Wstar, np.array([[4.0, 3.0]]), 1.0, z)[0]
    assert g == pytest.approx((0.7602, -0.7602), abs=0.01)
