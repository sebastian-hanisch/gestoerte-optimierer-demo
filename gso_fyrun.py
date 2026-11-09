"""Läufe dieser Demo (keine Streamlit-Abhängigkeit, nur numpy): KQ, SPO+ und das Training mit gestörten Lösungen auf einer Welt, dazu die Messung des Teilgradienten
an der wahren Vorhersage (Glättungsverzerrung).

Alle Verfahren starten bei der KQ-Lösung und laufen 300 Vollbatch-Adam-Schritte; verglichen wird der Parameter NACH t Schritten (nicht der mit dem kleinsten
Trainingsverlust: das Training mit gestörten Lösungen hat keinen Verlustwert, siehe `gso_perturb`). Das SPO+-Training ist das aus `gso_training`
(Teilgradient 2 (w*(c) − w(2ĉ − c))), nur mit anderer Auswahl des Parameters als in `spo-kuerzester-weg-demo`.
"""

from dataclasses import dataclass, field

import numpy as np

from gso_effort import CountingSolver, instances_in_loop, instances_upfront
from gso_grid import Grid
from gso_losses import normalized_regret
from gso_perturb import TAG_BIAS, TAG_PERTURB, draw_noise, train_fy  # noqa: F401  (TAG_PERTURB: Verweis für die Tests)
from gso_rng import SplitMix64, derive_seed
from gso_training import fit_least_squares, predict, train_spo_plus
from gso_world import WorldSpec, draw_weights, make_dataset, mean_costs

GRID_H, N_FEATURES, N_TEST, STEPS, LR, DEG = 5, 5, 1000, 300, 0.05, 6
CHECK_STEPS = (0, 5, 10, 25, 50, 100, 300)
CURVE_STEPS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300)


@dataclass
class MethodRun:
    """Ein Verfahren auf einer Welt: Regret (Anteil, nicht Prozent) nach den gewünschten Schritten und gezählte gelöste Instanzen."""

    name: str
    regret: dict                       # Schritt -> normierter Regret auf den Testfällen
    coef: np.ndarray                   # Parameter nach der letzten Runde
    loop_instances: int = 0            # gelöste Instanzen in den Trainingsrunden
    upfront_instances: int = 0         # einmalige Vorabrechnung der besten Wege


@dataclass
class WorldMethods:
    spec: WorldSpec
    seed: int
    n: int
    grid: Grid
    data: object
    coef_kq: np.ndarray
    regret_oracle: float
    regret_kq: float
    spo: MethodRun
    fy: dict = field(default_factory=dict)     # (σ, M) -> MethodRun


def _regrets(grid, data, snapshots: dict) -> dict:
    return {s: normalized_regret(grid, data.C_test, predict(c, data.X_test)) for s, c in snapshots.items()}


def run_spo_plus(grid: Grid, data, n: int, steps: int = STEPS, lr: float = LR, snapshot_steps=CHECK_STEPS) -> MethodRun:
    """SPO+ mit gezählten Löseraufrufen; Regret nach den gewünschten Schritten (Parameter nach t Schritten)."""
    counter = CountingSolver(grid)
    res = train_spo_plus(counter, data.X_train, data.C_train, steps, lr, snapshot_steps=tuple(snapshot_steps))
    snaps = dict(res.snapshots)
    # der Parameter nach der letzten Runde (nicht der mit kleinstem Trainingsverlust) steht in den Schnappschüssen, wenn `steps` darin liegt
    if steps not in snaps:
        raise ValueError("`steps` muss in `snapshot_steps` stehen")
    return MethodRun("SPO+", _regrets(grid, data, snaps), snaps[steps], counter.rows - n, n)


def run_fy(grid: Grid, data, n: int, sigma: float, m: int, seed: int, steps: int = STEPS, lr: float = LR, snapshot_steps=CHECK_STEPS) -> MethodRun:
    """Training mit gestörten Lösungen (σ, M) mit gezählten Löseraufrufen."""
    counter = CountingSolver(grid)
    res = train_fy(counter, data.X_train, data.C_train, sigma, m, seed, steps, lr, snapshot_steps=tuple(snapshot_steps))
    return MethodRun(f"FY σ = {sigma}, M = {m}", _regrets(grid, data, res.snapshots), res.coef, counter.rows - n, n)


def run_methods(noise: float, seed: int, configs=((1.0, 1),), n: int = 1000, steps: int = STEPS, snapshot_steps=CHECK_STEPS, with_spo: bool = True) -> WorldMethods:
    """Eine Welt (Grad 6, 5 Merkmale, 5 x 5 Gitter): Orakel, KQ, SPO+ und je Eintrag in `configs` = (σ, M) ein Lauf mit gestörten Lösungen."""
    spec = WorldSpec(p=N_FEATURES, deg=DEG, noise=noise)
    grid = Grid(GRID_H)
    data = make_dataset(spec, grid.d, n, N_TEST, seed)
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    r_kq = normalized_regret(grid, data.C_test, predict(coef_kq, data.X_test))
    r_or = normalized_regret(grid, data.C_test, data.M_test)
    spo = run_spo_plus(grid, data, n, steps, snapshot_steps=snapshot_steps) if with_spo else None
    fy = {(float(s), int(m)): run_fy(grid, data, n, float(s), int(m), seed, steps, snapshot_steps=snapshot_steps) for s, m in configs}
    return WorldMethods(spec, seed, n, grid, data, coef_kq, r_or, r_kq, spo, fy)


def expected_counts(n: int, steps: int, m: int) -> dict:
    """Die gezählten Instanzen als Formel (zum Gegenprüfen der Zählung): SPO+ und FY in der Schleife, vorab."""
    return {"spo+": instances_in_loop("spo+", n, steps), "fy": instances_in_loop("fy", n, steps, m), "upfront": instances_upfront("fy", n)}


# --- Teilgradient an der wahren Vorhersage (Glättungsverzerrung) -------------------------------------------------------------------------------------

def bias_ratios(noise: float, seed: int, sigmas=(0.1, 0.3, 1.0, 3.0), nx: int = 60, ndraw: int = 400) -> dict:
    """Wie systematisch ist der Teilgradient, wenn die Vorhersage die wahren bedingten Mittelkosten trifft?

    Für nx Merkmalsvektoren x (Welt Grad 6, Seed `seed`) mit ĉ = m(x) werden ndraw Kostenziehungen c = m(x) · ε gezogen (ε ~ U[1 − e, 1 + e]); je Ziehung ein Teilgradient
    (SPO+: 2 (w*(c) − w(2ĉ − c)); gestörte Lösung mit M = 1: w*(c) − w(ĉ + σ z)). Maß je x: |Mittel über die Ziehungen| / (Mittel der |Teilgradient|):
    0 = im Mittel kein Gradient (stationär an der Wahrheit), 1 = alle Ziehungen zeigen in dieselbe Richtung. Rückgabe: Median über x, je Verfahren.

    Bei e = 0 ist c = ĉ in jeder Ziehung: der harte SPO+-Teilgradient ist dann exakt 0, der gestörte nicht (Glättungsverzerrung: die Störung verschmiert die Lösung,
    auch wenn die Vorhersage stimmt)."""
    spec = WorldSpec(p=N_FEATURES, deg=DEG, noise=noise)
    grid = Grid(GRID_H)
    d = grid.d
    B = draw_weights(d, spec, seed)
    rng = SplitMix64(derive_seed(seed, TAG_BIAS))
    X = rng.normal(nx * spec.p).reshape(nx, spec.p)
    mean = mean_costs(B, X, spec)                                                       # nx x d
    if noise > 0:
        eps = 1.0 - noise + 2.0 * noise * rng.uniform(nx * ndraw * d).reshape(nx, ndraw, d)
    else:
        eps = np.ones((nx, ndraw, d))
    C = (mean[:, None, :] * eps).reshape(nx * ndraw, d)
    Chat = np.repeat(mean[:, None, :], ndraw, axis=1).reshape(nx * ndraw, d)
    Wstar, _ = grid.solve(C)

    def ratio(G):
        G = G.reshape(nx, ndraw, d)
        num = np.linalg.norm(G.mean(axis=1), axis=1)
        den = np.linalg.norm(G, axis=2).mean(axis=1)
        return float(np.median(num / np.maximum(den, 1e-12)))

    out = {"SPO+": ratio(2.0 * (Wstar - grid.solve(2.0 * Chat - C)[0]))}
    for s in sigmas:
        z = rng.normal(nx * ndraw * d).reshape(nx * ndraw, d)
        out[f"FY σ = {s}"] = ratio(Wstar - grid.solve(Chat + float(s) * z)[0])
    return out
