"""Löseraufwand: zählender Löser und Zeitmessung (keine Streamlit-Abhängigkeit, nur numpy).

Gezählt werden GELÖSTE INSTANZEN (Zeilen einer Kostenmatrix, die an den Löser gehen), nicht Sekunden: die Zahl hängt nicht vom Rechner ab, die Zeit schon.
Zählregel dieser Demo: "in der Schleife" sind alle Instanzen der Trainingsrunden; die einmalige Vorabrechnung der besten Wege w*(c) (n Instanzen) steht getrennt
("vorab"). Die Auswertung auf den Testfällen zählt nicht mit.

Die Zeit je 1000 Instanzen wird nur LIVE gemessen (`time_solver`) und nirgends festgeschrieben.
"""

import time

import numpy as np

from gso_grid import Grid
from gso_rng import SplitMix64


class CountingSolver:
    """Umhüllt einen Löser (`solve(C) -> (W, z)`) und zählt die gelösten Instanzen (Zeilen) und die Aufrufe."""

    def __init__(self, solver):
        self.solver = solver
        self.rows = 0
        self.calls = 0

    @property
    def d(self) -> int:
        return self.solver.d

    def solve(self, C):
        C = np.atleast_2d(np.asarray(C, dtype=float))
        self.rows += C.shape[0]
        self.calls += 1
        return self.solver.solve(C)


def instances_in_loop(method: str, n: int, steps: int, m: int = 1) -> int:
    """Gelöste Instanzen in den Trainingsrunden (ohne die einmalige Vorabrechnung der besten Wege).

    "kq": 0 (geschlossene Lösung, kein Löser). "spo+": ein Aufruf je Runde und Fall (auf 2ĉ − c). "fy": m Aufrufe je Runde und Fall.
    "bb": zwei Aufrufe je Runde und Fall (Interpolation durch den Löser; hier nur als Zahl, nicht trainiert)."""
    per_step = {"kq": 0, "spo+": 1, "fy": m, "bb": 2}[method]
    return steps * n * per_step


def instances_upfront(method: str, n: int) -> int:
    """Einmalige Vorabrechnung der besten Wege w*(c) (n Instanzen); KQ braucht keine, die Black-Box-Zahl hier ebenfalls nicht mitgezählt."""
    return 0 if method in ("kq", "bb") else n


def random_costs(n: int, d: int, seed: int = 0) -> np.ndarray:
    """n Kostenvektoren, gleichverteilt in [0,5; 3], aus dem SplitMix64-Strom (Instanzen der Zeitmessung)."""
    return 0.5 + 2.5 * SplitMix64(seed).uniform(n * d).reshape(n, d)


def time_solver(h: int, n_instances: int = 1000, repeats: int = 5, seed: int = 0, clock=time.perf_counter) -> float:
    """Kleinste gemessene Zeit in Sekunden für `n_instances` Instanzen auf dem h x h Gitter (beste von `repeats` Läufen, der erste Aufruf zählt als Aufwärmlauf).

    Eine Messung auf diesem Rechner in diesem Augenblick; nichts davon wird gespeichert oder festgeschrieben."""
    grid = Grid(h)
    C = random_costs(n_instances, grid.d, seed)
    grid.solve(C[:8])
    best = float("inf")
    for _ in range(repeats):
        t0 = clock()
        grid.solve(C)
        best = min(best, clock() - t0)
    return best
