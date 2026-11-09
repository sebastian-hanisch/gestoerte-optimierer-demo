"""Auswertung für das Gitter-Gerüst: ein Lauf je Welt, Zusammenfassung über Zufalls-Seeds (keine Streamlit-Abhängigkeit, nur numpy).

Ein Lauf (`run_world`) zieht Trainings- und Testdaten zu einem Zufalls-Seed, trainiert KQ und SPO+ und misst auf den Testfällen den normierten Regret und den
mittleren quadratischen Fehler der Kosten; das Orakel kennt die mittleren Kosten der Testfälle. `run_seeds` wiederholt das über feste Seeds, `summarize` gibt
Median, Minimum und Maximum (Median über Seeds, nie ein einzelner Seed als Befund).

Herkunft: aus `spo-kuerzester-weg-demo/skw_runs.py` kopiert und von `skw_` nach `gso_` umbenannt (sonst unverändert). Der Lauf mit gestörten Lösungen steht in `gso_fyrun.py`.
"""

from dataclasses import dataclass

import numpy as np

from gso_grid import Grid
from gso_losses import mse_per_arc, normalized_regret, spo_loss
from gso_training import fit_least_squares, predict, spo_plus_loss_mean, train_spo_plus
from gso_world import Dataset, WorldSpec, make_dataset

SNAPSHOT_STEPS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300)


@dataclass(frozen=True)
class RunSpec:
    """Alles, was einen Lauf festlegt."""

    deg: int = 6
    noise: float = 0.0
    n: int = 1000
    seed: int = 7
    p: int = 5
    h: int = 5
    steps: int = 300
    lr: float = 0.05
    l2: float = 0.0
    n_test: int = 1000

    @property
    def world(self) -> WorldSpec:
        return WorldSpec(p=self.p, deg=self.deg, noise=self.noise)


@dataclass
class WorldRun:
    spec: RunSpec
    grid: Grid
    data: Dataset
    coef_kq: np.ndarray
    coef_spo: np.ndarray
    best_step: int
    train_losses: np.ndarray        # SPO+-Trainingsverlust vor jedem Schritt
    snapshots: dict                 # Schritt -> Parameter nach so vielen Schritten
    regret: dict                    # "orakel", "kq", "spo": normierter Regret auf den Testfällen
    mse: dict                       # "kq", "spo": mittlerer quadratischer Fehler je Bogen auf den Testkosten
    curve: dict                     # Schritt -> Testregret des Parameters nach so vielen Schritten
    train_loss_kq: float            # mittlerer SPO+-Trainingsverlust von KQ
    train_loss_spo: float           # mittlerer SPO+-Trainingsverlust des gewählten Parameters
    case_regret: dict               # "kq", "spo", "orakel": Regret je Testfall in Bruchteilen der besten Kosten (Länge n_test)


def run_world(spec: RunSpec, snapshot_steps=SNAPSHOT_STEPS, keep_cases: bool = True) -> WorldRun:
    grid = Grid(spec.h)
    data = make_dataset(spec.world, grid.d, spec.n, spec.n_test, spec.seed)
    coef_kq = fit_least_squares(data.X_train, data.C_train)
    res = train_spo_plus(grid, data.X_train, data.C_train, spec.steps, spec.lr, spec.l2, init=coef_kq, snapshot_steps=snapshot_steps)
    Ct = data.C_test
    chat = {"kq": predict(coef_kq, data.X_test), "spo": predict(res.coef, data.X_test), "orakel": data.M_test}
    regret = {k: normalized_regret(grid, Ct, v) for k, v in chat.items()}
    mse = {k: mse_per_arc(Ct, chat[k]) for k in ("kq", "spo")}
    curve = {s: normalized_regret(grid, Ct, predict(c, data.X_test)) for s, c in res.snapshots.items()}
    _, zstar = grid.solve(Ct)
    case = {k: spo_loss(grid, Ct, v) / zstar for k, v in chat.items()} if keep_cases else {}
    return WorldRun(
        spec, grid, data, coef_kq, res.coef, res.best_step, res.losses, res.snapshots, regret, mse, curve,
        spo_plus_loss_mean(grid, data.X_train, data.C_train, coef_kq), spo_plus_loss_mean(grid, data.X_train, data.C_train, res.coef), case,
    )


def cell_result(run: WorldRun) -> dict:
    """Die Zahlen eines Laufs in Prozent (Regret) und als mittlerer Quadratfehler je Bogen; ohne die großen Felder."""
    return {
        "seed": run.spec.seed,
        "regret_orakel": 100.0 * run.regret["orakel"],
        "regret_kq": 100.0 * run.regret["kq"],
        "regret_spo": 100.0 * run.regret["spo"],
        "mse_kq": run.mse["kq"],
        "mse_spo": run.mse["spo"],
    }


def run_seeds(spec: RunSpec, seeds) -> list[dict]:
    """Läuft über feste Zufalls-Seeds; je Seed ein Wörterbuch aus `cell_result`."""
    from dataclasses import replace

    return [cell_result(run_world(replace(spec, seed=int(s)), snapshot_steps=(), keep_cases=False)) for s in seeds]


def summarize(values) -> dict:
    """Median, Minimum, Maximum einer Liste."""
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def count_better(rows: list[dict], better: str = "regret_spo", than: str = "regret_kq") -> int:
    """In wie vielen Seeds ist `better` kleiner als `than` (streng)?"""
    return int(sum(r[better] < r[than] for r in rows))
