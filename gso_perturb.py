"""Gestörte Optimierer: Gauß-Störung, gemittelte gestörte Lösung und Teilgradient (keine Streamlit-Abhängigkeit, nur numpy).

Idee (Berthet u. a., NeurIPS 2020, arXiv 2002.08676; nur Abstract-Ebene gelesen): Statt den Löser auf der Vorhersage ĉ zu fragen, wird ĉ mit Rauschen
gestört, der Löser mehrfach gefragt und die Lösungen werden gemittelt. Der Mittelwert E[w(ĉ + σ Z)] ist glatt in ĉ und braucht den Löser nur als Black Box.

Hier steht nur das, was selbst nachgerechnet ist (Handbeispiel, Differenzenquotient, Aufzählung in den Tests):

    gemittelte gestörte Lösung   w̄(ĉ) = (1/M) Σ_m w(ĉ + σ z_m),   z_m ~ N(0, I) (Gauß-Störung, Box-Muller aus SplitMix64)
    Teilgradient (nach ĉ)        w*(c) − w̄(ĉ)        (optimale Lösung minus gemittelte gestörte Lösung; Minimierung)

Der Teilgradient ist der Gradient von ĉᵀw*(c) − (1/M) Σ_m min_w (ĉ + σ z_m)ᵀw nach ĉ bei festen z_m (Satz über die Ableitung eines Minimums linearer Funktionen);
das ist nur die Rechenregel hinter der Handformel. Die genaue Form des Fenchel-Young-Verlustes der Quelle (Referenzverteilung, Normierung) ist NICHT geprüft und
hier nicht nachgebaut: das Training benutzt nur den Teilgradienten, keinen Verlustwert (es gibt deshalb auch keine Auswahl nach kleinstem Trainingsverlust;
der Parameter nach der letzten Runde zählt).

Zufall nur über `gso_rng.SplitMix64`; die Ströme der Störungen sind aus (Zufalls-Seed, Etikett) abgeleitet und hängen nicht an Plattform oder numpy-Version
(nur der Schritt zu Normalverteilten über `log` und `cos` der C-Bibliothek ist plattformabhängig, letzte Stelle).
"""

import math
from dataclasses import dataclass, field

import numpy as np

from gso_rng import SplitMix64, derive_seed
from gso_training import ADAM_B1, ADAM_B2, ADAM_EPS, add_intercept, fit_least_squares

TAG_PERTURB, TAG_HAND, TAG_BIAS, TAG_SHOW = 11, 12, 13, 14   # Etiketten der Teilströme (die Welt benutzt 1 bis 3)


def draw_noise(rng: SplitMix64, m: int, n: int, d: int) -> np.ndarray:
    """m x n x d Gauß-Störungen (Standardnormal, Box-Muller aus zwei Gleichverteilten); der Strom rückt weiter."""
    return rng.normal(m * n * d).reshape(m, n, d)


def perturbed_solutions(solver, Chat: np.ndarray, sigma: float, z: np.ndarray) -> np.ndarray:
    """Lösungen auf den gestörten Kosten ĉ + σ z_m: Array m x n x d (z hat die Form m x n x d, Chat n x d). Alle m·n Instanzen gehen in EINEN Löseraufruf."""
    Chat = np.atleast_2d(np.asarray(Chat, dtype=float))
    m, n, d = z.shape
    if Chat.shape != (n, d):
        raise ValueError(f"ĉ hat die Form {Chat.shape}, die Störungen verlangen {(n, d)}")
    costs = Chat[None, :, :] + float(sigma) * z
    W, _ = solver.solve(costs.reshape(m * n, d))
    return W.reshape(m, n, d)


def expected_solution(solver, Chat: np.ndarray, sigma: float, z: np.ndarray) -> np.ndarray:
    """Gemittelte gestörte Lösung w̄(ĉ) je Fall (n x d); jede Zeile summiert über die Bögen eines Weges, ist also ein Gewichtsvektor mit Summe 'Weglänge in Bögen'."""
    return perturbed_solutions(solver, Chat, sigma, z).mean(axis=0)


def fy_subgradient(solver, Wstar: np.ndarray, Chat: np.ndarray, sigma: float, z: np.ndarray) -> np.ndarray:
    """Teilgradient nach ĉ je Fall (n x d): optimale Lösung w*(c) (`Wstar`, vorab gelöst) minus gemittelte gestörte Lösung."""
    return np.asarray(Wstar, dtype=float) - expected_solution(solver, Chat, sigma, z)


# --- Handbeispiel: zwei Wege, geschlossene Form --------------------------------------------------------------------------------------------------

def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_path_weights(c1hat: float, c2hat: float, sigma: float) -> tuple[float, float]:
    """Gewicht (Wahrscheinlichkeit) beider Wege bei Gauß-Störung σ auf beiden Kosten: Weg 1 wird gewählt, wenn ĉ₁ + σ z₁ < ĉ₂ + σ z₂, also σ (z₁ − z₂) < ĉ₂ − ĉ₁;
    z₁ − z₂ ist N(0, 2)-verteilt: P(Weg 1) = Φ((ĉ₂ − ĉ₁) / (σ √2)). Bei σ = 0 gilt der harte Löser (Gleichstand: Weg 1, wie `PathSet`)."""
    if sigma < 0:
        raise ValueError("σ darf nicht negativ sein")
    if sigma == 0:
        return (1.0, 0.0) if c1hat <= c2hat else (0.0, 1.0)
    p1 = normal_cdf((c2hat - c1hat) / (sigma * math.sqrt(2.0)))
    return p1, 1.0 - p1


# --- Training --------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class FyResult:
    coef: np.ndarray                                 # Parameter nach der letzten Runde
    snapshots: dict = field(default_factory=dict)   # Schritt -> Parameter nach so vielen Schritten (0 = Start)


def adam_descend(grad_fn, coef0: np.ndarray, steps: int, lr: float = 0.05, snapshot_steps=()) -> FyResult:
    """Vollbatch-Adam (Kingma und Ba) mit Verzerrungskorrektur auf einer reinen Gradientenfunktion `grad_fn(coef, t) -> Gradient nach coef` (t = 1, 2, ...).
    Kein Verlustwert, keine Auswahl: zurückgegeben wird der Parameter nach `steps` Schritten und die gewünschten Zwischenstände (0 = Start)."""
    coef = np.array(coef0, dtype=float)
    mom = np.zeros_like(coef)
    var = np.zeros_like(coef)
    want = {int(s) for s in snapshot_steps}
    snaps = {0: coef.copy()} if 0 in want else {}
    for t in range(1, steps + 1):
        grad = grad_fn(coef, t)
        mom = ADAM_B1 * mom + (1 - ADAM_B1) * grad
        var = ADAM_B2 * var + (1 - ADAM_B2) * grad * grad
        coef = coef - lr * (mom / (1 - ADAM_B1 ** t)) / (np.sqrt(var / (1 - ADAM_B2 ** t)) + ADAM_EPS)
        if t in want:
            snaps[t] = coef.copy()
    return FyResult(coef, snaps)


def train_fy(solver, X: np.ndarray, C: np.ndarray, sigma: float, m: int, seed: int, steps: int = 300, lr: float = 0.05, init=None, snapshot_steps=()) -> FyResult:
    """Lineares Modell mit dem Teilgradienten der gemittelten gestörten Lösung: Start bei KQ (oder `init`), Adam, Vollbatch.

    Je Runde: m Gauß-Störungen je Fall und Bogen aus dem Strom (Zufalls-Seed, TAG_PERTURB), ein Löseraufruf auf allen m·n gestörten Instanzen.
    Der beste Weg w*(c) wird einmal vorab gelöst (n Instanzen). Gelöste Instanzen in der Schleife: steps · m · n."""
    if m < 1:
        raise ValueError("mindestens eine Störung je Schritt")
    Xa = add_intercept(X)
    C = np.asarray(C, dtype=float)
    n, d = C.shape
    Wstar, _ = solver.solve(C)
    rng = SplitMix64(derive_seed(seed, TAG_PERTURB))
    coef0 = fit_least_squares(X, C) if init is None else np.array(init, dtype=float)

    def grad_fn(coef, _t):
        z = draw_noise(rng, m, n, d)
        return Xa.T @ fy_subgradient(solver, Wstar, Xa @ coef, sigma, z) / n

    return adam_descend(grad_fn, coef0, steps, lr, snapshot_steps)
