"""Kennzahlen, Verdikt-Logik und Tabellen für die App (keine Streamlit-Abhängigkeit): Handbeispiel, Lauf einer Welt, vorgerechnete Bänder, Löseraufwand.

Verdikte sind reine Funktionen `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede Zahl im Text an handgesetzten Werten.
Zahlen erscheinen mit Dezimalkomma (`de`), große Ganzzahlen mit Leerzeichen als Tausendertrenner (`de_int`).
"""

from dataclasses import dataclass

import numpy as np

import gso_bands as B
import gso_constants as C
from gso_effort import instances_in_loop
from gso_grid import two_paths
from gso_perturb import TAG_HAND, TAG_SHOW, draw_noise, expected_solution, perturbed_solutions, two_path_weights
from gso_rng import SplitMix64, derive_seed
from gso_training import predict


def _shown(value: float, digits: int = 1) -> float:
    """Der Wert so, wie er mit `digits` Stellen angezeigt wird (Vergleiche stimmen mit dem überein, was die Leser sehen)."""
    return float(f"{float(value):.{digits}f}")


def de(value: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma; negative Nullen (-0,0) erscheinen als 0,0."""
    text = f"{float(value):.{digits}f}"
    if float(text) == 0:
        text = text.lstrip("-")
    return text.replace(".", ",")


def de_int(value: int) -> str:
    """Ganzzahl mit Leerzeichen als Tausendertrenner (300 000)."""
    return f"{int(value):,}".replace(",", " ")


def sig(value: float) -> str:
    """Stärke σ ohne überflüssige Nullen: 0,1 / 0,3 / 1 / 3."""
    text = de(value, 1)
    return text[:-2] if text.endswith(",0") else text


@dataclass(frozen=True)
class Settings:
    noise: float = C.DEFAULT_NOISE
    seed: int = C.DEFAULT_SEED
    sigma: float = C.DEFAULT_SIGMA
    m: int = C.DEFAULT_M
    case: int = C.DEFAULT_CASE
    c1: float = C.DEFAULT_C1HAT
    c2: float = C.DEFAULT_C2HAT
    hsigma: float = C.DEFAULT_HSIGMA


# --- Experiment 1: die gestörte Wahl von Hand --------------------------------------------------------------------------------------------------

def hand_numbers(c1hat: float, c2hat: float, sigma: float, draws: int = C.HAND_DRAWS) -> dict:
    """Handbeispiel: zwei Wege mit je einem Bogen, wahre Kosten c = (3, 5), Vorhersage ĉ = (c1hat, c2hat), Störung σ.

    Geschlossene Form (`two_path_weights`) und Stichprobenversuch mit `draws` Störungen aus dem SplitMix64-Strom (fester Zufalls-Seed). Teilgradient y − E[w] mit y = w*(c) = (1, 0)."""
    solver = two_paths()
    c = np.array([C.HAND_C])
    y = solver.solve(c)[0][0]
    chat = np.array([[c1hat, c2hat]])
    hard = solver.solve(chat)[0][0]
    p1, p2 = two_path_weights(c1hat, c2hat, sigma)
    z = draw_noise(SplitMix64(derive_seed(0, TAG_HAND)), draws, 1, 2)
    mc = expected_solution(solver, chat, sigma, z)[0]
    return {
        "hard_path": int(np.argmax(hard)) + 1, "best_path": int(np.argmax(y)) + 1,
        "weights": (p1, p2), "mc": (float(mc[0]), float(mc[1])), "draws": draws,
        "gradient": (float(y[0] - p1), float(y[1] - p2)),
        "gradient_hard": (float(y[0] - hard[0]), float(y[1] - hard[1])),
    }


def hand_curve(c1hat: float, c2hat: float, sigma_max: float = C.HSIGMA_MAX, points: int = C.HAND_SIGMA_GRID) -> dict:
    """Gewicht auf Weg 1 und erste Komponente des Teilgradienten in Abhängigkeit von σ (geschlossene Form) und der Stichprobenversuch an vier Stellen."""
    sigmas = np.linspace(0.05, sigma_max, points)
    w1 = np.array([two_path_weights(c1hat, c2hat, float(s))[0] for s in sigmas])
    y1 = 1.0   # y = w*(c) = (1, 0) für c = (3, 5)
    solver = two_paths()
    chat = np.array([[c1hat, c2hat]])
    z = draw_noise(SplitMix64(derive_seed(0, TAG_HAND)), C.HAND_DRAWS, 1, 2)
    mc_sigmas = np.array(C.SIGMA_CHOICES)
    mc_w1 = np.array([float(expected_solution(solver, chat, float(s), z)[0, 0]) for s in mc_sigmas])
    hard = two_paths().solve(chat)[0][0]
    return {"sigma": sigmas, "w1": w1, "g1": y1 - w1, "mc_sigma": mc_sigmas, "mc_w1": mc_w1, "hard_w1": float(hard[0])}


def verdict_hand(h: dict, sigma: float) -> tuple[str, str]:
    """Verdikt des Handbeispiels aus `hand_numbers`."""
    p1, p2 = h["weights"]
    g1, g2 = h["gradient"]
    head = f"Der harte Löser wählt Weg {h['hard_path']}. Mit Störung σ = {sig(sigma)} bekommt Weg 1 das Gewicht {de(p1, 3)} und Weg 2 {de(p2, 3)}; der Teilgradient ist ({de(g1, 3)}; {de(g2, 3)})."
    if _shown(p1, 3) in (0.0, 1.0):
        return "warning", head + " Die Störung ist zu klein, um die Wahl zu ändern: Gewicht und Teilgradient sind die des harten Lösers."
    if _shown(p1, 3) == 0.5:
        return "info", head + " Die beiden Wege sind gleich gewichtet: die Vorhersage unterscheidet sie bei dieser Störung nicht."
    return "success", head + " Beide Wege haben Gewicht, der Teilgradient ist ungleich null und ändert sich glatt mit ĉ."


# --- Experiment 2: σ und das harte Argmax ------------------------------------------------------------------------------------------------------

def sigma_rows(noise: float, m: int = 1) -> list[dict]:
    """Zeilen der Tabelle: Orakel, KQ, SPO+ und das Training mit gestörten Lösungen für jedes σ (Median, Spanne, Seeds mit besserem Regret als KQ) nach 300 Schritten."""
    c = B.cell(noise)
    rows = [
        {"Verfahren": "Orakel (kennt die mittleren Kosten)", "kind": "orakel", "sigma": None, "stat": B.summary(c["regret_orakel"]), "besser": None},
        {"Verfahren": "KQ (linear)", "kind": "kq", "sigma": None, "stat": B.summary(c["regret_kq"]), "besser": None},
        {"Verfahren": "SPO+ (linear, 300 Schritte)", "kind": "spo", "sigma": None, "stat": B.summary(B.regrets(noise, "SPO+")), "besser": B.better_than_kq(noise, "SPO+")},
    ]
    for s in C.SIGMA_CHOICES:
        name = B.fy_name(s, m)
        rows.append({"Verfahren": f"Gestörte Lösungen, σ = {sig(s)}, M = {m}", "kind": "fy", "sigma": s, "stat": B.summary(B.regrets(noise, name)), "besser": B.better_than_kq(noise, name)})
    return rows


def verdict_sigma(sigma: float, m: int, noise: float, r_fy: float, r_kq: float, r_spo: float, better: int, seeds: int) -> tuple[str, str]:
    """Verdikt zu σ: Regret des Trainings mit gestörten Lösungen gegen KQ und SPO+ (Median über `seeds` Zufalls-Seeds, Prozent)."""
    scope = f"Median über {seeds} Zufalls-Seeds, Rauschstufe {de(noise)}, M = {m}"
    head = f"σ = {sig(sigma)} erreicht {de(r_fy)} % Regret, KQ {de(r_kq)} %, SPO+ {de(r_spo)} % ({scope}; besser als KQ in {better} von {seeds} Seeds)."
    if _shown(r_fy) < _shown(r_kq):
        return "success", head
    if _shown(r_fy) == _shown(r_kq):
        return "info", head + " Im Median kein Unterschied zu KQ."
    return "warning", head + " Der Regret liegt über dem von KQ: bei zu kleinem σ verhält sich das Training wie der harte Löser."


# --- Ein Testfall im Gitter: gemittelte gestörte Lösung -------------------------------------------------------------------------------------------

def show_case(run, case: int, sigma: float, draws: int = C.SHOW_DRAWS) -> dict:
    """Testfall `case` (1-basiert) im Gitter: Vorhersage von KQ, harter Weg, Anteil der Störungen, die jeden Bogen benutzen, und Zahl verschiedener Wege.

    Die Störungen kommen aus einem festen Strom (Zufalls-Seed der Welt, Fallnummer); bei gleichen Einstellungen ist das Bild also dasselbe."""
    i = case - 1
    chat = predict(run.coef_kq, run.data.X_test[i:i + 1])
    hard, _ = run.grid.solve(chat)
    z = draw_noise(SplitMix64(derive_seed(run.seed * 1000 + case, TAG_SHOW)), draws, 1, run.grid.d)
    W = perturbed_solutions(run.grid, chat, sigma, z)[:, 0, :]
    freq = W.mean(axis=0)
    distinct = len({tuple(row.astype(int)) for row in W})
    on_hard = float(freq[hard[0] > 0].mean())
    return {"chat": chat[0], "hard": hard[0], "freq": freq, "distinct": distinct, "draws": draws, "on_hard": on_hard,
            "same_as_hard": float(np.mean(np.all(W == hard[0], axis=1)))}


def verdict_show(sigma: float, info: dict) -> tuple[str, str]:
    """Verdikt zum gezeigten Testfall: wie weit verteilt die Störung das Gewicht auf verschiedene Wege?"""
    d, n = info["distinct"], info["draws"]
    share = de(100 * info["same_as_hard"], 0)
    base = f"Bei σ = {sig(sigma)} liefern {n} Störungen {d} verschiedene Wege; {share} % der Störungen wählen denselben Weg wie der harte Löser."
    if d == 1:
        return "warning", base + " Alle Störungen liefern denselben Weg: das Mittel ist der harte Weg, der Gradient ist der des harten Lösers."
    return "info", base + f" Die Bögen des harten Weges tragen im Mittel das Gewicht {de(info['on_hard'], 2)}."


# --- Experiment 3: Training (live) -------------------------------------------------------------------------------------------------------------

def verdict_training(r_kq: float, r_spo: float, r_fy: float, sigma: float, m: int) -> tuple[str, str]:
    """Verdikt der Lernkurve einer Instanz (Regret in Prozent nach 300 Schritten): Training mit gestörten Lösungen gegen KQ und SPO+."""
    head = f"Nach 300 Schritten: KQ {de(r_kq)} %, SPO+ {de(r_spo)} %, gestörte Lösungen (σ = {sig(sigma)}, M = {m}) {de(r_fy)} %."
    if _shown(r_fy) < _shown(r_kq):
        tail = " Das Training mit gestörten Lösungen liegt unter KQ"
        if _shown(r_fy) < _shown(r_spo):
            tail += " und unter SPO+."
        elif _shown(r_fy) == _shown(r_spo):
            tail += " und gleichauf mit SPO+."
        else:
            tail += ", aber über SPO+."
        return "success", head + tail
    if _shown(r_fy) == _shown(r_kq):
        return "info", head + " Das Training mit gestörten Lösungen ändert den Regret von KQ in dieser Instanz nicht."
    return "warning", head + " Das Training mit gestörten Lösungen liegt in dieser Instanz über KQ."


# --- Experiment 4: Glättungsverzerrung -------------------------------------------------------------------------------------------------------------

def bias_rows(noise: float) -> list[dict]:
    """Je Verfahren das Verhältnis |Mittel des Teilgradienten| / Mittel von |Teilgradient| an der wahren Vorhersage: Median, Minimum, Maximum über die Seeds."""
    names = ["SPO+"] + [f"FY σ = {s}" for s in C.SIGMA_CHOICES]
    return [{"Verfahren": n if n == "SPO+" else f"Gestörte Lösungen, σ = {sig(float(n.split('= ')[1]))}", "key": n, "stat": B.summary(B.bias(noise, n))} for n in names]


def verdict_bias(noise: float, spo: float, fy: float, sigma: float) -> tuple[str, str]:
    """Verdikt zur Glättungsverzerrung (Mediane über die Seeds): SPO+ gegen das gewählte σ."""
    head = f"Rauschstufe {de(noise)}, Vorhersage = wahre Mittelkosten: Anteil des systematischen Teilgradienten bei SPO+ {de(spo, 2)}, bei σ = {sig(sigma)} {de(fy, 2)}."
    if noise == 0:
        return "info", head + " Ohne Rauschen ist der harte Teilgradient exakt null; der gestörte nicht, denn die Störung verschmiert die Lösung auch dann, wenn die Vorhersage stimmt."
    if _shown(fy, 2) > _shown(spo, 2):
        return "info", head + " Der gestörte Teilgradient hat einen größeren systematischen Anteil als SPO+; er ist nicht null, wenn die Vorhersage stimmt."
    return "info", head + " Der gestörte Teilgradient hat keinen größeren systematischen Anteil als SPO+."


# --- Experiment 5: Löseraufwand ---------------------------------------------------------------------------------------------------------------

def effort_rows(noise: float, sigma: float) -> list[dict]:
    """Verfahren mit gelösten Instanzen (Schleife und vorab) und Median-Regret nach 25 / 50 / 100 / 300 Schritten (eigene Messung); Black-Box nur als Zahl."""
    n, steps = C.N_TRAIN, C.STEPS
    rows = [{"Verfahren": "KQ", "key": "kq", "loop": 0, "upfront": 0, "regret": {s: B.median(noise, "kq") for s in (25, 50, 100, 300)}}]
    rows.append({"Verfahren": "SPO+", "key": "SPO+", "loop": B.instances(noise, "SPO+")[0], "upfront": B.instances(noise, "SPO+")[1],
                 "regret": {s: B.median(noise, "SPO+", s) for s in (25, 50, 100, 300)}})
    for m in C.M_CHOICES:
        name = B.fy_name(sigma, m)
        loop, up = B.instances(noise, name)
        rows.append({"Verfahren": f"Gestörte Lösungen, σ = {sig(sigma)}, M = {m}", "key": name, "loop": loop, "upfront": up,
                     "regret": {s: B.median(noise, name, s) for s in (25, 50, 100, 300)}})
    rows.append({"Verfahren": "Löser als Black Box (nur Zahl, nicht gerechnet)", "key": "bb", "loop": instances_in_loop("bb", n, steps), "upfront": 0, "regret": None})
    return rows


def verdict_effort(spo_at25: float, spo_at300: float, fy1: float, fy5: float, sigma: float, n: int, noise: float, paired: float | None = None) -> tuple[str, str]:
    """Verdikt zum Löseraufwand: Regret von SPO+ nach 25 gegen 300 Schritten (mit gelösten Instanzen) und der Preis der fünf Störungen gegen eine.

    `fy1`, `fy5` sind Mediane der Regrets; `paired` ist der Median der Differenzen je Seed (Regret mit M = 1 minus Regret mit M = 5), das Maß für den Gewinn.
    Ohne `paired` steht nur die Differenz der Mediane, ausdrücklich so benannt."""
    solved25 = instances_in_loop("spo+", n, 25)
    head = (f"SPO+ hat nach 25 Schritten ({de_int(solved25)} gelöste Instanzen) {de(spo_at25)} % Regret, nach 300 Schritten ({de_int(instances_in_loop('spo+', n, 300))}) {de(spo_at300)} % "
            f"(Median, Rauschstufe {de(noise)}). ")
    diff = fy1 - fy5
    if _shown(fy5) < _shown(fy1):
        gain = (f"je Seed gepaart im Median um {de(paired, 2)} Punkte" if paired is not None else f"in der Differenz der Mediane um {de(diff)} Punkte")
        tail = f"Fünf Störungen statt einer (σ = {sig(sigma)}) kosten das Fünffache an Löseraufrufen und senken den Regret {gain} (Mediane der Regrets {de(fy1)} auf {de(fy5)} %)."
    elif _shown(fy5) == _shown(fy1):
        tail = f"Fünf Störungen statt einer (σ = {sig(sigma)}) kosten das Fünffache an Löseraufrufen und ändern den Regret im Median nicht ({de(fy1)} %)."
    else:
        tail = f"Fünf Störungen statt einer (σ = {sig(sigma)}) kosten das Fünffache an Löseraufrufen, der Regret liegt dabei höher ({de(fy5)} gegen {de(fy1)} %)."
    return "info", head + tail


def timing_rows(measured: dict, n: int = C.N_TRAIN, steps: int = C.STEPS) -> list[dict]:
    """Aus einer LIVE-Messung (`measured`: Seitenlänge h -> Sekunden für 1000 Instanzen) die Zeit der Löseraufrufe je Verfahren in Sekunden.

    Reine Rechnung: gelöste Instanzen / 1000 · gemessene Zeit; sie enthält nur die Löserzeit, nicht die Gradientenrechnung, und gilt für den gemessenen Löser (hier ein Gitter-Löser)."""
    rows = []
    for h, seconds in measured.items():
        arcs = 2 * h * (h - 1)
        row = {"h": h, "arcs": arcs, "ms_per_1000": 1000 * seconds}
        for label, method, m in (("SPO+", "spo+", 1), ("FY, M = 1", "fy", 1), ("FY, M = 5", "fy", 5)):
            row[label] = instances_in_loop(method, n, steps, m) / 1000 * seconds
        rows.append(row)
    return rows
