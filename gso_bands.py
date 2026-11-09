"""Vorgerechnete Bänder (keine Streamlit-Abhängigkeit): `gso_bands.json` (eigene Messung, `tools/measure_bands.py`).

Struktur: `meta` (Seeds, Schritte, Kontrollschritte, σ- und M-Werte), `cells` (je Rauschstufe: Regret von Orakel und KQ je Seed, je Verfahren der Regret je Seed nach den
Kontrollschritten, gezählte gelöste Instanzen, Teilgradient-Verhältnis je Seed) und `stability` (100 weitere Zufalls-Seeds, nur Regret nach 300 Schritten).
Verfahrensnamen: "SPO+" und "FY|σ|M" (Training mit gestörten Lösungen). Alle Regretwerte in Prozent."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAST = "300"


@lru_cache(maxsize=1)
def own() -> dict:
    return json.loads((HERE / "gso_bands.json").read_text(encoding="utf-8"))


def meta() -> dict:
    return own()["meta"]


def cell(noise: float) -> dict:
    for c in own()["cells"]:
        if c["noise"] == noise:
            return c
    raise KeyError(noise)


def fy_name(sigma: float, m: int) -> str:
    return f"FY|{float(sigma)}|{int(m)}"


def summary(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def regrets(noise: float, name: str, step: int | str = LAST) -> list[float]:
    """Regret je Seed (Prozent) eines Verfahrens nach `step` Schritten; "kq" und "orakel" haben keinen Schritt."""
    c = cell(noise)
    if name == "kq":
        return c["regret_kq"]
    if name == "orakel":
        return c["regret_orakel"]
    return c["methods"][name]["regret"][str(step)]


def median(noise: float, name: str, step: int | str = LAST) -> float:
    return float(np.median(regrets(noise, name, step)))


def better_than_kq(noise: float, name: str, step: int | str = LAST) -> int:
    """In wie vielen Seeds liegt der Regret des Verfahrens streng unter dem von KQ?"""
    return int(sum(a < b for a, b in zip(regrets(noise, name, step), regrets(noise, "kq"))))


def instances(noise: float, name: str) -> tuple[int, int]:
    """(gelöste Instanzen in der Schleife nach 300 Schritten, einmalig vorab) eines Verfahrens."""
    m = cell(noise)["methods"][name]
    return int(m["loop_instances"]), int(m["upfront_instances"])


def bias(noise: float, name: str) -> list[float]:
    """Verhältnis |Mittel des Teilgradienten| / Mittel von |Teilgradient| je Seed (Median über 60 Merkmalsvektoren). Namen: "SPO+", "FY σ = 0.1" usw."""
    return cell(noise)["bias"][name]


def stability(name: str) -> list[float]:
    return own()["stability"]["regret"][name]
