"""Erzeugt die Zahlentabellen der README aus den vorgerechneten Bändern (`gso_bands.json`).

`tests/test_claims.py` prüft, dass jede dieser Zeilen wörtlich in der README steht: die Zahlen der README sind damit nie von Hand abgeschrieben, und ändert sich eine Messung,
schlägt der Test an. Ausgabe zum Einfügen: `python tests/readme_tables.py`."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gso_bands as B  # noqa: E402
import gso_constants as C  # noqa: E402
from gso_effort import instances_in_loop  # noqa: E402
from gso_evaluation import de, de_int, sig  # noqa: E402


def _band(s: dict, digits: int = 2) -> str:
    return f"{de(s['median'], digits)} [{de(s['min'], digits)}; {de(s['max'], digits)}]"


def _names(noise: float) -> list[tuple[str, str]]:
    out = [("SPO+", "SPO+ (300 Schritte)")]
    for m in C.M_CHOICES:
        for s in C.SIGMA_CHOICES:
            out.append((B.fy_name(s, m), f"Gestörte Lösungen, σ = {sig(s)}, M = {m}"))
    return out


def regret_table(noise: float) -> list[str]:
    """Median des Regrets in Prozent nach 25 / 50 / 100 / 300 Schritten, Spanne nach 300 Schritten, Seeds mit besserem Regret als KQ, gelöste Instanzen in der Schleife."""
    c = B.cell(noise)
    lines = ["| Verfahren | 25 Schritte | 50 Schritte | 100 Schritte | 300 Schritte [Minimum; Maximum] | besser als KQ | gelöste Instanzen |", "|---|---|---|---|---|---|---|",
             f"| Orakel | {de(np.median(c['regret_orakel']), 2)} | {de(np.median(c['regret_orakel']), 2)} | {de(np.median(c['regret_orakel']), 2)} | {_band(c['summary']['orakel'])} | – | – |",
             f"| KQ | {de(np.median(c['regret_kq']), 2)} | {de(np.median(c['regret_kq']), 2)} | {de(np.median(c['regret_kq']), 2)} | {_band(c['summary']['kq'])} | – | 0 |"]
    for key, label in _names(noise):
        r = c["methods"][key]["regret"]
        meds = [de(np.median(r[str(s)]), 2) for s in (25, 50, 100)]
        lines.append(f"| {label} | {meds[0]} | {meds[1]} | {meds[2]} | {_band(B.summary(r['300']))} | {B.better_than_kq(noise, key)} von {len(r['300'])} | {de_int(B.instances(noise, key)[0])} |")
    return lines


def bias_table() -> list[str]:
    lines = ["| Verfahren | Rauschstufe 0 | Rauschstufe 0,5 |", "|---|---|---|"]
    for name in ["SPO+"] + [f"FY σ = {s}" for s in C.SIGMA_CHOICES]:
        label = name if name == "SPO+" else f"Gestörte Lösungen, σ = {sig(float(name.split('= ')[1]))}"
        lines.append(f"| {label} | {_band(B.summary(B.bias(0.0, name)))} | {_band(B.summary(B.bias(0.5, name)))} |")
    return lines


def stability_table() -> list[str]:
    """Stabilitätsprobe: 100 weitere Zufalls-Seeds (20 bis 119), Rauschstufe 0,5, Regret nach 300 Schritten."""
    st = B.own()["stability"]["regret"]
    kq, sp = np.array(st["kq"]), np.array(st["SPO+"])
    lines = ["| Verfahren | Regret (Median) | besser als KQ | Median der Differenz zu KQ | besser als SPO+ | Median der Differenz zu SPO+ |", "|---|---|---|---|---|---|"]
    lines.append(f"| KQ | {de(np.median(kq), 2)} % | – | – | – | – |")
    lines.append(f"| SPO+ | {de(np.median(sp), 2)} % | {int((sp < kq).sum())} von {len(kq)} | {de(np.median(sp - kq), 2)} Punkte | – | – |")
    for key in ("FY|0.1|1", "FY|1.0|1", "FY|3.0|1", "FY|1.0|5"):
        v = np.array(st[key])
        _, s, m = key.split("|")
        lines.append(f"| Gestörte Lösungen, σ = {sig(float(s))}, M = {m} | {de(np.median(v), 2)} % | {int((v < kq).sum())} von {len(v)} | {de(np.median(v - kq), 2)} Punkte | {int((v < sp).sum())} von {len(v)} | {de(np.median(v - sp), 2)} Punkte |")
    return lines


def effort_table() -> list[str]:
    lines = ["| Verfahren | gelöste Instanzen (300 Schritte) | einmalig vorab |", "|---|---|---|",
             f"| KQ | 0 | 0 |", f"| SPO+ | {de_int(instances_in_loop('spo+', C.N_TRAIN, C.STEPS))} | {de_int(C.N_TRAIN)} |"]
    for m in C.M_CHOICES:
        lines.append(f"| Gestörte Lösungen, M = {m} | {de_int(instances_in_loop('fy', C.N_TRAIN, C.STEPS, m))} | {de_int(C.N_TRAIN)} |")
    lines.append(f"| Löser als Black Box (nur Zahl) | {de_int(instances_in_loop('bb', C.N_TRAIN, C.STEPS))} | – |")
    return lines


def compare_table() -> list[str]:
    """Vormessung (eingefroren, `gso_vormessung.json`) gegen diese Demo: Median des Regrets nach 300 Schritten, beide Rauschstufen."""
    vm = json.loads((Path(__file__).resolve().parent.parent / "gso_vormessung.json").read_text(encoding="utf-8"))["regret"]
    lines = ["| Verfahren | Rauschstufe 0, Vormessung | Rauschstufe 0, diese Demo | Rauschstufe 0,5, Vormessung | Rauschstufe 0,5, diese Demo |", "|---|---|---|---|---|"]
    rows = [("KQ", "KQ", "kq"), ("SPO+", "SPO+", "SPO+")] + [(f"Gestörte Lösungen, σ = {sig(s)}, M = {m}", f"FY_{s}_{m}", B.fy_name(s, m)) for s, m in ((0.1, 1), (0.3, 1), (1.0, 1), (0.3, 5), (1.0, 5))]
    for label, vkey, mine in rows:
        cells = [de(vm["0.0"][vkey]["300"], 2), de(B.median(0.0, mine), 2), de(vm["0.5"][vkey]["300"], 2), de(B.median(0.5, mine), 2)]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


if __name__ == "__main__":
    for title, lines in (("Regret ohne Rauschen", regret_table(0.0)), ("Regret mit Rauschen 0,5", regret_table(0.5)), ("Teilgradient an der Wahrheit", bias_table()),
                         ("Stabilitätsprobe", stability_table()), ("Löseraufwand", effort_table()), ("Vergleich", compare_table())):
        print(f"### {title}")
        print("\n".join(lines))
        print()
