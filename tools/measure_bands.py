"""Misst die Bänder, die die App als "vorgerechnet" zeigt, und schreibt `gso_bands.json` (feste Zufalls-Seeds 0 bis 19, 300 Adam-Schritte).

    python tools/measure_bands.py [--jobs 2] [--out gso_bands.json] [--stability-seeds 100]

Inhalt: (1) Welt Grad 6, 5 Merkmale, n = 1000, 1000 Testfälle, Rauschstufe 0 und 0,5: normierter Regret von Orakel, KQ, SPO+ und dem Training mit gestörten Lösungen
(σ = 0,1 / 0,3 / 1 / 3 mit M = 1 und M = 5 Störungen je Schritt) nach 0, 5, 10, 25, 50, 100 und 300 Schritten, dazu die gezählten gelösten Instanzen;
(2) Teilgradient an der wahren Vorhersage (Glättungsverzerrung), 60 Merkmalsvektoren mit je 400 Ziehungen je Seed; (3) Stabilitätsprobe: 100 weitere Zufalls-Seeds
(20 bis 119) bei Rauschstufe 0,5 für die Kernvergleiche. Zufall nur über den ganzzahligen SplitMix64. Die Zahlen sind trainiert und plattformabhängig;
die Tests prüfen sie nur als Bänder und Ungleichungen. Nie mehr als zwei Prozesse (`--jobs`)."""

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from gso_fyrun import CHECK_STEPS, DEG, LR, N_TEST, STEPS, bias_ratios, run_methods  # noqa: E402

SEEDS = tuple(range(20))
STAB_SEEDS = tuple(range(20, 120))
NOISES = (0.0, 0.5)
SIGMAS = (0.1, 0.3, 1.0, 3.0)
MS = (1, 5)
N = 1000
CONFIGS = tuple((s, m) for s in SIGMAS for m in MS)
STAB_CONFIGS = ((0.1, 1), (1.0, 1), (3.0, 1), (1.0, 5))


def key(sigma, m):
    return f"FY|{sigma}|{m}"


def _cell(args):
    noise, seed = args
    w = run_methods(noise, seed, configs=CONFIGS, n=N)
    out = {"noise": noise, "seed": seed, "orakel": 100 * w.regret_oracle, "kq": 100 * w.regret_kq,
           "SPO+": {"regret": {str(s): 100 * v for s, v in w.spo.regret.items()}, "loop": w.spo.loop_instances, "upfront": w.spo.upfront_instances}}
    for (s, m), r in w.fy.items():
        out[key(s, m)] = {"regret": {str(t): 100 * v for t, v in r.regret.items()}, "loop": r.loop_instances, "upfront": r.upfront_instances}
    out["bias"] = bias_ratios(noise, seed, sigmas=SIGMAS)
    return out


def _stab(seed):
    w = run_methods(0.5, seed, configs=STAB_CONFIGS, n=N, snapshot_steps=(0, STEPS))
    out = {"seed": seed, "kq": 100 * w.regret_kq, "orakel": 100 * w.regret_oracle, "SPO+": 100 * w.spo.regret[STEPS]}
    for (s, m), r in w.fy.items():
        out[key(s, m)] = 100 * r.regret[STEPS]
    return out


def _summary(values):
    a = np.asarray(values, dtype=float)
    return {"median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "gso_bands.json"))
    ap.add_argument("--stability-seeds", type=int, default=len(STAB_SEEDS))
    args = ap.parse_args()
    t0 = time.time()
    stab_seeds = STAB_SEEDS[: args.stability_seeds]
    with Pool(max(1, min(args.jobs, 2))) as pool:
        rows = pool.map(_cell, [(e, s) for e in NOISES for s in SEEDS], chunksize=1)
        print(f"Raster fertig nach {time.time() - t0:.0f} s", flush=True)
        stab = pool.map(_stab, stab_seeds, chunksize=2)
    cells = []
    for noise in NOISES:
        rs = [r for r in rows if r["noise"] == noise]
        methods = {}
        for name in ["SPO+"] + [key(s, m) for s, m in CONFIGS]:
            regret = {str(t): [r[name]["regret"][str(t)] for r in rs] for t in CHECK_STEPS}
            methods[name] = {"regret": regret, "summary": {t: _summary(v) for t, v in regret.items()},
                             "loop_instances": rs[0][name]["loop"], "upfront_instances": rs[0][name]["upfront"]}
        cells.append({
            "noise": noise, "seeds": [r["seed"] for r in rs],
            "regret_orakel": [r["orakel"] for r in rs], "regret_kq": [r["kq"] for r in rs],
            "summary": {"orakel": _summary([r["orakel"] for r in rs]), "kq": _summary([r["kq"] for r in rs])},
            "methods": methods,
            "bias": {name: [r["bias"][name] for r in rs] for name in rs[0]["bias"]},
        })
    stability = {"noise": 0.5, "seeds": list(stab_seeds), "regret": {name: [r[name] for r in stab] for name in stab[0] if name != "seed"}}
    meta = {"seeds": list(SEEDS), "steps": STEPS, "lr": LR, "n": N, "n_test": N_TEST, "deg": DEG, "check_steps": list(CHECK_STEPS), "sigmas": list(SIGMAS), "ms": list(MS),
            "hinweis": "Eigene Messung (tools/measure_bands.py); Zahlen sind trainiert und plattformabhängig, Tests prüfen Bänder und Ungleichungen."}
    Path(args.out).write_text(json.dumps({"meta": meta, "cells": cells, "stability": stability}, ensure_ascii=False), encoding="utf-8")
    print(f"fertig nach {time.time() - t0:.0f} s -> {args.out}")


if __name__ == "__main__":
    main()
