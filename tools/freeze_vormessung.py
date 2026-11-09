"""Friert die Zahlen der Vormessung ein (`gso_vormessung.json`), damit die README sie neben die eigene Messung stellen kann.

    python tools/freeze_vormessung.py [--src ../entscheidungsorientiert-planung/vormessung]

Quellen (Rohausgaben, nur gelesen): `out_eo_v7_9_stoerung_bb_aufwand.txt` (Regret nach 25 / 50 / 100 / 300 Schritten, 20 Seeds, Median in Prozent; dort mit anderer Zufallsquelle,
gestörte Lösungen aus numpy.random und Löserzeit-Messung) und `out_eo_v8_bias.txt` (Teilgradient an der wahren Vorhersage, eine Welt, Median über 60 Merkmalsvektoren).
Die Black-Box-Zeilen werden nicht übernommen (anderes Verfahren). Die Löserzeiten der Vormessung werden bewusst nicht übernommen: Zeiten werden nur live gemessen."""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT.parent / "entscheidungsorientiert-planung" / "vormessung"))
    ap.add_argument("--out", default=str(ROOT / "gso_vormessung.json"))
    args = ap.parse_args()
    src = Path(args.src)
    text = (src / "out_eo_v7_9_stoerung_bb_aufwand.txt").read_text(encoding="utf-8")
    regret = {}
    noise = None
    for line in text.splitlines():
        m = re.match(r"--- Welt deg 6, n = 1000, Rauschen e = ([0-9.]+),", line)
        if m:
            noise = m.group(1)
            regret[noise] = {}
            continue
        m = re.match(r"\s*(KQ|SPO\+|FY_[0-9.]+_[0-9]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*(\d+)", line)
        if m and noise is not None:
            regret[noise][m.group(1)] = {"25": float(m.group(2)), "50": float(m.group(3)), "100": float(m.group(4)), "300": float(m.group(5)), "instanzen": int(m.group(6))}
    bias = {}
    noise = None
    for line in (src / "out_eo_v8_bias.txt").read_text(encoding="utf-8").splitlines():
        m = re.match(r"--- Rauschen e = ([0-9.]+),", line)
        if m:
            noise = m.group(1)
            bias[noise] = {}
            continue
        m = re.match(r"\s*(SPO\+|FY) (\{.*\}): Median über x = ([0-9.]+)", line)
        if m and noise is not None and m.group(1) in ("SPO+", "FY"):
            bias[noise][m.group(1) + m.group(2)] = float(m.group(3))
    data = {"quelle": "entscheidungsorientiert-planung/vormessung/out_eo_v7_9_stoerung_bb_aufwand.txt und out_eo_v8_bias.txt (Vormessung, 20 Seeds bzw. 60 Merkmalsvektoren)",
            "hinweis": "Vormessung mit anderer Zufallsquelle (numpy.random) und anderem Aufbau; Zahlen sind mit denen dieser Demo vergleichbar, nicht gleich.",
            "regret": regret, "bias": bias}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: list(v) for k, v in regret.items()}), json.dumps(bias))


if __name__ == "__main__":
    main()
