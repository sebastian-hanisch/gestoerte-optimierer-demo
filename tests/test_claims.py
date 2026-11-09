"""Claims: jede Zahl der README steht hier als wörtliche Tabellenzeile, Band oder Ungleichung (DEMO-PLAYBOOK §11a).

Zahlen-Tabelle der README -> Test:

| README | Test |
|---|---|
| Tabellen Regret ohne und mit Rauschen, Teilgradient an der Wahrheit, Stabilitätsprobe, Löseraufwand, Vergleich mit der Vormessung | `test_readme_contains_every_generated_table_line` (wörtlich aus den JSON-Dateien erzeugt) |
| Aufbau der Messung (Seeds, Schritte, Lernrate, σ- und M-Werte) | `test_band_file_describes_the_documented_setup` |
| Trägt σ? (Zählung der Seeds) | `test_sigma_wins_against_kq_with_noise_and_without` |
| σ = 0,1 gegen KQ (20 Seeds und Stabilitätsprobe) | `test_small_sigma_gives_no_gain_over_kq` |
| gegen SPO+ | `test_against_spo_plus_the_difference_is_small` |
| fünf Störungen | `test_five_perturbations_gain_and_cost` |
| Teilgradient an der wahren Vorhersage | `test_gradient_at_the_truth` |
| SPO+ nach 25 Schritten, Lernkurve, σ = 3 am Anfang | `test_learning_curves_and_solver_calls` |
| Lücke zwischen KQ und Orakel | `test_share_of_the_gap_to_the_oracle` |
| Streuung zwischen Seeds | `test_spread_between_seeds_exceeds_the_differences` |
| Handbeispiel (Tabelle) | `test_handbeispiel_table` (und `tests/test_perturb.py`) |
| Korrekturen gegenüber dem Plan | `test_corrections_against_the_plan` |
| Gelöste Instanzen (Formeln, Zählung) | `tests/test_effort.py` und `test_solved_instances_in_the_band_file` |
| Live-Lauf bleibt im Band | `test_live_run_stays_inside_the_band_of_the_bands_file` |

Nicht im Test: die Scratch-Läufe der kopierten Orakel-Tests mit mehr Instanzen; die Vormessung selbst (Datei außerhalb des Repos, nur eingefroren und auf Übereinstimmung mit der README geprüft).
Trainierte Zahlen hängen von der Plattform ab: gegen die Bänder wird in Ungleichungen und mit Toleranz geprüft, exakt nur das Handbeispiel und die Zählung.
"""

import json
from pathlib import Path

import numpy as np
import pytest

import gso_bands as B
import gso_constants as C
import readme_tables as T
from gso_effort import instances_in_loop
from gso_evaluation import de
from gso_fyrun import run_methods
from gso_perturb import two_path_weights

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
VM = json.loads((ROOT / "gso_vormessung.json").read_text(encoding="utf-8"))
FY = B.fy_name


def _stab(name):
    return np.array(B.stability(name))


def test_readme_contains_every_generated_table_line():
    blocks = [T.regret_table(0.0), T.regret_table(0.5), T.bias_table(), T.stability_table(), T.effort_table(), T.compare_table()]
    missing = [line for lines in blocks for line in lines if line not in README]
    assert not missing, missing[:3]


def test_band_file_describes_the_documented_setup():
    meta = B.meta()
    assert meta["seeds"] == list(C.BAND_SEEDS) and meta["steps"] == C.STEPS and meta["lr"] == C.LR and meta["n"] == C.N_TRAIN and meta["n_test"] == C.N_TEST and meta["deg"] == C.DEG
    assert meta["check_steps"] == list(C.CHECK_STEPS) and meta["sigmas"] == list(C.SIGMA_CHOICES) and meta["ms"] == list(C.M_CHOICES)
    for noise in C.NOISE_CHOICES:
        c = B.cell(noise)
        assert c["seeds"] == list(range(20)) and len(c["regret_kq"]) == 20 and len(c["methods"]) == 1 + len(C.SIGMA_CHOICES) * len(C.M_CHOICES)
    st = B.own()["stability"]
    assert st["seeds"] == list(range(20, 120)) and st["noise"] == 0.5 and all(len(v) == 100 for v in st["regret"].values())
    for text in ("Zufalls-Seeds 0 bis 19", "300 Adam-Schritte", "Lernrate 0,05", "100 weiteren Zufalls-Seeds (20 bis 119)", "σ ∈ {0,1; 0,3; 1; 3} und M ∈ {1, 5}"):
        assert text in README, text


def test_sigma_wins_against_kq_with_noise_and_without():
    n5 = {(s, m): B.better_than_kq(0.5, FY(s, m)) for s in C.SIGMA_CHOICES for m in C.M_CHOICES}
    assert n5[(1.0, 1)] == n5[(3.0, 1)] == 20 and n5[(0.3, 1)] == 17 and n5[(0.1, 1)] == 7
    assert n5[(0.3, 5)] == n5[(1.0, 5)] == n5[(3.0, 5)] == 20 and n5[(0.1, 5)] == 19
    assert all(B.better_than_kq(0.0, FY(s, m)) == 20 for s in C.SIGMA_CHOICES for m in C.M_CHOICES)         # ohne Rauschen: alle acht Einstellungen in 20 von 20
    assert B.better_than_kq(0.5, "SPO+") == 20 and B.better_than_kq(0.0, "SPO+") == 20
    for fragment in ("σ = 1 und σ = 3 unter dem von KQ in 20 von 20 Seeds (12,2 und 11,9 gegen 13,1 %), bei σ = 0,3 in 17 von 20 (12,5 %), bei σ = 0,1 in 7 von 20 (13,1 %, gleich KQ)",
                     "Ohne Rauschen liegen alle acht Einstellungen in 20 von 20 Seeds unter KQ (0,57 bis 0,63 gegen 1,96 %)"):
        assert fragment in README, fragment
    assert (de(B.median(0.5, FY(1.0, 1))), de(B.median(0.5, FY(3.0, 1))), de(B.median(0.5, "kq")), de(B.median(0.5, FY(0.3, 1)))) == ("12,2", "11,9", "13,1", "12,5")
    zero = [B.median(0.0, FY(s, m)) for s in C.SIGMA_CHOICES for m in C.M_CHOICES]
    assert (de(min(zero), 2), de(max(zero), 2), de(B.median(0.0, "kq"), 2)) == ("0,57", "0,63", "1,96")
    assert "gewinnen σ = 1 und σ = 3 gegenüber KQ in allen 20 Zufalls-Seeds" in README


def test_small_sigma_gives_no_gain_over_kq():
    sm, kq = B.median(0.5, FY(0.1, 1)), B.median(0.5, "kq")
    assert (de(sm, 2), de(kq, 2)) == ("13,07", "13,08") and abs(sm - kq) < 0.05                              # im Median gleich
    assert de(sm) == de(kq) == "13,1"
    st_kq, st_sm = _stab("kq"), _stab(FY(0.1, 1))
    assert int((st_sm < st_kq).sum()) == 41 and de(np.median(st_sm - st_kq), 2) == "0,28" and np.median(st_sm - st_kq) > 0
    assert (de(np.median(st_sm), 2), de(np.median(st_kq), 2)) == ("13,16", "12,65")
    assert B.median(0.5, FY(0.1, 5)) < kq - 0.8 and B.better_than_kq(0.5, FY(0.1, 5)) == 19 and de(B.median(0.5, FY(0.1, 5))) == "12,2"
    for fragment in ("σ = 0,1 bringt im Median keinen Gewinn (13,1 gegen 13,1 %, in 7 von 20 Seeds besser)", "in einer Stabilitätsprobe mit 100 weiteren Zufalls-Seeds im Median 0,28 Punkte über KQ",
                     "in 41 von 100 Seeds besser als KQ und im Median der Differenzen je Seed 0,28 Punkte schlechter (Mediane der Regrets 13,16 gegen 12,65 %)", "Mit fünf Störungen ist es in 19 von 20 Seeds besser als KQ (12,2 %)"):
        assert fragment in README, fragment


def test_against_spo_plus_the_difference_is_small():
    assert (de(B.median(0.5, FY(1.0, 1)), 2), de(B.median(0.5, "SPO+"), 2)) == ("12,22", "12,20") and abs(B.median(0.5, FY(1.0, 1)) - B.median(0.5, "SPO+")) < 0.05
    sp, kq = _stab("SPO+"), _stab("kq")
    assert int((sp < kq).sum()) == 88 and de(np.median(sp - kq), 2) == "-0,64"
    expected = {FY(1.0, 1): (61, "-0,10"), FY(3.0, 1): (75, "-0,23"), FY(1.0, 5): (75, "-0,26")}
    for name, (count, diff) in expected.items():
        v = _stab(name)
        assert int((v < sp).sum()) == count and de(np.median(v - sp), 2) == diff
    assert int((_stab(FY(0.1, 1)) < sp).sum()) == 8
    zero_spo = B.median(0.0, "SPO+")
    assert all(zero_spo < B.median(0.0, FY(s, m)) for s in C.SIGMA_CHOICES for m in C.M_CHOICES)                # SPO+ vor jeder Einstellung ohne Rauschen
    assert (de(zero_spo, 2), de(min(B.median(0.0, FY(s, m)) for s in C.SIGMA_CHOICES for m in C.M_CHOICES), 2)) == ("0,52", "0,57")
    # Der Abstand ist klein gegen die Streuung zwischen den Seeds: gepaart ist SPO+ nicht in jedem Seed besser (die README sagt „unter der Streuung“, nicht „vor jeder Einstellung“).
    sp0 = np.array(B.regrets(0.0, "SPO+"))
    assert any(int((sp0 < np.array(B.regrets(0.0, FY(s, m)))).sum()) < len(sp0) for s in C.SIGMA_CHOICES for m in C.M_CHOICES)
    for fragment in ("mit Rauschen liegt σ = 1 bei einer Störung gleichauf (12,2 gegen 12,2 %); in der Stabilitätsprobe ist es im Median 0,10 Punkte besser, mit σ = 3 oder fünf Störungen 0,2 bis 0,3 Punkte",
                     "Ohne Rauschen hat SPO+ den kleinsten Median-Regret (0,52 gegen 0,57 bis 0,63 %); der Abstand liegt unter der Streuung zwischen den Seeds.",
                     "in der Stabilitätsprobe besser als SPO+ in 61 von 100 Seeds, im Median 0,10 Punkte; σ = 3 in 75 von 100 (0,23 Punkte), σ = 1 mit fünf Störungen in 75 von 100 (0,26 Punkte)"):
        assert fragment in README, fragment


def test_five_perturbations_gain_and_cost():
    m1, m5 = B.median(0.5, FY(1.0, 1)), B.median(0.5, FY(1.0, 5))
    assert de(m1 - m5, 2) == "0,29" and (de(m1, 2), de(m5, 2)) == ("12,22", "11,93")                          # Differenz der Mediane (nicht das Maß der README)
    p1, p5 = np.array(B.regrets(0.5, FY(1.0, 1))), np.array(B.regrets(0.5, FY(1.0, 5)))
    assert de(np.median(p1 - p5), 2) == "0,09" and int((p5 < p1).sum()) == 13                                   # gepaart je Seed
    s1, s5 = _stab(FY(1.0, 1)), _stab(FY(1.0, 5))
    assert de(np.median(s1) - np.median(s5), 2) == "0,23" and (de(np.median(s1), 2), de(np.median(s5), 2)) == ("12,04", "11,81")
    assert de(np.median(s1 - s5), 2) == "0,14" and int((s5 < s1).sum()) == 72
    z1, z5 = B.median(0.0, FY(1.0, 1)), B.median(0.0, FY(1.0, 5))
    assert abs(z1 - z5) < 0.02 and (de(z1, 2), de(z5, 2), de(z1 - z5, 2)) == ("0,58", "0,57", "0,01")
    q1, q5 = np.array(B.regrets(0.0, FY(1.0, 1))), np.array(B.regrets(0.0, FY(1.0, 5)))
    assert de(np.median(q1 - q5), 2) == "-0,01" and np.median(q1 - q5) < 0 and int((q5 < q1).sum()) == 9      # ohne Rauschen gepaart 0,01 Punkte schlechter
    assert B.instances(0.5, FY(1.0, 5))[0] == 5 * B.instances(0.5, FY(1.0, 1))[0] == 1_500_000
    for fragment in ("Mit Rauschen bei σ = 1: Mediane der Regrets 12,22 → 11,93 %; je Seed gepaart im Median 0,09 Punkte besser (13 von 20 Seeds). Stabilitätsprobe: Mediane 12,04 → 11,81 %, gepaart 0,14 Punkte (72 von 100). Ohne Rauschen bei σ = 1: 0,58 → 0,57 %, gepaart 0,01 Punkte schlechter (besser in 9 von 20).",
                     "senken den Regret bei σ = 1 und Rauschen je Seed im Median um etwa 0,1 Punkte (20 Seeds 0,09, besser in 13 von 20; Stabilitätsprobe 0,14, besser in 72 von 100), ohne Rauschen nicht (0,01 Punkte schlechter)",
                     "bestätigen sich, wenn je Seed gepaart wird: 0,09 Punkte (20 Seeds) und 0,14 Punkte in der Stabilitätsprobe. Die Differenz der Mediane (12,22 → 11,93 %, 0,29 Punkte; 12,04 → 11,81 %, 0,23 Punkte) ist größer"):
        assert fragment in README, fragment


def test_gradient_at_the_truth():
    spo0, spo5 = B.bias(0.0, "SPO+"), B.bias(0.5, "SPO+")
    assert max(spo0) == 0.0 and (de(np.median(spo5), 2), de(max(spo5), 2)) == ("0,05", "0,05")
    fy0 = [float(np.median(B.bias(0.0, f"FY σ = {s}"))) for s in C.SIGMA_CHOICES]
    assert all(0.69 < v < 0.76 for v in fy0) and [de(v, 2) for v in fy0] == ["0,75", "0,70", "0,70", "0,71"]
    fy5 = {s: float(np.median(B.bias(0.5, f"FY σ = {s}"))) for s in C.SIGMA_CHOICES}
    assert [de(fy5[s], 2) for s in C.SIGMA_CHOICES] == ["0,40", "0,11", "0,13", "0,21"]
    assert all(v > float(np.median(spo5)) for v in fy5.values()) and fy5[0.1] > fy5[3.0] > fy5[1.0] > fy5[0.3]
    assert "ohne Rauschen 0,70 bis 0,75 für alle σ" in README and "mit Rauschen 0,40 (σ = 0,1), 0,11 (σ = 0,3), 0,13 (σ = 1), 0,21 (σ = 3)" in README
    assert "SPO+: ohne Rauschen exakt 0, mit Rauschen im Median 0,05" in README


def test_learning_curves_and_solver_calls():
    s25, s300 = B.median(0.5, "SPO+", 25), B.median(0.5, "SPO+", 300)
    assert (de(s25, 2), de(s300, 2)) == ("12,30", "12,20") and s25 - s300 < 0.15                              # fast der Endwert nach 25 Schritten
    z25, z300 = B.median(0.0, "SPO+", 25), B.median(0.0, "SPO+", 300)
    assert (de(z25, 2), de(z300, 2)) == ("0,53", "0,52")
    f25, f300 = B.median(0.5, FY(1.0, 1), 25), B.median(0.5, FY(1.0, 1), 300)
    assert (de(f25, 2), de(f300, 2)) == ("12,19", "12,22")
    g25, g300 = B.median(0.0, FY(3.0, 1), 25), B.median(0.0, FY(3.0, 1), 300)
    assert (de(g25, 2), de(g300, 2)) == ("0,98", "0,59") and g25 > 1.4 * B.median(0.0, FY(1.0, 1), 25)       # σ = 3 startet ohne Rauschen langsamer
    assert B.median(0.0, "kq") > 3 * z25
    assert instances_in_loop("spo+", 1000, 25) == 25_000
    assert "Mit Rauschen liegt SPO+ nach 25 Schritten (25 000 gelöste Instanzen) bei 12,30 %, nach 300 Schritten (300 000) bei 12,20 % (Median); ohne Rauschen 0,53 gegen 0,52 %." in README
    assert "nach 25 Schritten bei 12,19 %, nach 300 bei 12,22 %; σ = 3 startet ohne Rauschen langsamer (0,98 % nach 25, 0,59 % nach 300 Schritten)" in README


def test_share_of_the_gap_to_the_oracle():
    def share(noise, name):
        o, k, r = B.median(noise, "orakel"), B.median(noise, "kq"), B.median(noise, name)
        return 100 * (k - r) / (k - o)

    assert (round(share(0.5, "SPO+")), round(share(0.5, FY(3.0, 1)))) == (35, 45)
    assert (round(share(0.0, "SPO+")), round(share(0.0, FY(1.0, 1)))) == (73, 70)
    assert (de(B.median(0.5, "orakel")), de(B.median(0.5, "kq"))) == ("10,5", "13,1")
    assert "SPO+ schließt 35 % der Lücke, σ = 3 mit einer Störung 45 %" in README and "Ohne Rauschen schließen SPO+ 73 % und σ = 1 mit einer Störung 70 %" in README


def test_spread_between_seeds_exceeds_the_differences():
    spans = [max(v) - min(v) for v in (B.regrets(0.5, n) for n in ["kq", "orakel", "SPO+"] + [FY(s, m) for s in C.SIGMA_CHOICES for m in C.M_CHOICES])]
    assert 2.5 <= min(spans) and max(spans) <= 4.0
    assert "Spannen von 2,5 bis 4 Punkten bei Rauschstufe 0,5" in README
    assert abs(B.median(0.5, FY(1.0, 1)) - B.median(0.5, "SPO+")) < 0.1 * min(spans)


def test_solved_instances_in_the_band_file():
    for noise in C.NOISE_CHOICES:
        assert B.instances(noise, "SPO+") == (300_000, 1000)
        for s in C.SIGMA_CHOICES:
            assert B.instances(noise, FY(s, 1)) == (300_000, 1000) and B.instances(noise, FY(s, 5)) == (1_500_000, 1000)
    assert instances_in_loop("bb", 1000, 300) == 600_000
    for line in ("| SPO+ | 300 000 | 1 000 |", "| Gestörte Lösungen, M = 5 | 1 500 000 | 1 000 |", "| Löser als Black Box (nur Zahl) | 600 000 | – |"):
        assert line in README, line
    assert "(300 · 1000 · 2)" in README


def test_handbeispiel_table():
    for sigma, w, g in ((0.1, (0.0, 1.0), 1.0), (1.0, (0.2398, 0.7602), 0.7602), (3.0, (0.4068, 0.5932), 0.5932)):
        p = two_path_weights(4.0, 3.0, sigma)
        assert p == pytest.approx(w, abs=5e-5) and 1.0 - p[0] == pytest.approx(g, abs=5e-5)
    for line in ("| 0,1 | (0,0000; 1,0000) | (1,0000; −1,0000) |", "| 1 | (0,2398; 0,7602) | (0,7602; −0,7602) |", "| 3 | (0,4068; 0,5932) | (0,5932; −0,5932) |"):
        assert line in README, line


def test_corrections_against_the_plan():
    """Die Plan-Zahlen stehen in der eingefrorenen Vormessung; die README nennt sie neben den neuen."""
    r5 = VM["regret"]["0.5"]
    assert (r5["KQ"]["300"], r5["SPO+"]["300"], r5["FY_0.1_1"]["300"], r5["FY_1.0_1"]["300"], r5["FY_1.0_5"]["300"]) == (12.47, 11.84, 12.83, 11.74, 11.63)
    r0 = VM["regret"]["0.0"]
    assert (r0["KQ"]["300"], r0["SPO+"]["300"]) == (1.87, 0.54) and r5["SPO+"]["instanzen"] == 300000 and r5["FY_1.0_5"]["instanzen"] == 1500000
    assert VM["bias"]["0.5"]["FY{'sig': 0.3}"] == 0.1 and VM["bias"]["0.5"]["FY{'sig': 1.0}"] == 0.114 and VM["bias"]["0.0"]["FY{'sig': 0.3}"] == 0.71
    assert (de(B.median(0.5, "kq"), 2), de(B.median(0.5, "SPO+"), 2), de(B.median(0.0, "kq"), 2), de(B.median(0.0, "SPO+"), 2)) == ("13,08", "12,20", "1,96", "0,52")
    for fragment in ("(13,08 und 12,20 gegen 12,47 und 11,84 %)", "(KQ 1,96 gegen 1,87, SPO+ 0,52 gegen 0,54 %)", "(Plan: 12,83 gegen 12,47 %)", "(Plan: 11,74 gegen 11,84 %)", "(Plan: 11,74 → 11,63 %)",
                     "12,22 gegen 12,20 %", "0,28 Punkte schlechter, aber in 41 von 100 Seeds besser", "Die Vormessung nannte für gestörte Lösungen mit Rauschen 0,10 und 0,114 (σ = 0,3 und 1), ohne Rauschen 0,71 und 0,706, für SPO+ 0,049 und 0,0",
                     "0,11 und 0,13 mit Rauschen, 0,70 ohne Rauschen und 0,05 und 0,00 für SPO+"):
        assert fragment in README, fragment
    assert "Millisekunden" in README and "Zeiten hängen vom Rechner ab" in README


def test_no_fixed_times_in_readme_or_app_sources():
    """Zeiten nur live: weder README noch Quelltext der App nennen eine feste Zeit je 1000 Instanzen."""
    import re

    sources = README + (ROOT / "app.py").read_text(encoding="utf-8") + (ROOT / "gso_evaluation.py").read_text(encoding="utf-8")
    assert not re.search(r"\d[\d,]*\s*(ms|µs|Millisekunden)\b", sources.replace("Millisekunden je 1000 Instanzen für drei Gittergrößen", ""))


def test_live_run_stays_inside_the_band_of_the_bands_file():
    """Seed 0, Rauschstufe 0,5, σ = 1, M = 1 live gerechnet: jeder Regret liegt im Band (Minimum bis Maximum mit Toleranz für die Plattform) der Messung; Ungleichungen dieser Welt."""
    w = run_methods(0.5, 0, configs=((1.0, 1),))
    kq, spo, fy = 100 * w.regret_kq, 100 * w.spo.regret[300], 100 * w.fy[(1.0, 1)].regret[300]
    for value, name in ((kq, "kq"), (spo, "SPO+"), (fy, FY(1.0, 1))):
        v = B.regrets(0.5, name)
        assert min(v) - 0.5 <= value <= max(v) + 0.5, name
    assert kq == pytest.approx(B.regrets(0.5, "kq")[0], abs=0.25) and fy == pytest.approx(B.regrets(0.5, FY(1.0, 1))[0], abs=0.4)
    assert fy < kq and spo < kq and w.regret_oracle * 100 < min(fy, spo)
    assert w.spo.loop_instances == 300_000 and w.fy[(1.0, 1)].loop_instances == 300_000


def test_cost_scale_of_the_world_without_noise():
    """„Median 1,4, 5. bis 95. Perzentil 1,0 bis 3,9“: Median über die Zufalls-Seeds 0 bis 19 der Perzentile der beobachteten Kosten (1000 Trainingsfälle, 40 Bögen, ohne Rauschen)."""
    from gso_world import WorldSpec, make_dataset

    q = np.array([np.percentile(make_dataset(WorldSpec(noise=0.0), 40, 1000, 10, seed).C_train, [5, 50, 95]) for seed in C.BAND_SEEDS])
    p5, p50, p95 = np.median(q, axis=0)
    assert (de(p50), de(p5), de(p95)) == ("1,4", "1,0", "3,9")
    assert "im Median bei 1,4 und zwischen dem 5. und 95. Perzentil bei 1,0 bis 3,9 (Median der Perzentile über 20 Welten)" in README
