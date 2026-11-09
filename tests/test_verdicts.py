"""Verdikte (reine Funktionen in gso_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text; dazu die Formatfunktionen."""

import pytest

import gso_evaluation as ev


def test_number_formats():
    assert ev.de(12.34) == "12,3" and ev.de(-0.04) == "0,0" and ev.de(0.2398, 3) == "0,240"
    assert ev.de_int(300000) == "300 000" and ev.de_int(1500000) == "1 500 000" and ev.de_int(25000) == "25 000"
    assert [ev.sig(s) for s in (0.1, 0.3, 1.0, 3.0, 2.5)] == ["0,1", "0,3", "1", "3", "2,5"]


# --- Handbeispiel ---------------------------------------------------------------------------------------------------------------------------------

def _hand(p1, g1, hard=2):
    return {"hard_path": hard, "weights": (p1, 1 - p1), "gradient": (g1, -g1)}


def test_verdict_hand_no_effect_when_sigma_is_too_small():
    kind, text = ev.verdict_hand(_hand(0.0, 1.0), 0.1)
    assert kind == "warning" and "σ = 0,1" in text and "Weg 2" in text and "(1,000; -1,000)" in text and "zu klein" in text


def test_verdict_hand_smooth_case_by_hand_values():
    kind, text = ev.verdict_hand(_hand(0.2398, 0.7602), 1.0)
    assert kind == "success" and "σ = 1 " in text and "0,240" in text and "0,760" in text and "(0,760; -0,760)" in text


def test_verdict_hand_tie():
    kind, text = ev.verdict_hand(_hand(0.5, 0.5, hard=1), 2.0)
    assert kind == "info" and "gleich gewichtet" in text


# --- σ ----------------------------------------------------------------------------------------------------------------------------------------------

def test_verdict_sigma_three_branches():
    kind, text = ev.verdict_sigma(1.0, 1, 0.5, 11.74, 12.47, 11.84, 17, 20)
    assert kind == "success" and "σ = 1 erreicht 11,7 % Regret, KQ 12,5 %, SPO+ 11,8 %" in text and "besser als KQ in 17 von 20 Seeds" in text and "Rauschstufe 0,5" in text and "M = 1" in text
    kind, text = ev.verdict_sigma(0.1, 1, 0.5, 12.83, 12.47, 11.84, 4, 20)
    assert kind == "warning" and "σ = 0,1 erreicht 12,8 %" in text and "über dem von KQ" in text
    kind, text = ev.verdict_sigma(0.3, 1, 0.0, 1.84, 1.80, 0.5, 0, 20)           # 1,8 gegen 1,8 nach Rundung
    assert kind == "info" and "kein Unterschied" in text


# --- Testfall im Gitter -----------------------------------------------------------------------------------------------------------------------

def test_verdict_show_branches():
    base = {"draws": 300, "distinct": 40, "same_as_hard": 0.25, "on_hard": 0.456}
    kind, text = ev.verdict_show(1.0, base)
    assert kind == "info" and "300 Störungen 40 verschiedene Wege" in text and "25 %" in text and "0,46" in text
    kind, text = ev.verdict_show(0.1, {**base, "distinct": 1, "same_as_hard": 1.0, "on_hard": 1.0})
    assert kind == "warning" and "1 verschiedene Wege" in text and "100 %" in text and "denselben Weg" in text


# --- Training ---------------------------------------------------------------------------------------------------------------------------------

def test_verdict_training_branches():
    kind, text = ev.verdict_training(12.5, 11.8, 11.7, 1.0, 1)
    assert kind == "success" and "KQ 12,5 %, SPO+ 11,8 %, gestörte Lösungen (σ = 1, M = 1) 11,7 %" in text and "unter KQ und unter SPO+" in text
    assert "und gleichauf mit SPO+" in ev.verdict_training(12.5, 11.8, 11.8, 1.0, 1)[1]
    assert "aber über SPO+" in ev.verdict_training(12.5, 11.0, 11.8, 3.0, 5)[1]
    kind, text = ev.verdict_training(12.5, 11.8, 12.5, 0.3, 1)
    assert kind == "info" and "nicht" in text
    kind, text = ev.verdict_training(12.5, 11.8, 12.8, 0.1, 1)
    assert kind == "warning" and "12,8 %" in text and "über KQ" in text


# --- Glättungsverzerrung ----------------------------------------------------------------------------------------------------------------------

def test_verdict_bias_branches():
    kind, text = ev.verdict_bias(0.0, 0.0, 0.71, 1.0)
    assert kind == "info" and "SPO+ 0,00, bei σ = 1 0,71" in text and "exakt null" in text
    kind, text = ev.verdict_bias(0.5, 0.05, 0.11, 1.0)
    assert "SPO+ 0,05, bei σ = 1 0,11" in text and "größeren systematischen Anteil" in text and "Rauschstufe 0,5" in text
    assert "keinen größeren" in ev.verdict_bias(0.5, 0.11, 0.05, 1.0)[1]


# --- Löseraufwand -----------------------------------------------------------------------------------------------------------------------------

def test_verdict_effort_branches_and_numbers():
    kind, text = ev.verdict_effort(11.81, 11.84, 11.74, 11.63, 1.0, 1000, 0.5)
    assert kind == "info" and "(25 000 gelöste Instanzen) 11,8 %" in text and "(300 000) 11,8 %" in text
    assert "kosten das Fünffache an Löseraufrufen und senken den Regret in der Differenz der Mediane um 0,1 Punkte (Mediane der Regrets 11,7 auf 11,6 %)" in text
    paired_text = ev.verdict_effort(11.81, 11.84, 11.74, 11.63, 1.0, 1000, 0.5, paired=0.087)[1]
    assert "senken den Regret je Seed gepaart im Median um 0,09 Punkte (Mediane der Regrets 11,7 auf 11,6 %)" in paired_text
    assert "ändern den Regret im Median nicht (11,7 %)" in ev.verdict_effort(11.8, 11.8, 11.74, 11.74, 1.0, 1000, 0.5)[1]
    assert "der Regret liegt dabei höher (11,9 gegen 11,7 %)" in ev.verdict_effort(11.8, 11.8, 11.74, 11.9, 1.0, 1000, 0.5)[1]


def test_timing_rows_are_pure_arithmetic_on_a_given_measurement():
    rows = ev.timing_rows({5: 0.002, 10: 0.004}, n=1000, steps=300)
    assert [r["arcs"] for r in rows] == [40, 180]
    assert rows[0]["ms_per_1000"] == pytest.approx(2.0)
    assert rows[0]["SPO+"] == pytest.approx(300_000 / 1000 * 0.002) == pytest.approx(0.6)
    assert rows[0]["FY, M = 5"] == pytest.approx(5 * rows[0]["FY, M = 1"])
    assert rows[1]["FY, M = 1"] == pytest.approx(1.2)
