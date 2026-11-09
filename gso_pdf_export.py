"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import gso_bands as B
import gso_constants as C
from gso_evaluation import Settings, de, de_int, hand_numbers, sig, sigma_rows

_REPLACE = {"→": "->", "–": "-", "—": "-", "ĉ": "c^", "−": "-", "≥": ">=", "·": "*", "×": "x", "σ": "sigma", "€": "EUR", "²": "^2", "ˆ": "^", "Φ": "Phi", "√": "sqrt", "ε": "eps"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, run) -> bytes:
    """`run`: das `WorldMethods` zu den Einstellungen (Live-Lauf einer Instanz mit dem gewählten σ und M)."""
    h = hand_numbers(settings.c1, settings.c2, settings.hsigma)
    rows = sigma_rows(settings.noise, settings.m)
    fy = run.fy[(settings.sigma, settings.m)]
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Gestörte Optimierer und Fenchel-Young"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Welt: 5 x 5 Gitter, 5 Merkmale, Grad {C.DEG}, Rauschstufe {de(settings.noise)}, {C.N_TRAIN} Trainingsfälle, Zufalls-Seed {settings.seed}, "
        f"{C.N_TEST} Testfälle. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Diese Instanz (ein Zufalls-Seed), Regret nach 300 Schritten"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Normierter Regret: Orakel {de(100 * run.regret_oracle)} %, KQ {de(100 * run.regret_kq)} %, SPO+ {de(100 * run.spo.regret[300])} %, "
        f"gestörte Lösungen (sigma = {sig(settings.sigma)}, M = {settings.m}) {de(100 * fy.regret[300])} %. "
        f"Gelöste Instanzen in den Trainingsrunden: SPO+ {de_int(run.spo.loop_instances)}, gestörte Lösungen {de_int(fy.loop_instances)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s(f"Median über {len(B.cell(settings.noise)['regret_kq'])} Zufalls-Seeds (vorgerechnet, 300 Adam-Schritte, M = {settings.m})"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    for r in rows:
        s = r["stat"]
        pdf.multi_cell(0, 5, _s(f"{r['Verfahren']}: Regret {de(s['median'])} % (Spanne {de(s['min'])} bis {de(s['max'])} %)"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, _s("Handbeispiel: zwei Wege"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(
        f"Wahre Kosten c = (3, 5), Vorhersage ĉ = ({de(settings.c1)}, {de(settings.c2)}), Störung sigma = {sig(settings.hsigma)}: harter Löser wählt Weg {h['hard_path']}; "
        f"Gewicht (Weg 1; Weg 2) = ({de(h['weights'][0], 3)}; {de(h['weights'][1], 3)}), Teilgradient ({de(h['gradient'][0], 3)}; {de(h['gradient'][1], 3)})."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Gestörte Optimierer und Fenchel-Young: Training mit gemittelten gestörten Lösungen gegen SPO+ und Kleinste Quadrate im Gitter. Quelle: gestoerte-optimierer-demo, Sebastian Hanisch."),
                   new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
