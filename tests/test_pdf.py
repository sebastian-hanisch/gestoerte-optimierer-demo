"""PDF-Export: erzeugt ein gültiges PDF für jedes Preset (Umlaute und Sonderzeichen brechen die Kernschrift nicht)."""

import pytest

import gso_constants as C
import gso_evaluation as ev
from gso_fyrun import run_methods
from gso_pdf_export import _s, build_pdf


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_pdf_is_built_for_every_preset(name):
    p = C.PRESETS[name]
    s = ev.Settings(noise=p["noise"], seed=p["seed"], sigma=p["sigma"], m=p["m"])
    run = run_methods(s.noise, s.seed, configs=((s.sigma, s.m),), n=100, steps=300, snapshot_steps=(0, 300))
    pdf = build_pdf(s, run)
    assert pdf.startswith(b"%PDF") and len(pdf) > 1500


def test_special_characters_are_replaced_by_latin1_safe_text():
    assert _s("ĉ → 3 – σ Φ") == "c^ -> 3 - sigma Phi"
    assert _s("Löser für Gitter ß") == "Löser für Gitter ß"
    assert _s("λ") == "?"
