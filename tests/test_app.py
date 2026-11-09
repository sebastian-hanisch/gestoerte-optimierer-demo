"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Button, fährt jeden Regler an seine Grenzen
und prüft, dass kein Python-Fehler auftritt (insbesondere `StreamlitDuplicateElementId`), dazu Permalink, Verdikte je Preset und die Gliederung."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import gso_constants as C
import gso_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _all_verdicts(at):
    return [e.value for e in list(at.success) + list(at.info) + list(at.warning)]


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_regler_at_its_min_and_max_does_not_raise():
    """Alle Regler aus SETTING_SPECS (Schieberegler, Auswahlregler, Zahlenfeld) an beiden Grenzen."""
    for state_key, spec in P.SETTING_SPECS.items():
        for edge, value in (("min", spec.lo), ("max", spec.hi)):
            at = _fresh_app()
            found = False
            for art in ("slider", "select_slider", "number_input"):
                try:
                    widget = getattr(at, art)(key=state_key)
                except KeyError:
                    continue
                widget.set_value(value).run(timeout=TIMEOUT)
                found = True
                break
            assert found, state_key
            assert not at.exception, (state_key, edge, [str(e) for e in at.exception])


def test_worst_case_most_perturbations_and_noise_finishes_without_error():
    """Größte Last: Rauschstufe 0,5, σ = 3, M = 5 (1 500 000 gelöste Instanzen live): Timeout statt Zeit-Assertion."""
    at = _fresh_app()
    at.select_slider(key="noise_slider").set_value(0.5)
    at.select_slider(key="sigma_slider").set_value(3.0)
    at.select_slider(key="m_slider").set_value(5)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def test_time_button_measures_live_and_shows_a_table_without_stored_numbers():
    at = _fresh_app()
    assert "timing" not in at.session_state
    at.button(key="time_button").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    measured = at.session_state["timing"]
    assert set(measured) == set(C.TIMING_GRIDS) and all(v > 0 for v in measured.values())


def test_permalink_snaps_values_to_the_allowed_choices():
    at = AppTest.from_file(APP_PATH)
    at.query_params["noise"] = "0.3"
    at.query_params["seed"] = "-4"
    at.query_params["sigma"] = "2"
    at.query_params["m"] = "4"
    at.query_params["case"] = "500"
    at.query_params["c1"] = "4.3"
    at.query_params["c2"] = "abc"
    at.query_params["hsigma"] = "9"
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    s = at.session_state
    assert (s["noise_slider"], s["seed_input"], s["sigma_slider"], s["m_slider"], s["case_slider"], s["c1_slider"], s["c2_slider"], s["hsigma_slider"]) == (
        0.5, 0, 1.0, 5, C.CASE_MAX, 4.5, C.DEFAULT_C2HAT, C.HSIGMA_MAX)


def test_permalink_roundtrip_writes_the_settings_to_the_address():
    at = _fresh_app()
    at.select_slider(key="sigma_slider").set_value(0.3).run(timeout=TIMEOUT)
    assert dict(at.query_params) == {"noise": "0.0", "seed": "0", "sigma": "0.3", "m": "1", "case": "1", "c1": "4.0", "c2": "3.0", "hsigma": "1.0"}


@pytest.mark.parametrize("name,expected_kind,fragment", [
    ("Rauschfreie Welt, σ = 1 (Voreinstellung)", "success", "σ = 1 erreicht"),
    ("Mit Rauschen, σ zu klein", "info", "kein Unterschied zu KQ"),
    ("Mit Rauschen, σ = 1", "success", "σ = 1 erreicht"),
    ("Fünf Störungen je Schritt", "success", "σ = 1 erreicht"),
])
def test_every_preset_shows_the_expected_core_question_verdict(name, expected_kind, fragment):
    at = _fresh_app()
    at.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    shown = {"success": at.success, "info": at.info, "warning": at.warning}[expected_kind]
    assert any(fragment in e.value for e in shown), (name, _all_verdicts(at))


def test_all_five_experiments_the_boundaries_and_the_formulas_are_there():
    at = _fresh_app()
    headings = [s.value for s in at.subheader]
    for i in range(1, 6):
        assert any(h.startswith(f"🔬 Experiment {i}:") for h in headings), i
    assert "🚧 Wo die Annahmen enden" in headings
    assert any(e.label == "📐 Mathematische Formulierung" for e in at.expander)
    assert any(e.label.startswith("So funktioniert") for e in at.expander)
    assert [t.label for t in at.tabs] == ["Gelöste Instanzen", "Zeit je 1000 Instanzen (live gemessen)"]


def test_hand_example_verdict_changes_with_sigma():
    at = _fresh_app()
    at.slider(key="hsigma_slider").set_value(0.1).run(timeout=TIMEOUT)
    assert any("zu klein" in v for v in _all_verdicts(at))
    at.slider(key="hsigma_slider").set_value(1.0).run(timeout=TIMEOUT)
    assert any("ändert sich glatt" in v for v in _all_verdicts(at))
