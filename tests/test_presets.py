"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; Permalink begrenzt und rastet ein."""

import gso_constants as C
import gso_evaluation as ev
import gso_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_choices():
    for preset in C.PRESETS.values():
        assert preset["noise"] in C.NOISE_CHOICES and preset["sigma"] in C.SIGMA_CHOICES and preset["m"] in C.M_CHOICES and 0 <= preset["seed"] <= C.SEED_MAX


def test_default_preset_equals_the_default_settings():
    first = C.PRESETS[next(iter(C.PRESETS))]
    d = ev.Settings()
    assert (first["noise"], first["sigma"], first["m"], first["seed"]) == (d.noise, d.sigma, d.m, d.seed)


def test_setting_specs_cover_every_preset_key_and_the_experiment_regler():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)
    assert set(P.SETTING_SPECS) - set(P.PRESET_KEYS.values()) == {"case_slider", "c1_slider", "c2_slider", "hsigma_slider"}


def test_snap_limits_and_snaps_to_choices_and_grid():
    S = P.SETTING_SPECS
    assert P.snap(S["noise_slider"], 0.3) == 0.5 and P.snap(S["noise_slider"], 0.2) == 0.0
    assert P.snap(S["sigma_slider"], 2.0) == 1.0 and P.snap(S["sigma_slider"], 100) == 3.0 and P.snap(S["sigma_slider"], -5) == 0.1
    assert P.snap(S["m_slider"], 3) == 1 and P.snap(S["m_slider"], 4) == 5
    assert P.snap(S["c1_slider"], 4.3) == 4.5 and P.snap(S["c1_slider"], 99) == C.CHAT_MAX and P.snap(S["c1_slider"], -1) == C.CHAT_MIN
    assert P.snap(S["hsigma_slider"], 0.04) == C.HSIGMA_MIN and P.snap(S["hsigma_slider"], 7.77) == C.HSIGMA_MAX and P.snap(S["hsigma_slider"], 1.04) == 1.0
    assert P.snap(S["seed_input"], -4) == 0 and P.snap(S["seed_input"], 10**9) == C.SEED_MAX
