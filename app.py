"""Gestörte Optimierer und Fenchel-Young - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Statt den Löser auf der Vorhersage zu fragen, wird die Vorhersage mit Gauß-Rauschen gestört, der Löser mehrfach gefragt und die Lösungen werden gemittelt:
der Mittelwert ist glatt in der Vorhersage und braucht den Löser nur als Black Box. Hier im 5 x 5 Gitter, gegen SPO+ und Kleinste Quadrate, mit dem Löseraufwand
als Zahl gelöster Instanzen. Aufbau nach DEMO-PLAYBOOK §2. Kern ohne Streamlit: gso_rng, gso_grid, gso_world, gso_losses, gso_training, gso_runs (aus
spo-kuerzester-weg-demo kopiert) und gso_perturb, gso_effort, gso_fyrun (neu).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import gso_bands as B
import gso_constants as C
from gso_effort import time_solver
from gso_evaluation import (
    Settings,
    bias_rows,
    de,
    de_int,
    effort_rows,
    hand_curve,
    hand_numbers,
    show_case,
    sig,
    sigma_rows,
    timing_rows,
    verdict_bias,
    verdict_effort,
    verdict_hand,
    verdict_show,
    verdict_sigma,
    verdict_training,
)
from gso_fyrun import run_methods
from gso_layout import dunkles_design, kopf_css
from gso_pdf_export import build_pdf
from gso_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from gso_visualization import (
    build_bias_figure,
    build_curve_figure,
    build_effort_figure,
    build_hand_figure,
    build_show_figure,
    build_sigma_figure,
)

st.set_page_config(page_title="Gestörte Optimierer und Fenchel-Young – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Training läuft: KQ, SPO+ und gestörte Lösungen mit je 300 Adam-Schritten (mit fünf Störungen dauert es spürbar länger) …")
def _world_run(noise: float, seed: int, sigma: float, m: int):
    return run_methods(noise, seed, configs=((sigma, m),), n=C.N_TRAIN, steps=C.STEPS, snapshot_steps=C.CURVE_STEPS)


def _pct(x: float) -> str:
    return f"{de(100 * x)} %"


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in gso_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Gestörte Optimierer und Fenchel-Young")
st.markdown(
    """
Wenn die Entscheidung ein **Weg im Netz** ist, ist die Wahl des Lösers stückweise konstant in der Vorhersage: der Weg ändert sich nur an Sprungstellen, dazwischen ist der Gradient null.
Ein **gestörter Optimierer** stört die Vorhersage mit **Rauschen**, löst das Problem mehrfach und **mittelt die Lösungen**: der Mittelwert ist glatt in der Vorhersage und braucht den Löser nur als Black Box.
Der Teilgradient ist einfach **optimale Lösung minus gemittelte gestörte Lösung**. Die **Stärke σ** der Störung ist der Regler: zu klein verhält sich das Verfahren wie das harte Argmax; ab etwa der Größe der Kosten (σ = 1 bis 3 bei Kosten um 1,4) reicht es. Zu große σ sind nicht gemessen. Hier im 5 × 5 Gitter mit 40 Bögen und 70 Wegen, gegen SPO+ und Kleinste Quadrate (KQ), mit dem **Löseraufwand als Zahl gelöster Instanzen**.

Weiter unten: **🔬 fünf Experimente**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert die gestörte Wahl", expanded=True):
    st.markdown(
        """
1. **Kosten vorhersagen:** ein lineares Modell macht aus den Merkmalen $x$ einen Kostenvektor $\\hat c$ für alle 40 Bögen (240 Parameter, Start bei Kleinsten Quadraten).
2. **Stören und lösen:** zu $\\hat c$ kommt Gauß-Rauschen $\\sigma z_m$, der Löser (Rückwärtsrechnung im azyklischen Gitter) liefert für jede Störung einen Weg $w(\\hat c + \\sigma z_m)$, $m = 1, \\dots, M$.
3. **Mitteln:** $\\bar w(\\hat c) = \\frac{1}{M}\\sum_m w(\\hat c + \\sigma z_m)$ ist ein Gewichtsvektor über die Bögen, der sich glatt mit $\\hat c$ ändert (Mittel von Wegen, kein einzelner Weg).
4. **Teilgradient:** $w^*(c) - \\bar w(\\hat c)$ (optimale Lösung unter den wahren Kosten minus gemittelte gestörte Lösung) geht per Kettenregel in das lineare Modell; Adam macht 300 Vollbatch-Schritte. Der Löser läuft dabei $M$-mal je Fall und Schritt.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

# --- Seitenleiste: Schnellstart und Welt -----------------------------------------------------------------------------------------------------

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kostenwelt**")
    noise = st.select_slider("Rauschstufe", options=list(C.NOISE_CHOICES), key="noise_slider", format_func=lambda v: de(v),
                             help="Jeder Bogen kostet das Mittel mal einen Faktor aus [1 − e, 1 + e]. Bei 0 sind die Kosten ohne Rauschen, bei 0,5 liegt auch das Orakel beim Regret deutlich über null.")
    seed = st.number_input("Zufalls-Seed der Welt", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Welt generieren", width="stretch", on_click=randomize_seed)
    st.caption(f"Feste Annahmen: Grad {C.DEG} (stark nichtlinear), {C.N_FEATURES} Merkmale, {C.N_TRAIN} Trainingsfälle, {C.N_TEST} Testfälle, lineares Modell. Nur diese Zelle ist vorgerechnet. "
               "Stärke σ und Zahl der Störungen M stehen bei Experiment 2, das Handbeispiel bei Experiment 1.")

sigma = float(st.session_state.get("sigma_slider", C.DEFAULT_SIGMA))
m = int(st.session_state.get("m_slider", C.DEFAULT_M))
run = _world_run(float(noise), int(seed), sigma, m)
fy_run = run.fy[(sigma, m)]
bs = B.cell(float(noise))
seeds_n = len(bs["regret_kq"])

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis dieser Welt")
st.caption(f"Eine Instanz (Zufalls-Seed {int(seed)}): {C.N_TRAIN} Trainingsfälle, {C.N_TEST} Testfälle, Regret nach 300 Adam-Schritten, normiert (Summe der SPO-Verluste geteilt durch die Summe der besten Wegkosten).")
m1, m2, m3, m4 = st.columns(4)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Regret Orakel", _pct(run.regret_oracle), help="Das Orakel kennt die mittleren Kosten der Testfälle; sein Regret ist der unvermeidbare Teil (Rauschen).")
m2.metric("Regret KQ", _pct(run.regret_kq), delta=f"{de(100 * (run.regret_kq - run.regret_oracle))} Punkte über dem Orakel", delta_color="off")
m3.metric("Regret SPO+", _pct(run.spo.regret[C.STEPS]), delta=f"{de(100 * (run.spo.regret[C.STEPS] - run.regret_kq))} Punkte gegen KQ", delta_color="inverse")
m4.metric(f"Regret gestörte Lösungen (σ = {sig(sigma)}, M = {m})", _pct(fy_run.regret[C.STEPS]), delta=f"{de(100 * (fy_run.regret[C.STEPS] - run.regret_kq))} Punkte gegen KQ", delta_color="inverse")

settings = Settings(float(noise), int(seed), sigma, m, int(st.session_state.get("case_slider", C.DEFAULT_CASE)),
                    float(st.session_state.get("c1_slider", C.DEFAULT_C1HAT)), float(st.session_state.get("c2_slider", C.DEFAULT_C2HAT)),
                    float(st.session_state.get("hsigma_slider", C.DEFAULT_HSIGMA)))
if st.button("📄 PDF dieser Welt erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, run))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="gestoerte-optimierer.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welche Stärke σ reicht, und was kostet sie?")
name_sel = B.fy_name(sigma, m)
kind, text = verdict_sigma(sigma, m, float(noise), B.median(float(noise), name_sel), B.median(float(noise), "kq"), B.median(float(noise), "SPO+"), B.better_than_kq(float(noise), name_sel), seeds_n)
getattr(st, kind)(text)
st.caption(
    f"Das Verdikt gilt für diese Zelle (Grad {C.DEG}, Rauschstufe {de(noise)}, n = {C.N_TRAIN}), das lineare Modell und M Störungen je Schritt: Median über {seeds_n} Zufalls-Seeds, "
    "vorgerechnet mit `tools/measure_bands.py`. Es sagt nichts über andere Löser, Netze oder Rauschformen."
)

st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: die gestörte Wahl von Hand")
st.markdown(
    "Zwei Wege mit je einem Bogen. Die **wahren Kosten** sind $c = (3, 5)$: Weg 1 ist der beste, also $w^*(c) = (1, 0)$. Stellen Sie eine **Vorhersage** $\\hat c$ und die **Stärke σ** ein. "
    "Der harte Löser wählt den Weg mit den kleineren vorhergesagten Kosten; mit Störung bekommt Weg 1 das Gewicht $\\Phi\\big((\\hat c_2 - \\hat c_1)/(\\sigma\\sqrt 2)\\big)$."
)
h1, h2, h3 = st.columns(3)
with h1:
    c1 = st.slider("Vorhersage ĉ₁ (Weg 1)", C.CHAT_MIN, C.CHAT_MAX, key="c1_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 1 (wahr: 3).")
with h2:
    c2 = st.slider("Vorhersage ĉ₂ (Weg 2)", C.CHAT_MIN, C.CHAT_MAX, key="c2_slider", step=C.CHAT_STEP, format="%.1f", help="Vorhergesagte Kosten von Weg 2 (wahr: 5).")
with h3:
    hsigma = st.slider("Stärke σ der Störung", C.HSIGMA_MIN, C.HSIGMA_MAX, key="hsigma_slider", step=C.HSIGMA_STEP, format="%.1f",
                       help="Standardabweichung des Gauß-Rauschens auf jedem Bogen, in Einheiten der Kosten.")
hn = hand_numbers(float(c1), float(c2), float(hsigma))
k1, k2, k3, k4 = st.columns(4)
k1.metric("Harter Löser wählt", f"Weg {hn['hard_path']}", help="Der Weg mit den kleineren vorhergesagten Kosten; bei Gleichstand Weg 1.")
k2.metric("Gewicht (Weg 1; Weg 2)", f"({de(hn['weights'][0], 3)}; {de(hn['weights'][1], 3)})", help="Mittel der gestörten Lösungen, geschlossene Form.")
k3.metric("Teilgradient (Weg 1; Weg 2)", f"({de(hn['gradient'][0], 3)}; {de(hn['gradient'][1], 3)})", help="w*(c) − gemittelte gestörte Lösung, mit w*(c) = (1, 0).")
k4.metric(f"Mittel aus {C.HAND_DRAWS} Störungen", f"({de(hn['mc'][0], 3)}; {de(hn['mc'][1], 3)})", help="Stichprobenversuch mit festem Zufalls-Seed: weicht von der geschlossenen Form um den Stichprobenfehler ab (Standardfehler höchstens 0,012).")
st.plotly_chart(build_hand_figure(hand_curve(float(c1), float(c2)), hn, float(hsigma)), width="stretch", key="hand_chart")
st.caption("Gestrichelt: der harte Löser (σ = 0), Gewicht 0 oder 1. Je stärker die Störung, desto weicher die Lösung und desto kleiner der Teilgradient, aber er wird nie null (anders als beim harten Löser, bei dem er fast überall null ist).")
kind, text = verdict_hand(hn, float(hsigma))
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: σ und das harte Argmax")
st.markdown(
    "Regret nach 300 Adam-Schritten für jede Stärke σ (mit M Störungen je Schritt), **Median über 20 Zufalls-Seeds**, Balken: Minimum bis Maximum; KQ, SPO+ und Orakel zum Vergleich. "
    "**σ und M gelten auch für die Experimente 3 und 5 und für das Ergebnis oben.**"
)
s1, s2 = st.columns(2)
with s1:
    st.select_slider("Stärke σ der Störung im Training", options=list(C.SIGMA_CHOICES), key="sigma_slider", format_func=sig,
                     help="Standardabweichung des Gauß-Rauschens auf jedem Bogen. Beobachtete Kosten je Bogen ohne Rauschen: Median 1,4, 5. bis 95. Perzentil 1,0 bis 3,9 (Median über 20 Welten).")
with s2:
    st.select_slider("Zahl der Störungen M je Schritt und Fall", options=list(C.M_CHOICES), key="m_slider",
                     help="Mittelung über M gestörte Lösungen je Schritt; der Löser läuft M-mal je Fall und Schritt.")
rows2 = sigma_rows(float(noise), m)
st.plotly_chart(build_sigma_figure(rows2, sigma), width="stretch", key="sigma_chart")
table2 = pd.DataFrame([
    {"Verfahren": r["Verfahren"], "Regret (Median)": f"{de(r['stat']['median'])} %", "Spanne über Seeds": f"{de(r['stat']['min'])} bis {de(r['stat']['max'])} %",
     "besser als KQ": "–" if r["besser"] is None else f"{r['besser']} von {seeds_n} Seeds"} for r in rows2
])
st.dataframe(table2, hide_index=True, width="stretch")
st.caption("Alle Zeilen: Zufalls-Seeds 0 bis 19, 1000 Testfälle, vorgerechnet; die Zeilen mit σ gehören zu dem M, das Sie eingestellt haben.")
sel_row = next(r for r in rows2 if r["kind"] == "fy" and r["sigma"] == sigma)
kind, text = verdict_sigma(sigma, m, float(noise), sel_row["stat"]["median"], next(r for r in rows2 if r["kind"] == "kq")["stat"]["median"],
                           next(r for r in rows2 if r["kind"] == "spo")["stat"]["median"], sel_row["besser"], seeds_n)
getattr(st, kind)(text)

st.markdown("**Ein Testfall im Gitter:** die Vorhersage von KQ für einen Testfall, gestört mit dem gewählten σ.")
cs1, _ = st.columns([3, 1])
with cs1:
    case = st.slider("Testfall Nr.", C.CASE_MIN, C.CASE_MAX, key="case_slider", help=f"Einer der ersten {C.CASE_MAX} von {C.N_TEST} Testfällen dieser Welt.")
info = show_case(run, int(case), sigma)
st.plotly_chart(build_show_figure(run.grid, info), width="stretch", key="show_chart")
st.caption("Linienbreite: Anteil der Störungen (feste Zufallsströme), deren Weg den Bogen benutzt; gepunktet: der Weg des harten Lösers auf der ungestörten Vorhersage von KQ. Beschriftet sind Bögen ab 10 %.")
kind, text = verdict_show(sigma, info)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Training, gestörte Lösungen gegen SPO+ gegen KQ")
st.markdown(
    f"Adam, Lernrate 0,05, Vollbatch, **300 Schritte**, Start bei der KQ-Lösung, **live** für diese Instanz (Zufalls-Seed {int(seed)}). Gezeigt ist der Regret des Parameters **nach t Schritten** "
    f"(σ = {sig(sigma)} und M = {m} aus Experiment 2)."
)
steps3 = sorted(run.spo.regret)
st.plotly_chart(build_curve_figure(steps3, [100 * run.spo.regret[s] for s in steps3], [100 * fy_run.regret[s] for s in steps3], 100 * run.regret_kq, 100 * run.regret_oracle, sigma, m),
                width="stretch", key="curve_chart")
c_a, c_b = st.columns(2)
c_a.metric("Gelöste Instanzen SPO+", de_int(run.spo.loop_instances), help=f"300 Schritte · {C.N_TRAIN} Fälle · 1 Löseraufruf; plus {de_int(run.spo.upfront_instances)} einmalig vorab für die besten Wege.")
c_b.metric("Gelöste Instanzen gestörte Lösungen", de_int(fy_run.loop_instances), help=f"300 Schritte · {C.N_TRAIN} Fälle · M = {m} Löseraufrufe; plus {de_int(fy_run.upfront_instances)} einmalig vorab.")
kind, text = verdict_training(100 * run.regret_kq, 100 * run.spo.regret[C.STEPS], 100 * fy_run.regret[C.STEPS], sigma, m)
getattr(st, kind)(text)
st.caption("Eine Instanz sagt wenig: der Median über 20 Zufalls-Seeds steht in Experiment 2 und 5. Verglichen wird der Parameter nach der letzten Runde, nicht der mit dem kleinsten Trainingsverlust.")

st.markdown("---")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Glättungsverzerrung")
st.markdown(
    "Trifft die Vorhersage die wahren bedingten Mittelkosten, sollte ein guter Teilgradient im Mittel null sein. Gemessen wird der Anteil "
    "$\\lVert \\text{Mittel des Teilgradienten} \\rVert / \\text{Mittel von } \\lVert \\text{Teilgradient} \\rVert$ (0 = im Mittel null, 1 = alle Ziehungen zeigen in dieselbe Richtung): 60 Merkmalsvektoren mit je 400 Kostenziehungen, "
    "**Median über 20 Zufalls-Seeds** (vorgerechnet), Balken: Minimum bis Maximum."
)
bias_by_noise = {n_: bias_rows(n_) for n_ in C.NOISE_CHOICES}
st.plotly_chart(build_bias_figure(bias_by_noise), width="stretch", key="bias_chart")
rows4 = bias_by_noise[float(noise)]
st.dataframe(pd.DataFrame([{"Verfahren": r["Verfahren"], "Anteil (Median)": de(r["stat"]["median"], 2), "Spanne über Seeds": f"{de(r['stat']['min'], 2)} bis {de(r['stat']['max'], 2)}"} for r in rows4]),
             hide_index=True, width="stretch")
spo_b = next(r for r in rows4 if r["key"] == "SPO+")["stat"]["median"]
fy_b = next(r for r in rows4 if r["key"] == f"FY σ = {sigma}")["stat"]["median"]
kind, text = verdict_bias(float(noise), spo_b, fy_b, sigma)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Löseraufwand")
st.markdown(
    "Der Löser ist der teure Teil, wenn er ein Optimierungsproblem ist. Gezählt werden deshalb **gelöste Instanzen** (Zeilen, die an den Löser gehen), nicht Sekunden: die Zahl hängt nicht vom Rechner ab. "
    f"Gezählt sind die Trainingsrunden ({C.N_TRAIN} Fälle, 300 Schritte); die einmalige Vorabrechnung der besten Wege ({C.N_TRAIN} Instanzen) steht getrennt."
)
tab_count, tab_time = st.tabs(["Gelöste Instanzen", "Zeit je 1000 Instanzen (live gemessen)"])
with tab_count:
    rows5 = effort_rows(float(noise), sigma)
    st.plotly_chart(build_effort_figure(rows5), width="stretch", key="effort_chart")
    table5 = pd.DataFrame([
        {"Verfahren": r["Verfahren"], "gelöste Instanzen (300 Schritte)": de_int(r["loop"]), "einmalig vorab": de_int(r["upfront"]),
         **({f"Regret nach {s} Schritten": f"{de(r['regret'][s])} %" for s in (25, 50, 100, 300)} if r["regret"] else {f"Regret nach {s} Schritten": "–" for s in (25, 50, 100, 300)})} for r in rows5
    ])
    st.dataframe(table5, hide_index=True, width="stretch")
    st.caption(
        f"Regret: Median über {seeds_n} Zufalls-Seeds, Rauschstufe {de(noise)}, eigene Messung. Black-Box-Zeile: nur die Zahl (zwei Löseraufrufe je Fall und Schritt für die Interpolation durch den Löser); "
        "dieses Verfahren ist hier nicht trainiert und hat keinen Regret."
    )
    spo_r = {s: B.median(float(noise), "SPO+", s) for s in (25, 300)}
    kind, text = verdict_effort(spo_r[25], spo_r[300], B.median(float(noise), B.fy_name(sigma, 1)), B.median(float(noise), B.fy_name(sigma, 5)), sigma, C.N_TRAIN, float(noise),
                              paired=float(np.median(np.array(B.regrets(float(noise), B.fy_name(sigma, 1))) - np.array(B.regrets(float(noise), B.fy_name(sigma, 5))))))
    getattr(st, kind)(text)
with tab_time:
    st.markdown(
        "Zeit, die der **Gitter-Löser dieser Demo** für 1000 Instanzen braucht, auf dem Rechner, auf dem diese App gerade läuft, und die daraus folgende Zeit der Löseraufrufe im Training "
        "(gelöste Instanzen geteilt durch 1000 mal die gemessene Zeit). **Nur live gemessen und nicht gespeichert.**"
    )
    if st.button("⏱ Zeit messen", key="time_button"):
        st.session_state["timing"] = {h: time_solver(h, C.TIMING_INSTANCES) for h in C.TIMING_GRIDS}
    measured = st.session_state.get("timing")
    if measured:
        trows = timing_rows(measured)
        st.dataframe(pd.DataFrame([
            {"Gitter": f"{r['h']} × {r['h']}", "Bögen": r["arcs"], "Zeit je 1000 Instanzen": f"{de(r['ms_per_1000'], 2)} ms",
             "Löserzeit SPO+ (300 Schritte)": f"{de(r['SPO+'], 2)} s", "Löserzeit gestörte Lösungen, M = 1": f"{de(r['FY, M = 1'], 2)} s", "Löserzeit gestörte Lösungen, M = 5": f"{de(r['FY, M = 5'], 2)} s"} for r in trows
        ]), hide_index=True, width="stretch")
    st.info(
        "Die Zeit ist hier nur für den billigen Gitter-Löser (eine Rückwärtsrechnung) belegt. Für Löser mit Ganzzahligkeit (zum Beispiel ein gemischt-ganzzahliges Programm) gibt es in dieser Demo keine Messung; "
        "dort kann ein Aufruf Größenordnungen länger dauern, und die Zahl der gelösten Instanzen entscheidet über die Machbarkeit. Die Tabelle rechnet nur die Löserzeit, nicht die Gradientenrechnung."
    )

st.markdown("---")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
c0 = B.cell(0.0)
_stab = B.own()['stability']['regret']
_stab_diff = float(np.median(np.array(_stab['FY|0.1|1']) - np.array(_stab['kq'])))
_stab_better = int(sum(x < y for x, y in zip(_stab['FY|0.1|1'], _stab['kq'])))
_m_gain = float(np.median(np.array(B.regrets(0.5, B.fy_name(1.0, 1))) - np.array(B.regrets(0.5, B.fy_name(1.0, 5)))))   # je Seed gepaart
st.markdown(
    f"""
| Annahme | Was passiert, wenn sie verletzt ist |
|---|---|
| **σ passt zur Kostenskala** | Mit Rauschstufe 0,5 und einer Störung je Schritt liegt σ = 0,1 im Median bei {de(B.median(0.5, B.fy_name(0.1, 1)))} % Regret, KQ bei {de(B.median(0.5, 'kq'))} %: kein Gewinn gegenüber KQ (in der Stabilitätsprobe mit 100 weiteren Zufalls-Seeds im Median der Differenzen je Seed {de(_stab_diff, 2)} Punkte schlechter, besser in {_stab_better} von 100). Die Kosten dieser Welt liegen ohne Rauschen im Median bei 1,4 und zwischen dem 5. und 95. Perzentil bei 1,0 bis 3,9; in anderen Einheiten ist σ neu zu wählen (σ und Kosten mit demselben Faktor skaliert ändern die Wahl nicht). |
| **Der Löser ist billig** | Bei σ = 1 und Rauschstufe 0,5 bringt M = 5 gegen M = 1 je Seed gepaart im Median {de(_m_gain, 2)} Punkte (Mediane der Regrets {de(B.median(0.5, B.fy_name(1.0, 1)))} auf {de(B.median(0.5, B.fy_name(1.0, 5)))} %) für das Fünffache an Löseraufrufen (Experiment 5). Bei einem Löser, der Sekunden je Instanz braucht, ist das nicht gemessen. |
| **Die Vorhersage trifft die Wahrheit** | Selbst dann ist der gestörte Teilgradient im Mittel nicht null (Glättungsverzerrung, Experiment 4); ohne Rauschen und mit der wahren Vorhersage ist er es bei SPO+ exakt. |
| **Gauß-Störung** | Nur Gauß-Rauschen mit gleichem σ auf jedem Bogen; andere Verteilungen sind nicht gemessen. |
| **Das Modell ist die lineare Klasse** | Wie bei SPO+ rettet das Verfahren nur das falsche lineare Modell; eine flexible Modellklasse ist nicht gemessen. |
| **Eine Welt, ein Gitter** | Nur 5 × 5, Grad 6, 1000 Trainingsfälle, Rauschstufe 0 und 0,5, Adam mit fester Lernrate. Ob andere Gitter, Wahrheiten oder Trainer dasselbe zeigen, ist nicht gemessen. |
"""
)
st.caption(
    f"Zellen der Tabelle: Median über {len(c0['regret_kq'])} Zufalls-Seeds, Grad {C.DEG}, n = {C.N_TRAIN}, 300 Adam-Schritte; die Stabilitätsprobe mit {len(B.own()['stability']['seeds'])} weiteren Zufalls-Seeds steht in der README."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Gegeben und gesucht.** Gegeben sind Fälle $(x_i, c_i)$ mit Merkmalen $x_i \in \mathbb{R}^p$ und beobachteten Bogenkosten $c_i \in \mathbb{R}^d$ und ein Löser, der zu Kosten $c$ einen kürzesten Weg $w(c)$ liefert. Gesucht ist ein Modell $\hat c(x)$, dessen gewählter Weg unter den wahren Kosten möglichst gut ist.

| Symbol | Bedeutung |
|---|---|
| $p = 5$, $d = 40$ | Zahl der Merkmale, Zahl der Bögen im 5 × 5 Gitter |
| $c$, $\hat c$ | wahre und vorhergesagte Bogenkosten |
| $w(c)$ | der Weg (0/1-Vektor über die Bögen), den der Löser bei Kosten $c$ wählt |
| $w^*(c)$ | der beste Weg unter den wahren Kosten, also $w(c)$ |
| $\sigma$, $M$ | Stärke der Gauß-Störung, Zahl der Störungen je Fall und Schritt |
| $z_1, \dots, z_M$ | unabhängige Störungen aus $\mathcal N(0, I_d)$ |
| $\bar w(\hat c)$ | gemittelte gestörte Lösung |

**Gemittelte gestörte Lösung.**

$$\bar w(\hat c) = \frac{1}{M} \sum_{m=1}^{M} w(\hat c + \sigma z_m)$$

Für $M \to \infty$ ist das $\mathbb{E}_z\, w(\hat c + \sigma z)$; es ist glatt in $\hat c$ und hängt vom Löser nur über die Wahl $w(\cdot)$ ab (Black Box).

**Teilgradient (Handformel).**

$$g(\hat c, c) = w^*(c) - \bar w(\hat c)$$

Bei festen Störungen ist $g$ der Gradient nach $\hat c$ von $\hat c^\top w^*(c) - \frac{1}{M}\sum_m \min_w (\hat c + \sigma z_m)^\top w$ (Ableitung eines Minimums linearer Funktionen). Für das lineare Modell $\hat c = \theta^\top (1, x)$ folgt der Teilgradient nach $\theta$ aus der Kettenregel: $(1, x)\, g^\top$, gemittelt über die Fälle. Die genaue Form des Fenchel-Young-Verlustes der Quelle (Referenzverteilung, Normierung) ist hier nicht nachgebaut und nicht gegen den Primärtext geprüft; das Training braucht nur $g$.

**Zwei Wege.** Mit $\hat c = (\hat c_1, \hat c_2)$ wird Weg 1 gewählt, wenn $\hat c_1 + \sigma z_1 < \hat c_2 + \sigma z_2$, also $\sigma (z_1 - z_2) < \hat c_2 - \hat c_1$. Da $z_1 - z_2 \sim \mathcal N(0, 2)$ ist,

$$\mathbb{P}(\text{Weg 1}) = \Phi\!\left(\frac{\hat c_2 - \hat c_1}{\sigma \sqrt 2}\right).$$

Handbeispiel $c = (3, 5)$, $\hat c = (4, 3)$, $w^*(c) = (1, 0)$: bei $\sigma = 1$ ist $\Phi(-1/\sqrt 2) = 0{,}2398$, das Gewicht also $(0{,}2398;\, 0{,}7602)$ und $g = (0{,}7602;\, -0{,}7602)$; bei $\sigma = 3$ ist es $(0{,}4068;\, 0{,}5932)$ und $g = (0{,}5932;\, -0{,}5932)$; für $\sigma \to \infty$ geht es gegen $(0{,}5;\, 0{,}5)$, für $\sigma \to 0$ gegen den harten Löser.

**Skalierung.** Werden $\hat c$ und $\sigma$ mit demselben Faktor $\lambda > 0$ multipliziert, ändert sich keine gestörte Wahl: $w(\lambda \hat c + \lambda \sigma z) = w(\hat c + \sigma z)$.

**Vergleich SPO+.** $s(\hat c, c) = 2\,(w^*(c) - w(2\hat c - c))$ ist ein Teilgradient des konvexen SPO+-Verlustes (siehe `spo-kuerzester-weg-demo`); er braucht einen Löseraufruf je Fall und Schritt, die gestörte Wahl $M$.

**Löseraufwand.** In den Trainingsrunden werden $T \cdot M \cdot n$ Instanzen gelöst (SPO+: $T \cdot n$), plus einmalig $n$ für die besten Wege; mit $T = 300$ und $n = 1000$ sind das 300 000 (SPO+ und $M = 1$) und 1 500 000 ($M = 5$).

**Literatur.** Berthet, Blondel, Teboul, Cuturi, Vert, Bach, *Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020, arXiv 2002.08676: Rauschen auf den Eingängen, Lösungen mitteln, differenzierbar und nie lokal konstant; Verbindung zu Fenchel-Young-Verlusten (Blondel, Martins, Niculae 2019). **Nur Abstract-Ebene gelesen.** Neuere Verluste, nur Abstract-Ebene und nicht nachgebaut: Perturbation-Gradient-Verluste (Gupta, Huang, NeurIPS 2024, arXiv 2402.03256), LCGLN (Jeon u. a., arXiv 2403.01875), Dual-Guided Loss (Rodriguez-Diaz, Paulson, arXiv 2511.04909). Das Gitter und die Kostenformel sind aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut (Elmachtoub und Grigas, Management Science 68(1), 2022, nur Abstract-Ebene), nicht gegen sie geprüft.
        """
    )

sync_query_params({
    "noise_slider": float(noise), "seed_input": int(seed), "sigma_slider": float(sigma), "m_slider": int(m),
    "case_slider": int(case), "c1_slider": float(c1), "c2_slider": float(c2), "hsigma_slider": float(hsigma),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html)."
)
