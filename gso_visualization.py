"""Plotly-Abbildungen (alle Achsen gesperrt, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gso_constants as C
from gso_evaluation import de, sig

_LOCK = dict(fixedrange=True)


def _lock(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(**_LOCK)
    fig.update_yaxes(**_LOCK)
    return fig


# --- Experiment 1: Gewicht und Teilgradient in Abhängigkeit von σ ----------------------------------------------------------------------------------

def build_hand_figure(curve: dict, h: dict, sigma: float) -> go.Figure:
    """Links: Gewicht auf Weg 1 gegen σ (geschlossene Form, Stichprobenversuch an vier Stellen, harter Löser als gestrichelte Linie); rechts: erste Komponente des Teilgradienten."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Gewicht auf Weg 1", "Teilgradient, Komponente Weg 1"), horizontal_spacing=0.1)
    s = curve["sigma"]
    fig.add_trace(go.Scatter(x=s, y=curve["w1"], mode="lines", line=dict(color=C.COLOR_FY, width=3), name="geschlossene Form", hovertemplate="σ = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[s[0], s[-1]], y=[curve["hard_w1"]] * 2, mode="lines", line=dict(color=C.COLOR_BEST, dash="dash"), name="harter Löser (σ = 0)", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=curve["mc_sigma"], y=curve["mc_w1"], mode="markers", marker=dict(color=C.COLOR_SPO, size=9, symbol="x"),
                             name=f"Mittel aus {C.HAND_DRAWS} Störungen", hovertemplate="σ = %{x}: %{y:.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=s, y=curve["g1"], mode="lines", line=dict(color=C.COLOR_FY, width=3), showlegend=False, hovertemplate="σ = %{x:.2f}: %{y:.3f}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[s[0], s[-1]], y=[h["gradient_hard"][0]] * 2, mode="lines", line=dict(color=C.COLOR_BEST, dash="dash"), showlegend=False, hoverinfo="skip"), row=1, col=2)
    for col, y in ((1, h["weights"][0]), (2, h["gradient"][0])):
        fig.add_trace(go.Scatter(x=[sigma], y=[y], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), name="Ihr σ", showlegend=(col == 1),
                                 hovertemplate="Ihr σ = %{x}: %{y:.3f}<extra></extra>"), row=1, col=col)
    fig.update_xaxes(title_text="σ", row=1, col=1)
    fig.update_xaxes(title_text="σ", row=1, col=2)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=70), legend=dict(orientation="h", y=-0.28, x=0.0))
    return _lock(fig)


# --- Experiment 2: Bänder gegen σ und ein Testfall im Gitter --------------------------------------------------------------------------------------

def build_sigma_figure(rows: list[dict], chosen_sigma: float) -> go.Figure:
    """Regret (Median, Balken von Minimum bis Maximum über die Seeds) für KQ, SPO+ und jedes σ; das gewählte σ ist hervorgehoben, das Orakel ist eine Linie."""
    body = [r for r in rows if r["kind"] != "orakel"]
    names = ["KQ" if r["kind"] == "kq" else "SPO+" if r["kind"] == "spo" else f"σ = {sig(r['sigma'])}" for r in body]
    colors = [C.COLOR_KQ if r["kind"] == "kq" else C.COLOR_SPO if r["kind"] == "spo" else (C.COLOR_FY if r["sigma"] == chosen_sigma else "#e8a09f") for r in body]
    med = [r["stat"]["median"] for r in body]
    fig = go.Figure(go.Bar(x=names, y=med, marker_color=colors, showlegend=False, hovertemplate="%{x}: %{y:.2f} %<extra></extra>",
                           error_y=dict(type="data", symmetric=False, array=[r["stat"]["max"] - r["stat"]["median"] for r in body], arrayminus=[r["stat"]["median"] - r["stat"]["min"] for r in body])))
    oracle = next(r for r in rows if r["kind"] == "orakel")["stat"]["median"]
    fig.add_hline(y=oracle, line=dict(color=C.COLOR_ORACLE, dash="dot"), annotation_text="Orakel", annotation_position="bottom right")
    fig.update_yaxes(title_text="Regret (%), Median und Spanne")
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=20, b=40))
    return _lock(fig)


def build_show_figure(grid, info: dict) -> go.Figure:
    """5 x 5 Gitter: Linienbreite = Anteil der gestörten Lösungen, die den Bogen benutzen; der Weg des harten Lösers (ohne Störung) gepunktet."""
    h = grid.h
    fig = go.Figure()
    freq = info["freq"]
    for k, (a, b) in enumerate(grid.edges):
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color="#cccccc", width=1), hoverinfo="skip", showlegend=False))
    for k, (a, b) in enumerate(grid.edges):
        if freq[k] <= 0:
            continue
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=C.COLOR_FY, width=1 + 11 * float(freq[k])), opacity=0.55,
                                 hovertext=f"Bogen {k + 1}: Anteil der Störungen {de(100 * freq[k], 0)} %", hoverinfo="text", showlegend=False))
    xs, ys = [], []
    for k in np.flatnonzero(info["hard"]):
        a, b = grid.edges[k]
        (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=C.COLOR_BEST, width=3, dash="dot"), name="Weg des harten Lösers", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[grid.node_xy(v)[0] for v in range(grid.nodes)], y=[grid.node_xy(v)[1] for v in range(grid.nodes)], mode="markers",
                             marker=dict(size=9, color="#333", line=dict(color="white", width=1)), hoverinfo="skip", showlegend=False))
    for k, (a, b) in enumerate(grid.edges):
        if freq[k] >= 0.1:
            (x0, y0), (x1, y1) = grid.node_xy(a), grid.node_xy(b)
            fig.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=de(100 * freq[k], 0) + " %", showarrow=False, font=dict(size=9), bgcolor="rgba(255,255,255,0.88)", bordercolor="rgba(0,0,0,0)")
    fig.add_annotation(x=0, y=0, text="Start", showarrow=False, xshift=-22, yshift=-12, font=dict(size=11))
    fig.add_annotation(x=h - 1, y=h - 1, text="Ziel", showarrow=False, xshift=20, yshift=12, font=dict(size=11))
    fig.update_xaxes(visible=False, range=[-0.35, h - 0.65])
    fig.update_yaxes(visible=False, range=[h - 0.45, -0.45])
    fig.update_layout(height=400, margin=dict(l=5, r=5, t=5, b=5), legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center", yanchor="top"))
    return _lock(fig)


# --- Experiment 3: Lernkurven --------------------------------------------------------------------------------------------------------------------

def build_curve_figure(steps: list[int], spo: list[float], fy: list[float], r_kq: float, r_oracle: float, sigma: float, m: int) -> go.Figure:
    """Regret (%) auf den Testfällen nach t Adam-Schritten: SPO+ und gestörte Lösungen, KQ-Start und Orakel als Linien."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=spo, mode="lines+markers", line=dict(color=C.COLOR_SPO, width=3), name="SPO+", hovertemplate="Schritt %{x}: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=steps, y=fy, mode="lines+markers", line=dict(color=C.COLOR_FY, width=3), name=f"gestörte Lösungen (σ = {sig(sigma)}, M = {m})", hovertemplate="Schritt %{x}: %{y:.2f} %<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, max(steps)], y=[r_kq] * 2, mode="lines", line=dict(color=C.COLOR_KQ, dash="dash"), name="KQ (Start)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0, max(steps)], y=[r_oracle] * 2, mode="lines", line=dict(color=C.COLOR_ORACLE, dash="dot"), name="Orakel", hoverinfo="skip"))
    fig.update_xaxes(title_text="Adam-Schritte")
    fig.update_yaxes(title_text="Regret auf den Testfällen (%)")
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=20, b=70), legend=dict(orientation="h", y=-0.3, x=0.0))
    return _lock(fig)


# --- Experiment 4: Glättungsverzerrung -------------------------------------------------------------------------------------------------------------

def build_bias_figure(rows_by_noise: dict) -> go.Figure:
    """Zwei Teilbilder (Rauschstufe 0 und 0,5): Anteil des systematischen Teilgradienten an der wahren Vorhersage (Median über die Seeds, Balken Minimum bis Maximum)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=tuple(f"Rauschstufe {de(n)}" for n in rows_by_noise), horizontal_spacing=0.1)
    for col, (noise, rows) in enumerate(rows_by_noise.items(), start=1):
        names = [("SPO+" if r["key"] == "SPO+" else f"σ = {r['key'].split('= ')[1]}".replace(".", ",")) for r in rows]
        med = [r["stat"]["median"] for r in rows]
        colors = [C.COLOR_SPO if r["key"] == "SPO+" else C.COLOR_FY for r in rows]
        fig.add_trace(go.Bar(x=names, y=med, marker_color=colors, showlegend=False, hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                             error_y=dict(type="data", symmetric=False, array=[r["stat"]["max"] - r["stat"]["median"] for r in rows], arrayminus=[r["stat"]["median"] - r["stat"]["min"] for r in rows])), row=1, col=col)
        fig.update_yaxes(range=[0, 1.05], title_text="Anteil (0 = im Mittel null)" if col == 1 else None, row=1, col=col)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=40))
    return _lock(fig)


# --- Experiment 5: Regret gegen gelöste Instanzen ----------------------------------------------------------------------------------------------------

def build_effort_figure(rows: list[dict]) -> go.Figure:
    """Regret (Median) gegen die Zahl der gelösten Instanzen (logarithmische Achse) nach 25 / 50 / 100 / 300 Schritten; KQ braucht keinen Löser und ist eine Linie."""
    fig = go.Figure()
    steps = (25, 50, 100, 300)
    n = C.N_TRAIN
    colors = {"SPO+": C.COLOR_SPO}
    for r in rows:
        if r["key"] in ("kq", "bb"):
            continue
        per_step = r["loop"] // C.STEPS
        xs = [per_step * s for s in steps]
        color = colors.get(r["key"], C.COLOR_FY if r["key"].endswith("|1") else C.COLOR_FY5)
        fig.add_trace(go.Scatter(x=xs, y=[r["regret"][s] for s in steps], mode="lines+markers", line=dict(color=color, width=3), name=r["Verfahren"],
                                 hovertemplate="%{x:,} gelöste Instanzen: %{y:.2f} %<extra></extra>"))
    kq = rows[0]["regret"][300]
    fig.add_hline(y=kq, line=dict(color=C.COLOR_KQ, dash="dash"), annotation_text="KQ (kein Löser)", annotation_position="top left")
    fig.update_xaxes(title_text=f"gelöste Instanzen in den Trainingsrunden (n = {n}; Punkte: 25, 50, 100, 300 Schritte)", type="log")
    fig.update_yaxes(title_text="Regret (%), Median")
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=20, b=70), legend=dict(orientation="h", y=-0.32, x=0.0))
    return _lock(fig)
