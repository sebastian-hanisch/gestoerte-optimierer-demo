"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

# --- Das Gitter und die Welt (feste Annahmen: nur diese Zelle wurde vorgerechnet) --------------------------------------------------------------
GRID_H = 5                       # 5 x 5 Knoten: 40 Bögen, 70 Wege, je 8 Bögen
N_FEATURES = 5                   # Merkmale je Fall
DEG = 6                          # Grad der Wahrheit (feste Annahme dieser Demo)
N_TRAIN = 1000                   # Trainingsfälle (feste Annahme)
N_TEST = 1000                    # Testfälle je Welt
STEPS, LR = 300, 0.05            # Adam: Schritte und Lernrate
BAND_SEEDS = tuple(range(20))    # Zufalls-Seeds der vorgerechneten Bänder (gso_bands.json)
CURVE_STEPS = (0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 250, 300)
CHECK_STEPS = (0, 5, 10, 25, 50, 100, 300)

NOISE_CHOICES = (0.0, 0.5)       # Rauschstufe e
SIGMA_CHOICES = (0.1, 0.3, 1.0, 3.0)   # Stärke der Gauß-Störung (Einheit der Kosten; ohne Rauschen Median 1,4, 5. bis 95. Perzentil 1,0 bis 3,9)
M_CHOICES = (1, 5)               # Zahl der Störungen je Schritt und Fall
DEFAULT_NOISE, DEFAULT_SIGMA, DEFAULT_M, DEFAULT_SEED = 0.0, 1.0, 1, 0
SEED_MAX = 999999
CASE_MIN, CASE_MAX, DEFAULT_CASE = 1, 100, 1    # Nummer des gezeigten Testfalls (1 bis 100 von 1000)
SHOW_DRAWS = 300                 # Störungen für das Bild des gemittelten Weges im Gitter

# --- Das Handbeispiel (zwei Wege mit je einem Bogen) ---------------------------------------------------------------------------------------
HAND_C = (3.0, 5.0)              # wahre Kosten
CHAT_MIN, CHAT_MAX, CHAT_STEP = 0.0, 8.0, 0.5
DEFAULT_C1HAT, DEFAULT_C2HAT = 4.0, 3.0
HSIGMA_MIN, HSIGMA_MAX, HSIGMA_STEP, DEFAULT_HSIGMA = 0.1, 5.0, 0.1, 1.0
HAND_DRAWS = 2000                # Störungen des Stichprobenversuchs im Handbeispiel
HAND_SIGMA_GRID = 60             # Punkte der Kurve über σ

# --- Löseraufwand ------------------------------------------------------------------------------------------------------------------------------
TIMING_GRIDS = (5, 10, 15)       # Seitenlängen h: 40, 180, 420 Bögen
TIMING_INSTANCES = 1000

# --- Anzeige ------------------------------------------------------------------------------------------------------------------------------
COLOR_KQ = "#4c78a8"
COLOR_SPO = "#2F6B65"
COLOR_FY = "#d62728"
COLOR_FY5 = "#9467bd"
COLOR_ORACLE = "#7f7f7f"
COLOR_BEST = "#222222"
COLOR_BB = "#b8860b"

PRESETS = {
    "Rauschfreie Welt, σ = 1 (Voreinstellung)": {"noise": 0.0, "sigma": 1.0, "m": 1, "seed": 0},
    "Mit Rauschen, σ zu klein": {"noise": 0.5, "sigma": 0.1, "m": 1, "seed": 0},
    "Mit Rauschen, σ = 1": {"noise": 0.5, "sigma": 1.0, "m": 1, "seed": 0},
    "Fünf Störungen je Schritt": {"noise": 0.5, "sigma": 1.0, "m": 5, "seed": 0},
}
PRESET_HELP = {
    "Rauschfreie Welt, σ = 1 (Voreinstellung)": "Rauschstufe 0, Störung σ = 1, eine Störung je Schritt: die Kosten sind exakt bekannt, das lineare Modell passt trotzdem nicht zur nichtlinearen Wahrheit (Grad 6).",
    "Mit Rauschen, σ zu klein": "Rauschstufe 0,5, σ = 0,1, klein gegen die Kosten (Median 1,4): gegenüber KQ bringt das im Median über 20 Zufalls-Seeds keinen Gewinn.",
    "Mit Rauschen, σ = 1": "Rauschstufe 0,5, σ = 1 (etwa die Größe der Kosten), eine Störung je Schritt.",
    "Fünf Störungen je Schritt": "Rauschstufe 0,5, σ = 1, fünf Störungen je Schritt: das Fünffache an Löseraufrufen.",
}
