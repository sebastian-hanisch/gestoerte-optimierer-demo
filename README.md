# Gestörte Optimierer und Fenchel-Young: Rauschen, mitteln, lernen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gestoerte-optimierer-demo.streamlit.app/)**

Stück der Reihe **Entscheidungsorientiertes Lernen** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“. Vehikel ist ein abstraktes **5 × 5 Gitter** (Wege von links oben nach rechts unten, 40 Bögen, 70 Wege) wie in `spo-kuerzester-weg-demo`: Aus Merkmalen sollen Bogenkosten vorhergesagt werden, und ein exakter Löser wählt daraus den kürzesten Weg. Die Wahl des Lösers ist stückweise konstant in der Vorhersage; ein **gestörter Optimierer** stört die Vorhersage mit **Gauß-Rauschen**, löst mehrfach und **mittelt die Lösungen**. Die Demo trainiert dasselbe lineare Modell mit **Kleinsten Quadraten (KQ)**, mit **SPO+** und mit dem Teilgradienten der gemittelten gestörten Lösung, und sie zählt, **wie oft der Löser dafür laufen muss**.

## Warum dieses Problem

Der Löser (hier eine Rückwärtsrechnung, in der Praxis oft ein Optimierungsprogramm) ist eine Black Box, die zu Kosten einen Weg liefert. Der Weg ändert sich nur an Sprungstellen der Kosten; dazwischen ist die Ableitung null. Mittelt man dagegen die Lösungen über gestörte Kosten, ändert sich das Mittel stetig mit der Vorhersage, und der Teilgradient hat die einfache Form „optimale Lösung minus gemittelte gestörte Lösung“. Die Kernfrage der Demo: **Welche Stärke σ der Störung reicht, und was kostet sie an Löseraufrufen?**

- **σ muss groß genug gegenüber den Kosten dieser Welt sein.** Mit Rauschstufe 0,5 und einer Störung je Schritt gewinnen σ = 1 und σ = 3 gegenüber KQ in allen 20 Zufalls-Seeds; σ = 0,1 bringt im Median keinen Gewinn (13,1 gegen 13,1 %, in 7 von 20 Seeds besser) und liegt in einer Stabilitätsprobe mit 100 weiteren Zufalls-Seeds im Median 0,28 Punkte über KQ (Tabellen unten). Die beobachteten Kosten je Bogen liegen ohne Rauschen im Median bei 1,4 und zwischen dem 5. und 95. Perzentil bei 1,0 bis 3,9 (Median der Perzentile über 20 Welten). Gemessen ist nur dieses σ-Raster auf einer Kostenskala; wie die gute Zone wandert, wenn die Kosten anders skaliert sind, ist nicht gemessen.
- **Gegenüber SPO+ ist der Unterschied klein:** mit Rauschen liegt σ = 1 bei einer Störung gleichauf (12,2 gegen 12,2 %); in der Stabilitätsprobe ist es im Median 0,10 Punkte besser, mit σ = 3 oder fünf Störungen 0,2 bis 0,3 Punkte. Ohne Rauschen hat SPO+ den kleinsten Median-Regret (0,52 gegen 0,57 bis 0,63 %); der Abstand liegt unter der Streuung zwischen den Seeds.
- **Der Preis:** SPO+ löst in 300 Schritten 300 000 Instanzen, das gestörte Training mit einer Störung ebenso viele, mit fünf Störungen 1 500 000. Fünf Störungen statt einer senken den Regret bei σ = 1 und Rauschen je Seed im Median um etwa 0,1 Punkte (20 Seeds 0,09, besser in 13 von 20; Stabilitätsprobe 0,14, besser in 72 von 100), ohne Rauschen nicht (0,01 Punkte schlechter).

## Modell

**Welt** (unverändert aus `spo-kuerzester-weg-demo`, dort aus dem Gedächtnis nach dem Schema der SPO-Literatur nachgebaut, nicht gegen sie geprüft): Merkmale $x \sim \mathcal N(0, I_5)$; eine Gewichtsmatrix $B \in \{0,1\}^{40 \times 5}$ mit fairen Münzwürfen; mittlere Kosten $m_j(x) = ((Bx)_j/\sqrt{5} + 3)^{6} / 3{,}5^{6} + 1$ (Grad 6 fest); beobachtete Kosten $c_j = m_j(x)\,\varepsilon_j$ mit $\varepsilon_j \sim U[1 - e, 1 + e]$, Rauschstufe $e$ = 0 oder 0,5. 1000 Trainingsfälle, 1000 Testfälle. Das **Orakel** kennt $m(x)$.

**Löser.** Rückwärtsrechnung über die 25 Knoten, vektorisiert über alle Fälle; Gleichstand: am Knoten „rechts“ vor „unten“.

**Gestörte Lösung.** Mit Vorhersage $\hat c$, Stärke $\sigma$ und $M$ Störungen $z_m \sim \mathcal N(0, I_{40})$ ist $\bar w(\hat c) = \frac{1}{M}\sum_m w(\hat c + \sigma z_m)$ die gemittelte gestörte Lösung. **Teilgradient** (nach $\hat c$, Handformel): $w^*(c) - \bar w(\hat c)$. Er ist bei festen Störungen der Gradient von $\hat c^\top w^*(c) - \frac{1}{M}\sum_m \min_w (\hat c + \sigma z_m)^\top w$ (Ableitung eines Minimums linearer Funktionen). Die Störungen kommen aus dem ganzzahligen SplitMix64 (Box-Muller aus zwei Gleichverteilten); ein Lauf ist bei festem Zufalls-Seed reproduzierbar.

**Zwei Wege von Hand.** Bei zwei Wegen mit je einem Bogen und $\hat c = (\hat c_1, \hat c_2)$ wird Weg 1 gewählt, wenn $\sigma(z_1 - z_2) < \hat c_2 - \hat c_1$; $z_1 - z_2$ ist $\mathcal N(0, 2)$-verteilt, also hat Weg 1 das Gewicht $\Phi\big((\hat c_2 - \hat c_1)/(\sigma\sqrt 2)\big)$. Handbeispiel wahre Kosten $c = (3, 5)$, $\hat c = (4, 3)$, $w^*(c) = (1, 0)$: σ = 1 gibt das Gewicht (0,2398; 0,7602) und den Teilgradienten (0,7602; −0,7602); σ = 3 gibt (0,4068; 0,5932) und (0,5932; −0,5932); σ = 0,1 praktisch (0; 1) und (1; −1), also den harten Löser. Für σ → ∞ geht das Gewicht gegen (0,5; 0,5), für σ → 0 gegen den harten Löser.

**Verfahren im Vergleich.** Lineares Modell mit Achsenabschnitt (6 × 40 = 240 Parameter). KQ über `numpy.linalg.lstsq`. SPO+ und das Training mit gestörten Lösungen: Adam (Lernrate 0,05, Vollbatch, **300 Schritte**), Start bei der KQ-Lösung; verglichen wird der **Parameter nach t Schritten** (das gestörte Training hat keinen Verlustwert und keine Auswahl nach kleinstem Trainingsverlust).

**Zählregel des Löseraufwands.** Gezählt werden gelöste Instanzen (Zeilen, die an den Löser gehen), nicht Sekunden. „In der Schleife“ sind alle Instanzen der Trainingsrunden: SPO+ $T \cdot n$ (ein Aufruf auf $2\hat c - c$), gestörte Lösungen $T \cdot M \cdot n$; die einmalige Vorabrechnung der besten Wege $w^*(c)$ ($n$ Instanzen) steht getrennt, die Auswertung auf den Testfällen zählt nicht mit. Mit $T = 300$ und $n = 1000$: 300 000, 300 000 (M = 1) und 1 500 000 (M = 5). Die Zeit je 1000 Instanzen misst die App nur live (Tab „Zeit je 1000 Instanzen“ in Experiment 5) und schreibt sie nirgends fest.

## Methodik

- **Auswertung.** Jede Zelle (Rauschstufe 0 und 0,5) läuft mit den festen Zufalls-Seeds 0 bis 19, 1000 Testfällen, 5 Merkmalen, Grad 6, n = 1000. Angegeben werden **Median und Spanne (Minimum bis Maximum)** über die Seeds, nie ein einzelner Seed. Die Bänder stehen vorgerechnet in `gso_bands.json` und sind mit `python tools/measure_bands.py` reproduzierbar (rund 20 Minuten mit zwei Prozessen; Zufall nur über den ganzzahligen SplitMix64 aus `gso_rng.py`, nie Pythons `hash()`).
- **Stabilitätsprobe.** Die Kernvergleiche (Rauschstufe 0,5) laufen mit **100 weiteren Zufalls-Seeds (20 bis 119)** noch einmal, weil die Unterschiede zwischen den Verfahren kleiner sind als die Streuung zwischen Seeds. „Differenz“ heißt dort: je Seed Regret des Verfahrens minus Regret des Vergleichs, dann der Median über die Seeds (nicht die Differenz der Mediane).
- **Glättungsverzerrung.** 60 Merkmalsvektoren je Zufalls-Seed, je 400 Kostenziehungen, Vorhersage = wahre Mittelkosten; Maß je Merkmalsvektor: Norm des Mittels der Teilgradienten geteilt durch das Mittel der Normen (0 = im Mittel kein Gradient, 1 = alle Ziehungen in dieselbe Richtung), Median über die 60 Merkmalsvektoren, dann Median und Spanne über 20 Seeds. Gestörte Lösungen mit $M = 1$.
- **Trainierte Zahlen sind plattformabhängig** (Adam, Gleitkomma, Löserentscheidungen bei knappen Unterschieden, `log` und `cos` der C-Bibliothek im Box-Muller-Schritt). Die Tests prüfen sie deshalb nur als Bänder und Ungleichungen; exakt geprüft werden nur Rechnungen ohne Training (Handbeispiel, Zählung).
- **Stärke σ und Zahl M** sind in der App auf die vorgerechneten Werte σ ∈ {0,1; 0,3; 1; 3} und M ∈ {1, 5} beschränkt; die Lernkurve in Experiment 3 wird live für den gewählten Zufalls-Seed gerechnet.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen dieses Abschnitts stammen aus `gso_bands.json` (eigene Messung: lineares Modell, Grad 6, n = 1000, 300 Adam-Schritte, Zufalls-Seeds 0 bis 19, 1000 Testfälle, Regret in Prozent als Median [Minimum; Maximum]), wenn nichts anderes dabeisteht. Die Tabellen werden von `tests/readme_tables.py` aus der JSON-Datei erzeugt, und `tests/test_claims.py` prüft, dass jede Zeile wörtlich hier steht; die Aussagen darunter stehen als Bänder und Ungleichungen im selben Test.

### Regret ohne Rauschen (Rauschstufe 0)

| Verfahren | 25 Schritte | 50 Schritte | 100 Schritte | 300 Schritte [Minimum; Maximum] | besser als KQ | gelöste Instanzen |
|---|---|---|---|---|---|---|
| Orakel | 0,00 | 0,00 | 0,00 | 0,00 [0,00; 0,00] | – | – |
| KQ | 1,96 | 1,96 | 1,96 | 1,96 [1,38; 3,00] | – | 0 |
| SPO+ (300 Schritte) | 0,53 | 0,52 | 0,53 | 0,52 [0,37; 0,83] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,1, M = 1 | 0,67 | 0,62 | 0,63 | 0,63 [0,45; 0,86] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,3, M = 1 | 0,58 | 0,63 | 0,60 | 0,60 [0,39; 0,95] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 1, M = 1 | 0,61 | 0,59 | 0,58 | 0,58 [0,33; 0,92] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 3, M = 1 | 0,98 | 0,69 | 0,62 | 0,59 [0,38; 0,99] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,1, M = 5 | 0,64 | 0,58 | 0,57 | 0,60 [0,36; 0,91] | 20 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 0,3, M = 5 | 0,59 | 0,60 | 0,58 | 0,61 [0,37; 0,95] | 20 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 1, M = 5 | 0,61 | 0,59 | 0,56 | 0,57 [0,36; 0,89] | 20 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 3, M = 5 | 0,98 | 0,68 | 0,61 | 0,57 [0,36; 0,89] | 20 von 20 | 1 500 000 |

### Regret mit Rauschen (Rauschstufe 0,5)

| Verfahren | 25 Schritte | 50 Schritte | 100 Schritte | 300 Schritte [Minimum; Maximum] | besser als KQ | gelöste Instanzen |
|---|---|---|---|---|---|---|
| Orakel | 10,54 | 10,54 | 10,54 | 10,54 [9,31; 12,32] | – | – |
| KQ | 13,08 | 13,08 | 13,08 | 13,08 [11,28; 14,48] | – | 0 |
| SPO+ (300 Schritte) | 12,30 | 12,26 | 12,31 | 12,20 [10,71; 13,35] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,1, M = 1 | 13,62 | 12,90 | 12,88 | 13,07 [11,42; 15,01] | 7 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,3, M = 1 | 12,54 | 12,51 | 12,41 | 12,55 [10,44; 14,38] | 17 von 20 | 300 000 |
| Gestörte Lösungen, σ = 1, M = 1 | 12,19 | 12,00 | 12,08 | 12,22 [10,36; 14,29] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 3, M = 1 | 11,79 | 11,90 | 11,85 | 11,93 [10,37; 14,25] | 20 von 20 | 300 000 |
| Gestörte Lösungen, σ = 0,1, M = 5 | 13,19 | 12,29 | 12,33 | 12,22 [10,84; 14,19] | 19 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 0,3, M = 5 | 12,00 | 12,10 | 12,13 | 11,96 [10,99; 13,54] | 20 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 1, M = 5 | 11,98 | 12,09 | 11,81 | 11,93 [10,48; 13,71] | 20 von 20 | 1 500 000 |
| Gestörte Lösungen, σ = 3, M = 5 | 11,82 | 11,99 | 11,94 | 11,89 [10,27; 13,68] | 20 von 20 | 1 500 000 |

„besser als KQ“ zählt die Seeds (von 20), in denen der Regret nach 300 Schritten streng unter dem von KQ liegt. Der Regret nach 25, 50 und 100 Schritten ist der Median des Parameters nach so vielen Adam-Schritten; KQ und Orakel haben keine Schritte.

### Teilgradient an der wahren Vorhersage (Glättungsverzerrung)

Anteil des systematischen Teilgradienten, wenn die Vorhersage die wahren bedingten Mittelkosten trifft (Median [Minimum; Maximum] über 20 Zufalls-Seeds, gestörte Lösungen mit M = 1):

| Verfahren | Rauschstufe 0 | Rauschstufe 0,5 |
|---|---|---|
| SPO+ | 0,00 [0,00; 0,00] | 0,05 [0,04; 0,05] |
| Gestörte Lösungen, σ = 0,1 | 0,75 [0,72; 0,79] | 0,40 [0,35; 0,50] |
| Gestörte Lösungen, σ = 0,3 | 0,70 [0,69; 0,72] | 0,11 [0,09; 0,16] |
| Gestörte Lösungen, σ = 1 | 0,70 [0,69; 0,71] | 0,13 [0,11; 0,15] |
| Gestörte Lösungen, σ = 3 | 0,71 [0,70; 0,72] | 0,21 [0,19; 0,26] |

### Stabilitätsprobe (Rauschstufe 0,5, 100 weitere Zufalls-Seeds 20 bis 119, Regret nach 300 Schritten)

| Verfahren | Regret (Median) | besser als KQ | Median der Differenz zu KQ | besser als SPO+ | Median der Differenz zu SPO+ |
|---|---|---|---|---|---|
| KQ | 12,65 % | – | – | – | – |
| SPO+ | 12,14 % | 88 von 100 | -0,64 Punkte | – | – |
| Gestörte Lösungen, σ = 0,1, M = 1 | 13,16 % | 41 von 100 | 0,28 Punkte | 8 von 100 | 0,80 Punkte |
| Gestörte Lösungen, σ = 1, M = 1 | 12,04 % | 89 von 100 | -0,75 Punkte | 61 von 100 | -0,10 Punkte |
| Gestörte Lösungen, σ = 3, M = 1 | 11,87 % | 93 von 100 | -0,88 Punkte | 75 von 100 | -0,23 Punkte |
| Gestörte Lösungen, σ = 1, M = 5 | 11,81 % | 98 von 100 | -0,95 Punkte | 75 von 100 | -0,26 Punkte |

### Löseraufwand (n = 1000, 300 Schritte)

| Verfahren | gelöste Instanzen (300 Schritte) | einmalig vorab |
|---|---|---|
| KQ | 0 | 0 |
| SPO+ | 300 000 | 1 000 |
| Gestörte Lösungen, M = 1 | 300 000 | 1 000 |
| Gestörte Lösungen, M = 5 | 1 500 000 | 1 000 |
| Löser als Black Box (nur Zahl) | 600 000 | – |

Die Black-Box-Zeile ist nur eine Zahl (zwei Löseraufrufe je Fall und Schritt, wie bei einer Interpolation durch den Löser); dieses Verfahren ist in dieser Demo nicht trainiert und hat keinen Regret, die Zahl stammt aus der Vormessung (`out_eo_v7_9_stoerung_bb_aufwand.txt`) und ist hier nachgerechnet (300 · 1000 · 2).

| Frage | Befund |
|---|---|
| **Reicht σ?** | Mit Rauschen (Rauschstufe 0,5, eine Störung je Schritt) liegt der Median-Regret bei σ = 1 und σ = 3 unter dem von KQ in 20 von 20 Seeds (12,2 und 11,9 gegen 13,1 %), bei σ = 0,3 in 17 von 20 (12,5 %), bei σ = 0,1 in 7 von 20 (13,1 %, gleich KQ). Ohne Rauschen liegen alle acht Einstellungen in 20 von 20 Seeds unter KQ (0,57 bis 0,63 gegen 1,96 %). |
| **Ist σ = 0,1 schlechter als KQ?** | Im Median über die 20 Seeds nicht (13,07 gegen 13,08 %). In der Stabilitätsprobe mit 100 weiteren Zufalls-Seeds (Rauschstufe 0,5) ist σ = 0,1 mit einer Störung in 41 von 100 Seeds besser als KQ und im Median der Differenzen je Seed 0,28 Punkte schlechter (Mediane der Regrets 13,16 gegen 12,65 %). Mit fünf Störungen ist es in 19 von 20 Seeds besser als KQ (12,2 %). |
| **Schlägt es SPO+?** | Nicht sicher. Mit Rauschen: σ = 1 und eine Störung 12,22 gegen 12,20 % (20 Seeds); in der Stabilitätsprobe besser als SPO+ in 61 von 100 Seeds, im Median 0,10 Punkte; σ = 3 in 75 von 100 (0,23 Punkte), σ = 1 mit fünf Störungen in 75 von 100 (0,26 Punkte). Ohne Rauschen hat SPO+ den kleineren Median-Regret als jede der acht Einstellungen (0,52 gegen 0,57 bis 0,63 %); der Abstand liegt unter der Streuung zwischen den Seeds. |
| **Was bringen fünf Störungen?** | Mit Rauschen bei σ = 1: Mediane der Regrets 12,22 → 11,93 %; je Seed gepaart im Median 0,09 Punkte besser (13 von 20 Seeds). Stabilitätsprobe: Mediane 12,04 → 11,81 %, gepaart 0,14 Punkte (72 von 100). Ohne Rauschen bei σ = 1: 0,58 → 0,57 %, gepaart 0,01 Punkte schlechter (besser in 9 von 20). Preis: das Fünffache an gelösten Instanzen (1 500 000 statt 300 000). |
| **Bleibt der Teilgradient an der Wahrheit im Mittel null?** | SPO+: ohne Rauschen exakt 0, mit Rauschen im Median 0,05. Gestörte Lösungen: ohne Rauschen 0,70 bis 0,75 für alle σ (die Störung verschmiert die Lösung, auch wenn die Vorhersage stimmt), mit Rauschen 0,40 (σ = 0,1), 0,11 (σ = 0,3), 0,13 (σ = 1), 0,21 (σ = 3). |
| **Wie schnell trainiert SPO+, und was heißt das für die Löseraufrufe?** | Mit Rauschen liegt SPO+ nach 25 Schritten (25 000 gelöste Instanzen) bei 12,30 %, nach 300 Schritten (300 000) bei 12,20 % (Median); ohne Rauschen 0,53 gegen 0,52 %. Die gestörte Lösung mit σ = 1 und M = 1 liegt mit Rauschen nach 25 Schritten bei 12,19 %, nach 300 bei 12,22 %; σ = 3 startet ohne Rauschen langsamer (0,98 % nach 25, 0,59 % nach 300 Schritten). Gemessen sind Schritte und gelöste Instanzen, keine Sekunden. |
| **Wie viel der Lücke zwischen KQ und Orakel schließt das Verfahren?** | Mit Rauschen: Orakel 10,5 %, KQ 13,1 %; SPO+ schließt 35 % der Lücke, σ = 3 mit einer Störung 45 % (aus den ungerundeten Medianen berechnet). Ohne Rauschen schließen SPO+ 73 % und σ = 1 mit einer Störung 70 %. Der Regret des Orakels ist der unvermeidbare Teil und steht nicht in der Lücke; der Abstand zum Orakel bleibt, ob er an der linearen Klasse, an der Datenmenge oder am Training liegt, ist nicht untersucht. |

### Handbeispiel (zwei Wege, exakt, ohne Training)

Wahre Kosten $c = (3, 5)$, $\hat c = (4, 3)$, $w^*(c) = (1, 0)$:

| σ | Gewicht (Weg 1; Weg 2) | Teilgradient |
|---|---|---|
| 0,1 | (0,0000; 1,0000) | (1,0000; −1,0000) |
| 1 | (0,2398; 0,7602) | (0,7602; −0,7602) |
| 3 | (0,4068; 0,5932) | (0,5932; −0,5932) |

### Vormessung und diese Demo (Rauschstufe 0,5, Regret nach 300 Schritten, Median über 20 Zufalls-Seeds)

Vormessung: `out_eo_v7_9_stoerung_bb_aufwand.txt` (eingefroren in `gso_vormessung.json`, `tools/freeze_vormessung.py`; eine andere Zufallsquelle, gestörte Lösungen aus `numpy.random`, Parameter nach 300 Schritten). Nicht gleich, nur vergleichbar:

| Verfahren | Rauschstufe 0, Vormessung | Rauschstufe 0, diese Demo | Rauschstufe 0,5, Vormessung | Rauschstufe 0,5, diese Demo |
|---|---|---|---|---|
| KQ | 1,87 | 1,96 | 12,47 | 13,08 |
| SPO+ | 0,54 | 0,52 | 11,84 | 12,20 |
| Gestörte Lösungen, σ = 0,1, M = 1 | 0,61 | 0,63 | 12,83 | 13,07 |
| Gestörte Lösungen, σ = 0,3, M = 1 | 0,65 | 0,60 | 12,34 | 12,55 |
| Gestörte Lösungen, σ = 1, M = 1 | 0,60 | 0,58 | 11,74 | 12,22 |
| Gestörte Lösungen, σ = 0,3, M = 5 | 0,61 | 0,61 | 11,64 | 11,96 |
| Gestörte Lösungen, σ = 1, M = 5 | 0,58 | 0,57 | 11,63 | 11,93 |

## Befunde und Korrekturen gegenüber dem Plan

- **Die Zahlen der Vormessung reproduzieren sich nicht auf die Stelle.** Diese Demo benutzt eine andere Zufallsquelle (SplitMix64 mit Box-Muller statt `numpy.random`), eine andere Welt je Zufalls-Seed und das SPO+-Training mit der Auswahl „Parameter nach 300 Schritten“. Mit Rauschen liegen KQ und SPO+ höher (13,08 und 12,20 gegen 12,47 und 11,84 %), ohne Rauschen nahezu gleich (KQ 1,96 gegen 1,87, SPO+ 0,52 gegen 0,54 %). Es gilt die neue Messung; die Rangfolge KQ, SPO+ ist dieselbe.
- **„σ = 0,1 mit Rauschen schlechter als KQ“ (Plan: 12,83 gegen 12,47 %) stimmt nur schwach.** Im Median über die 20 Zufalls-Seeds sind beide gleich (13,07 und 13,08 %), in 7 von 20 Seeds ist σ = 0,1 besser. In einer Stabilitätsprobe mit 100 weiteren Zufalls-Seeds ist es im Median 0,28 Punkte schlechter, aber in 41 von 100 Seeds besser. Gesagt wird deshalb: kein Gewinn gegenüber KQ, im Median bei mehr Seeds etwas schlechter.
- **„σ = 1 auf SPO+-Niveau und etwas besser“ (Plan: 11,74 gegen 11,84 %) stimmt in den 20 Zufalls-Seeds nicht:** 12,22 gegen 12,20 %. In der Stabilitätsprobe ist σ = 1 mit einer Störung im Median 0,10 Punkte besser (61 von 100 Seeds), σ = 3 und σ = 1 mit fünf Störungen 0,23 und 0,26 Punkte (je 75 von 100).
- **„Fünf Störungen bringen etwa 0,1 Punkte“ (Plan: 11,74 → 11,63 %)** bestätigen sich, wenn je Seed gepaart wird: 0,09 Punkte (20 Seeds) und 0,14 Punkte in der Stabilitätsprobe. Die Differenz der Mediane (12,22 → 11,93 %, 0,29 Punkte; 12,04 → 11,81 %, 0,23 Punkte) ist größer und nicht das Maß dieser Demo. Ohne Rauschen bleibt es bei 0,01 Punkten.
- **σ = 3 ist neu** (in der Vormessung nicht gemessen): mit Rauschen der kleinste Median-Regret der Einstellungen mit einer Störung (11,93 %), ohne Rauschen langsamer am Anfang (0,98 % nach 25 Schritten).
- **Glättungsverzerrung:** Die Vormessung nannte für gestörte Lösungen mit Rauschen 0,10 und 0,114 (σ = 0,3 und 1), ohne Rauschen 0,71 und 0,706, für SPO+ 0,049 und 0,0 (eine Welt). Diese Demo misst über 20 Welten 0,11 und 0,13 mit Rauschen, 0,70 ohne Rauschen und 0,05 und 0,00 für SPO+; σ = 0,1 und σ = 3 sind neu (0,40 und 0,21 mit Rauschen). Das Verhältnis ist bei σ = 0,1 und 3 mit Rauschen größer als bei σ = 0,3 und 1.
- **Löserzeiten der Vormessung nicht übernommen:** Plan und Vormessung nennen Millisekunden je 1000 Instanzen für drei Gittergrößen. Zeiten hängen vom Rechner ab und stehen deshalb weder hier noch in der App; die App misst sie auf Knopfdruck live.
- **Der „Tab Löseraufwand“** ist in der App Experiment 5 mit zwei Reitern („Gelöste Instanzen“ und „Zeit je 1000 Instanzen“).

## Ehrliche Grenzen

- **Quellen nur auf Abstract-Ebene.** Berthet u. a. (*Learning with Differentiable Perturbed Optimizers*, NeurIPS 2020, arXiv 2002.08676) und Blondel, Martins, Niculae (2019) habe ich nur auf der Ebene des Abstracts gelesen. Die genaue Form des Fenchel-Young-Verlustes (Referenzverteilung, Normierung) ist nicht geprüft und nicht nachgebaut; die Demo benutzt nur den Teilgradienten als Handformel und keinen Verlustwert. Die neueren Verluste (Perturbation-Gradient-Verluste von Gupta und Huang, NeurIPS 2024, arXiv 2402.03256; LCGLN von Jeon u. a., arXiv 2403.01875; Dual-Guided Loss von Rodriguez-Diaz und Paulson, arXiv 2511.04909) sind nur auf Abstract-Ebene eingeordnet und nicht nachgebaut; ob sie hier besser abschneiden würden, ist nicht gemessen.
- **Nur Gauß-Störung** mit gleichem σ auf jedem Bogen; andere Verteilungen sind nicht gemessen. σ ist auf die Größe der Kosten dieser Welt bezogen (Median 1,4) und nur auf vier Werte gerastert.
- **Eine Welt, ein Gitter, ein Rauschmodell:** nur 5 × 5, Grad 6, n = 1000, Rauschstufe 0 und 0,5, ein lineares Modell, Adam mit fester Lernrate und 300 Schritten. Aussagen über andere Gitter, Wahrheiten, Trainer oder Lernraten sind nicht gemessen. Das Gitter und die Kostenformel sind aus dem Gedächtnis nachgebaut, nicht gegen den Primärtext geprüft.
- **Der Löser ist billig.** Zeit und Machbarkeit bei teuren Lösern (zum Beispiel ein gemischt-ganzzahliges Programm) sind nicht gemessen; die Zahl gelöster Instanzen ist eine Zählung, keine Zeitaussage. Die Zeit-Tabelle der App rechnet aus der live gemessenen Zeit je 1000 Instanzen die reine Löserzeit; sie enthält keine Gradientenrechnung.
- **Zahlen sind trainiert und plattformabhängig.** Gemessen unter Windows; unter Linux (CI und Streamlit Cloud) können einzelne Werte um Zehntelpunkte abweichen. Die Tests prüfen Bänder und Ungleichungen.
- **Median über 20 Zufalls-Seeds, kein Signifikanztest.** Die Spanne ist keine Konfidenzaussage. Unterschiede von 0,1 bis 0,3 Punkten liegen unter der Streuung zwischen Seeds (Spannen von 2,5 bis 4 Punkten bei Rauschstufe 0,5); sie stehen deshalb nur zusammen mit der Stabilitätsprobe (100 Seeds, gepaarte Differenzen).
- **Ursachen sind Deutungen:** warum fünf Störungen bei σ = 0,1 den Regret von 13,1 auf 12,2 % senken, warum σ = 3 am Anfang langsamer ist, ist hier nicht untersucht.
- **Die Glättungsverzerrung ist an einer Welt gemessen** (60 Merkmalsvektoren je Zufalls-Seed, gestörte Lösungen mit M = 1); ohne Rauschen folgt aus der Definition, dass der gestörte Teilgradient an der Wahrheit nicht null sein muss (die gemittelte gestörte Lösung weicht vom harten Weg ab, sobald eine Störung den Weg ändert); wie groß der Anteil ist, ist Messung.
- **Die Aussage gilt für das lineare Modell.** Gegen eine flexible Modellklasse ist nichts gemessen (siehe `spo-kuerzester-weg-demo` für die Vormessung mit Gauß-Kern-Ridge).

## Weitergabe: Module

Der Kern liegt in Modulen ohne Streamlit-Abhängigkeit (nur numpy). Sechs davon sind aus `spo-kuerzester-weg-demo` kopiert und von `skw_` nach `gso_` umbenannt (Herkunft im Modulkopf), nicht importiert:

| Modul | Inhalt |
|---|---|
| `gso_rng.py`, `gso_grid.py`, `gso_world.py`, `gso_losses.py`, `gso_training.py`, `gso_runs.py` | SplitMix64, Gitter und Löser, Kostenwelt, SPO-Verlust und SPO+, KQ und Adam, Lauf einer Welt (kopiert) |
| `gso_perturb.py` | Gauß-Störung, gemittelte gestörte Lösung, Teilgradient, geschlossene Form für zwei Wege, Training mit gestörten Lösungen |
| `gso_effort.py` | zählender Löser, Formeln der gelösten Instanzen, Zeitmessung (live) |
| `gso_fyrun.py` | Läufe: KQ, SPO+, gestörte Lösungen; Teilgradient an der wahren Vorhersage |

## Verifikation

Die Suite hat **172 Tests**.

- **Kern gegen Handrechnung:** Handbeispiel mit zwei Wegen (σ = 0,1 / 1 / 3: Gewicht und Teilgradient), σ → 0 (harter Löser) und σ → ∞ ((0,5; 0,5)); ein Adam-Schritt von Hand; Zählung gelöster Instanzen von Hand (n = 3, 10 Schritte, M = 5: 150 plus 3 vorab, SPO+: 30 plus 3); Löser auf 2 × 2, 3 × 3 und 5 × 5 (`tests/test_perturb.py`, `tests/test_effort.py`, `tests/test_grid.py`, `tests/test_losses.py`, `tests/test_training.py`).
- **Orakel** (`tests/test_oracle_gestoerte_optimierer.py`, unabhängige Rechenwege): geschlossene Form $\Phi((\hat c_2 - \hat c_1)/(\sigma\sqrt 2))$ gegen den Anteil gewählter Wege bei 200 000 Störungen aus `numpy.random` (anderer Zufallsstrom); gestörter Löser gegen die Aufzählung aller 70 Wege auf denselben Störungen; gemittelte gestörte Lösung auf dem Gitter (150 SplitMix64-Störungen) gegen die Aufzählung mit 6000 Störungen aus `numpy.random`; Teilgradient und Gradient durch das lineare Modell gegen den zentralen Differenzenquotienten des Surrogats `ĉᵀw*(c) − Mittel der Minima`; dazu aus `spo-kuerzester-weg-demo` kopiert: Löser gegen Aufzählung, SPO+ gegen das Maximum über alle Wege, SPO+-Teilgradient gegen den Differenzenquotienten, KQ gegen die Normalengleichungen, SplitMix64 gegen eine reine Ganzzahlrechnung (`tests/test_rng.py`).
- **Eigenschaften:** Gewichte summieren zu 1, die gemittelte Lösung hat die Bogensumme eines Weges, Gradient null bei y = gemittelter Lösung, Skalierung von σ und ĉ mit demselben Faktor ändert keine Wahl, σ = 0 liefert in jeder Ziehung die harte Lösung, Störströme reproduzierbar und standardnormal.
- **Randfälle:** σ = 0, gleiche Vorhersagen, M = 1, n = 1 und n = 2 Trainingsfälle, kleinstes Gitter, extreme σ, Ränder der Handbeispiel-Regler, Bias ohne Rauschen (`tests/test_edge_cases.py`).
- **Claims:** jede Zahl dieser README steht in `tests/test_claims.py` als wörtliche Tabellenzeile, Band oder Ungleichung; die Vormessungswerte werden nur als eingefroren geprüft. Nicht im Test: die Scratch-Läufe der Orakel-Tests mit mehr Instanzen (nicht im Repo).
- **Verdikte:** reine Funktionen, Schwellen und jede Zahl im Text (`tests/test_verdicts.py`); jedes Preset zeigt das erwartete Verdikt (`tests/test_app.py`).
- **Oberfläche:** AppTest (jeder Button, jeder Regler an den Grenzen, größte Last, Permalink mit Einrasten, Zeitknopf), Layout (Schnellstart und Welt in der Seitenleiste, Regler der Experimente im Hauptbereich, fester Kopf, aktives Preset hervorgehoben), Footer, Laufzeit-Importe, PDF.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Seitenleiste (Schnellstart, Welt), Ergebnis, Kernfrage, fünf Experimente, Grenzen, Formeln |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, plotly, pandas, fpdf2 |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `gso_rng.py`, `gso_grid.py`, `gso_world.py`, `gso_losses.py`, `gso_training.py`, `gso_runs.py` | aus `spo-kuerzester-weg-demo` kopierter Kern (siehe „Weitergabe“) |
| `gso_perturb.py`, `gso_effort.py`, `gso_fyrun.py` | neuer Kern: gestörte Lösungen, Löseraufwand, Läufe |
| `gso_bands.py`, `gso_bands.json` | lädt die vorgerechneten Bänder; eigene Messung (`tools/measure_bands.py`) |
| `gso_vormessung.json` | eingefrorene Zahlen der Vormessung (`tools/freeze_vormessung.py`) |
| `gso_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `gso_evaluation.py` | Handbeispiel, Zahlen der Experimente, Verdikte (reine Funktionen), Tabellen aus den Bändern |
| `gso_layout.py` | Fester Kopf (Stylesheet) |
| `gso_presets.py` | Permalink (Begrenzen und Einrasten), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `gso_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `gso_pdf_export.py` | PDF-Export der aktuellen Welt (fpdf2) |
| `tools/measure_bands.py`, `tools/freeze_vormessung.py` | erzeugen die beiden JSON-Dateien |
| `tests/` | Tests (siehe „Verifikation“); `tests/readme_tables.py` erzeugt die Tabellen dieser README |

## Bewusst nicht umgesetzt

- **Der Fenchel-Young-Verlust mit Verlustwert:** erst nach Primärlektüre von Berthet u. a. festzulegen (Referenzverteilung, Normierung); die Demo benutzt nur den Teilgradienten.
- **Nachbau neuerer Verluste** (Perturbation-Gradient, LCGLN, Dual-Guided Loss): nur Einordnung auf Abstract-Ebene.
- **Andere Störverteilungen, Lernen von σ, eine Wahlregel für σ:** nicht gemessen.
- **Löser aus einem gemischt-ganzzahligen Programm und Zeitaussagen für teure Löser:** keine Messung.
- **Training der Black-Box-Interpolation:** nur die Zahl der gelösten Instanzen steht in der Tabelle.
- **Flexible Modellklassen, Zufallswald, Minibatches, Lernratenwahl:** nicht Gegenstand.
- **Konsistenz- und Ratenbeweise:** nicht Gegenstand; die Demo zeigt den Mechanismus und Messungen.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly und NumPy.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsorientiert: Regret bis Wahl](https://sebastianhanisch.net/konzepte-entscheidungsorientiert.html).
