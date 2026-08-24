"""Draw the two designed thesis diagrams (system architecture, mechanism signal flow).

Both are drawn by hand from the model as src/microgrid_sim defines it (the hourly step
order in environment/model.py, the charge/allocation/surcharge path in
environment/capacity.py, the deferral path in agents/consumer.py) and exported as SVG
(vector, embedded in the .docx) with a PNG fallback at 300 dpi.

    python scripts/make_thesis_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "results" / "plots"

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "svg.fonttype": "none",  # keep text as text in the SVG
})

INK = "#222222"
EDGE = "#555555"
FILL_INPUT = "#eef2f7"
FILL_MODEL = "#ffffff"
FILL_MECH = "#fbf0e6"
FILL_OUT = "#eef7ee"
FILL_SWITCH = "#fff7d6"


def box(ax, x, y, w, h, text, fill=FILL_MODEL, fs=9, bold=False, lw=0.9, ls="-", align="center"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.18",
                       linewidth=lw, edgecolor=EDGE, facecolor=fill, linestyle=ls)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=INK, linespacing=1.25)
    return (x, y, w, h)


def arrow(ax, p0, p1, text=None, fs=8, style="-|>", lw=0.9, color=INK, connection="arc3,rad=0.0",
          text_offset=(0, 0.12), ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=9, linewidth=lw, color=color,
                        connectionstyle=connection, linestyle=ls, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if text:
        mx, my = (p0[0] + p1[0]) / 2 + text_offset[0], (p0[1] + p1[1]) / 2 + text_offset[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs, color=INK,
                bbox=dict(boxstyle="square,pad=0.12", facecolor="white", edgecolor="none"))


def right(b):
    return (b[0] + b[2], b[1] + b[3] / 2)


def left(b):
    return (b[0], b[1] + b[3] / 2)


def top(b):
    return (b[0] + b[2] / 2, b[1] + b[3])


def bottom(b):
    return (b[0] + b[2] / 2, b[1])


def architecture():
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 10.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- inputs (left) ---
    ax.text(1.9, 10.45, "Είσοδοι", ha="center", fontsize=10, fontweight="bold", color=INK)
    cfg = box(ax, 0.3, 8.3, 3.2, 1.5,
              "Ρυθμίσεις\nconfig/default.yaml\n+ σενάρια (extends)\nσπόρος, k, ablation, broker_count", FILL_INPUT, fs=8)
    sol = box(ax, 0.3, 6.2, 3.2, 1.4, "Ηλιακή παραγωγή\nPVGIS seriescalc, Θεσσαλονίκη\nέτος 2020, ωριαία (1 kWp)", FILL_INPUT, fs=8)
    dem = box(ax, 0.3, 4.1, 3.2, 1.4, "Σειρά ζήτησης\nσυνθετική (προεπιλογή)\nή OPSD (μετρημένη, opt-in)", FILL_INPUT, fs=8)
    tests = box(ax, 0.3, 1.3, 3.2, 1.5, "Έλεγχοι και golden master\ntests/ (pytest, 461 + 3 skipped)\nμεταλλάξεις, ντετερμινισμός", FILL_INPUT, fs=8, ls="--")

    # --- model (centre) ---
    mx, my, mw, mh = 4.2, 1.0, 6.6, 9.0
    ax.add_patch(FancyBboxPatch((mx, my), mw, mh, boxstyle="round,pad=0.0,rounding_size=0.25",
                                linewidth=1.2, edgecolor=INK, facecolor="#fafafa"))
    ax.text(mx + mw / 2, my + mh - 0.32, "MicrogridModel (Mesa 3.5.1), ωριαίος βρόχος step()",
            ha="center", va="center", fontsize=10, fontweight="bold", color=INK)

    bx, bw = 4.95, 5.65
    brokers = box(ax, bx, 7.55, bw, 1.6,
                  "1. Πάροχοι (brokers), σταθεροί κανόνες\nρυθμιζόμενος (επίπεδος, εποχικός) | ασταθής χαμηλού\nκόστους (AR(1), Student-t) | premium πράσινος (επίπεδος)\nπροσφορά = βασική τιμή + επιβάρυνση προηγούμενης ώρας",
                  FILL_MODEL, fs=7.5)
    agents = box(ax, bx, 5.3, bw, 1.95,
                 "2. Πληθυσμός 200 πρακτόρων (D1 έως D4)\n160 καταναλωτές, 40 παραγωγοί-καταναλωτές (PV + μπαταρία)\nπροφίλ: ευαίσθητοι στην τιμή / στη σταθερότητα / πράσινοι\nεπιλογή παρόχου με αδράνεια, λεξικογραφικοί κανόνες\nαναβολή ζήτησης (D7), αντιδραστική αποστολή μπαταρίας (D3)",
                 FILL_MODEL, fs=7.5)
    feeder = box(ax, bx, 3.85, bw, 1.1,
                 "3. Συνάθροιση τροφοδότη\nκαθαρή εισαγωγή = Σ (ζήτηση - PV - εκφόρτιση),\nσυνολικά και ανά πάροχο",
                 FILL_MODEL, fs=7.5)
    mech = box(ax, bx, 1.3, bw, 2.2,
               "4-6. Μηχανισμός χωρητικότητας (D6/D7, προαιρετικός)\nκατώφλι μ + k·σ (παράθυρο 168 h), υπέρβαση,\nχρέωση 0.15 EUR/kWh υπέρβασης, επιμερισμός κατά μερίδιο\nκανάλι P&L: χρέωση λογιστικού βιβλίου παρόχου\nκανάλι τιμολόγησης: επιβάρυνση = passthrough · μερίδιο (επόμενη ώρα)",
               FILL_MECH, fs=7.3)

    arrow(ax, bottom(brokers), top(agents), "τιμές + επιβάρυνση", fs=7.3, text_offset=(1.5, 0))
    arrow(ax, bottom(agents), top(feeder), "καθαρή εισαγωγή ανά πράκτορα", fs=7.3, text_offset=(1.8, 0))
    arrow(ax, bottom(feeder), top(mech), "I_t και συνεισφορές ανά πάροχο", fs=7.3, text_offset=(1.8, 0))
    # feedback loop routed along the left channel inside the model frame
    cx = 4.55
    arrow(ax, (bx, 2.4), (cx, 2.4), style="-", lw=0.9)
    arrow(ax, (cx, 2.4), (cx, 8.35), style="-", lw=0.9)
    arrow(ax, (cx, 8.35), (bx, 8.35), style="-|>", lw=0.9)
    ax.text(cx - 0.02, 5.4, "επιβάρυνση u_b, επόμενη ώρα (μόνο κανάλι τιμολόγησης)", rotation=90,
            ha="center", va="center", fontsize=7, color=INK,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="#fafafa", edgecolor="none"))

    # inputs -> model
    arrow(ax, right(cfg), (mx, 9.0), "", fs=7.5)
    arrow(ax, right(sol), (mx, 6.7), "", fs=7.5)
    arrow(ax, right(dem), (mx, 5.8), "", fs=7.5)
    arrow(ax, right(tests), (mx, 2.0), "", fs=7.5, ls="--")

    # --- outputs (right) ---
    ax.text(12.9, 10.45, "Έξοδοι", ha="center", fontsize=10, fontweight="bold", color=INK)
    metrics = box(ax, 11.1, 7.9, 3.6, 1.9,
                  "Μετρικές ανά εκτέλεση\n1 μέσο κόστος ανά πράκτορα\n2 κατανομή φορτίου (HHI)\n3 σταθερότητα τροφοδότη (CoV, PAR)\n4 αυτάρκεια παραγωγών-καταναλωτών",
                  FILL_OUT, fs=7.8)
    audit = box(ax, 11.1, 5.9, 3.6, 1.5,
                "Στήλες ελέγχου\nποσοστό ενεργοποίησης, συνολική\nχρέωση, μέση επιβάρυνση ανά πάροχο,\nαναβληθείσα ενέργεια, αλλαγές παρόχου",
                FILL_OUT, fs=7.8)
    sweeps = box(ax, 11.1, 3.6, 3.6, 1.8,
                 "Σαρώσεις\nscripts/run_*.py (pool 16 διεργασιών,\nεπανεκκινήσιμες)\nresults/*.parquet, 4630 εκτελέσεις",
                 FILL_OUT, fs=7.8)
    analysis = box(ax, 11.1, 1.3, 3.6, 1.8,
                   "Ανάλυση\nscripts/analyze_*.py: ζευγαρωτές\nσυγκρίσεις, Holm/BH, bootstrap\nresults/*.csv, results/plots/*.png",
                   FILL_OUT, fs=7.8)
    arrow(ax, (mx + mw, 8.85), left(metrics), "")
    arrow(ax, (mx + mw, 6.65), left(audit), "")
    arrow(ax, bottom(metrics), top(audit), "", lw=0.6)
    arrow(ax, bottom(audit), top(sweeps), "μία γραμμή ανά εκτέλεση", fs=7.5)
    arrow(ax, bottom(sweeps), top(analysis), "")

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    fig.savefig(OUT / "thesis_fig_architecture.svg")
    fig.savefig(OUT / "thesis_fig_architecture.png", dpi=300)
    plt.close(fig)


def signal_flow():
    fig, ax = plt.subplots(figsize=(7.5, 7.4))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 14.8)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

    # row 1: feeder -> window -> threshold -> excess -> charge
    y1, h1, w1, g = 12.9, 1.55, 2.65, 0.34
    xs = [0.3 + i * (w1 + g) for i in range(5)]
    feeder = box(ax, xs[0], y1, w1, h1, "Καθαρή εισαγωγή\nτροφοδότη I_t\n(μετά την αναβολή)", FILL_MODEL, fs=8)
    window = box(ax, xs[1], y1, w1, h1, "Κυλιόμενο παράθυρο\nW = 168 ώρες\nμ_t, σ_t", FILL_MECH, fs=8)
    thr = box(ax, xs[2], y1, w1, h1, "Κατώφλι\nθ_t = μ_t + k·σ_t\nk = 0.5 έως 2.0", FILL_MECH, fs=8)
    exc = box(ax, xs[3], y1, w1, h1, "Υπέρβαση\ne_t = max(0, I_t - θ_t)\n0 κάτω από το κατώφλι", FILL_MECH, fs=8)
    charge = box(ax, xs[4], y1, w1, h1, "Συνολική χρέωση\nC_t = c · e_t\nc = 0.15 EUR/kWh", FILL_MECH, fs=8)
    for p0, p1 in ((feeder, window), (window, thr), (thr, exc), (exc, charge)):
        arrow(ax, right(p0), left(p1))

    # row 2: contributions and allocation
    contrib = box(ax, 0.3, 10.2, 6.6, 1.35,
                  "Συνεισφορές ανά πάροχο b: καθαρή εισαγωγή των πελατών\nπου εξυπηρέτησε ο πάροχος b στο ίδιο βήμα",
                  FILL_MODEL, fs=7.6)
    alloc = box(ax, 8.0, 10.2, 6.7, 1.35,
                "Επιμερισμός ανά πάροχο\nμερίδιο s_b = max(0, συνεισφορά_b) / S_t,   A_b = C_t · s_b",
                FILL_MECH, fs=8)
    arrow(ax, bottom(feeder), top(contrib))
    arrow(ax, right(contrib), left(alloc), "μερίδια s_b", fs=7.5)
    arrow(ax, bottom(charge), (13.35, 11.55), "C_t", fs=7.5, text_offset=(0.35, 0))

    # row 3: channels and switches
    pnl = box(ax, 0.3, 7.35, 4.6, 1.9,
              "Κανάλι P&L (feedback_pnl)\nχρέωση A_b στο λογιστικό\nβιβλίο του παρόχου b\nκαμία εγγραφή σε τιμή,\nκαμία φυσική επίδραση",
              FILL_MODEL, fs=7.8)
    pricing = box(ax, 5.3, 7.35, 5.0, 1.9,
                  "Κανάλι τιμολόγησης (feedback_pricing)\nεπιβάρυνση u_b = passthrough · s_b\npassthrough = 0.10: συντελεστής\nέντασης σήματος, όχι τιμολόγιο",
                  FILL_MECH, fs=7.8)
    sw = box(ax, 10.7, 7.35, 4.0, 1.9,
             "Διακόπτες ελέγχου (control arms)\nproportional: passthrough · s_b\nsynchronized: passthrough / N\nrenormalized: passthrough · s_b · N\nδιαιρέτης M: N αντικαθίσταται από M",
             FILL_SWITCH, fs=7.3)
    arrow(ax, (9.4, 10.2), (2.6, 9.25), "A_b", fs=7.5, text_offset=(-0.35, 0.1))
    arrow(ax, (11.4, 10.2), (7.8, 9.25), "s_b", fs=7.5, text_offset=(0.35, 0.1))
    arrow(ax, left(sw), right(pricing), style="-|>", lw=0.7, ls="--")

    # row 4: agent response
    agent = box(ax, 0.3, 4.35, 7.2, 2.0,
                "Πράκτορας (καταναλωτής ή παραγωγός-καταναλωτής), επόμενη ώρα\nδιαβάζει την επιβάρυνση u_b του δικού του παρόχου\nσυντελεστής = clip(u_b / r, 0, 1),  r = 0.05\nαναβολή = 0.2 · βασική ζήτηση · συντελεστής   (μόνο όταν u_b > 0)",
                FILL_MODEL, fs=7.8)
    bucket = box(ax, 7.9, 4.35, 3.1, 2.0,
                 "Κάδος αναβληθείσας\nενέργειας ανά πράκτορα\nμεταφέρεται πέρα από\nτα μεσάνυχτα, χωρίς\nβίαιη εκκένωση",
                 FILL_MECH, fs=7.6)
    payback = box(ax, 11.4, 4.35, 3.3, 2.0,
                  "Αποπληρωμή σε ώρες\nχωρίς επιβάρυνση (u_b = 0)\nmin(κάδος, 0.5 · βασική\nζήτηση) ανά ώρα",
                  FILL_MODEL, fs=7.6)
    arrow(ax, bottom(pricing), (3.9, 6.35), "επιβάρυνση στην προσφορά της\nεπόμενης ώρας (καθυστέρηση 1 ώρας)", fs=7.3,
          text_offset=(2.25, 0.0))
    arrow(ax, right(agent), left(bucket), "αναβολή", fs=7.5, text_offset=(0, 0.2))
    arrow(ax, right(bucket), left(payback), "εκροή", fs=7.5, text_offset=(0, 0.2))
    # return path: payback -> (bottom margin) -> left margin -> feeder input, next hour
    ry = 3.55
    arrow(ax, bottom(payback), (13.05, ry), style="-")
    arrow(ax, (13.05, ry), (0.12, ry), style="-")
    arrow(ax, (0.12, ry), (0.12, 13.675), style="-")
    arrow(ax, (0.12, 13.675), left(feeder), style="-|>")
    ax.text(7.0, ry - 0.22, "επόμενη ώρα: η αναβολή μειώνει την καθαρή εισαγωγή του πράκτορα στην αιχμή, η αποπληρωμή την αυξάνει εκτός αιχμής (μετατόπιση, όχι διαγραφή ενέργειας)",
            ha="center", va="top", fontsize=7.3, color=INK, style="italic")
    ax.text(0.12 - 0.02, 8.6, "επιστροφή στον τροφοδότη", rotation=90, ha="center", va="center", fontsize=7, color=INK,
            bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))

    # note
    ax.add_patch(FancyBboxPatch((0.3, 0.7), 14.4, 1.9, boxstyle="round,pad=0.0,rounding_size=0.18",
                                linewidth=0.8, edgecolor=EDGE, facecolor="white", linestyle="--"))
    ax.text(7.5, 1.65,
            "Απομόνωση καναλιών: το κανάλι P&L γράφει μόνο σε λογιστικά βιβλία παρόχων, το κανάλι τιμολόγησης μόνο στην προσφορά της επόμενης ώρας.\n"
            "Με τον μηχανισμό απενεργοποιημένο ή με feedback_pricing = false, κάθε u_b είναι ακριβώς 0.0, ο κλάδος αναβολής είναι αδρανής\n"
            "και η βάση αναπαράγεται byte προς byte. Ο μηχανισμός δεν καταναλώνει τυχαίους αριθμούς, άρα οι τέσσερις διαμορφώσεις\n"
            "μοιράζονται την ίδια ακολουθία τυχαιότητας για δεδομένο σπόρο.",
            ha="center", va="center", fontsize=7.4, color=INK, linespacing=1.3)

    fig.savefig(OUT / "thesis_fig_signal_flow.svg")
    fig.savefig(OUT / "thesis_fig_signal_flow.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    signal_flow()
    print("wrote", OUT / "thesis_fig_architecture.{svg,png}", OUT / "thesis_fig_signal_flow.{svg,png}")
