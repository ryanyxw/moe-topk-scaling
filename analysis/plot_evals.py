"""Downstream accuracy vs validation loss (SVG with CSS-variable colors, same style as plot_isoflop.py)."""
import json, io, os, re, numpy as np, matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
FG, MU, LN = "#010101", "#020202", "#030303"
plt.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif", "font.size": 10.5,
 "text.color": FG, "axes.labelcolor": FG, "xtick.color": MU, "ytick.color": MU, "axes.edgecolor": LN,
 "axes.spines.top": False, "axes.spines.right": False, "grid.color": LN, "grid.linewidth": 0.6,
 "figure.facecolor": "none", "axes.facecolor": "none", "legend.frameon": False})
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
R = json.load(open(f"{RES}/evals/results.json"))
VAL = {"c1_k1": 2.7574, "c1_k2": 2.8271, "c1_k4": 2.7428, "c1_k8": 2.7421, "c1_k16": 2.8100, "c2_k4": 2.5726, "c2_k8": 2.5707,
       "c2_k16": 2.6016, "c2_k32": 2.6903, "c3_k4": 2.4602, "c3_k8": 2.4385, "c3_k16": 2.4565, "c3_k32": 2.5197, "c4_k4": 2.3898,
       "c4_k8": 2.3619, "c4_k16": 2.3750, "final_k12": 2.2405, "final_k12_26k": 2.458, "final_k12_52k": 2.414,
       "final_k12_78k": 2.394, "final_k12_104k": 2.375}
A = ["HellaSwag", "ARC-C", "ARC-E", "PIQA", "Winogrande 5s", "OBQA", "SciQ", "LAMBADA"]
col = {"c1": "#3f88c5", "c2": "#e09f3e", "c3": "#d1495b", "c4": "#2a9d6f"}
lab = {"c1": "C1 8e18", "c2": "C2 2e19", "c3": "C3 5e19", "c4": "C4 1e20"}
def svg(fig):
    b = io.StringIO(); fig.savefig(b, format="svg", bbox_inches="tight"); s = b.getvalue()[b.getvalue().index("<svg"):]
    s = re.sub(r'<svg([^>]*?) width="[^"]*" height="[^"]*"', r'<svg\1 style="width:100%;height:auto"', s, count=1)
    for c, v in ((FG, "var(--fg)"), (MU, "var(--muted)"), (LN, "var(--line)")): s = s.replace(c, v)
    return s
fig, axs = plt.subplots(1, 2, figsize=(7.8, 3.6))
for ax, key, title in ((axs[0], "avg", "Average of 8 tasks (%)"), (axs[1], "HellaSwag", "HellaSwag 10-shot (%)")):
    y = lambda m: 100 * np.mean([R[m][c][0] for c in A]) if key == "avg" else 100 * R[m][key][0]
    cooled = [m for m in R if "_k12_" not in m]
    x = np.array([VAL[m] for m in cooled]); yy = np.array([y(m) for m in cooled])
    p = np.polyfit(x, yy, 1); xs = np.linspace(2.2, 2.85, 2)
    ax.plot(xs, np.polyval(p, xs), "-", color=MU, lw=1, alpha=.6, label=f"linear fit, r = {np.corrcoef(x, yy)[0, 1]:.3f}")
    for b in col:
        ms = [m for m in R if m.startswith(b + "_")]
        ax.plot([VAL[m] for m in ms], [y(m) for m in ms], "o", color=col[b], ms=6, label=lab[b])
    pre = ["final_k12_26k", "final_k12_52k", "final_k12_78k", "final_k12_104k"]
    ax.plot([VAL[m] for m in pre], [y(m) for m in pre], "^", mfc="none", mec="#7b5cc4", ms=7, mew=1.3, label="final run, pre-cooldown")
    ax.plot(VAL["final_k12"], y("final_k12"), "*", color="#7b5cc4", ms=15, mec=FG, mew=.5, label="final run (4.5e20)")
    ax.invert_xaxis(); ax.set_xlabel("Validation loss (lower is better →)"); ax.set_title(title, fontsize=10.5, loc="left"); ax.grid(True)
axs[0].legend(fontsize=8, loc="upper left")
fig.tight_layout(); open(f"{RES}/figures/evals.svg", "w").write(svg(fig))
