import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

N_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]

bnt_auc  = [69.60, 70.01, 70.16, 70.66, 69.17, 69.17, 71.65, 72.59, 70.39, 72.69, 75.92]
bnt_std  = [0.66,  1.18,  1.70,  0.86,  0.53,  0.91,  1.37,  1.25,  1.61,  2.08,  1.27]

maml_auc = [71.59, 71.22, 71.60, 71.88, 70.47, 70.54, 73.01, 74.78, 72.56, 75.04, 79.42]
maml_std = [0.19,  1.29,  1.68,  0.73,  0.44,  0.98,  1.06,  1.75,  1.52,  2.31,  2.25]

def sem(std): return std / np.sqrt(5)

bnt_sem  = [sem(s) for s in bnt_std]
maml_sem = [sem(s) for s in maml_std]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# BNT source-only
ax.errorbar(N_values, bnt_auc, yerr=bnt_sem,
            marker="o", color="#4472C4", linewidth=2.0,
            markersize=6, capsize=3, capthick=1.2, elinewidth=1.0,
            linestyle="--", label="BNT source-only", zorder=3)
ax.fill_between(N_values,
                [v-e for v,e in zip(bnt_auc, bnt_sem)],
                [v+e for v,e in zip(bnt_auc, bnt_sem)],
                color="#4472C4", alpha=0.10)

# MAML+TTA
ax.errorbar(N_values, maml_auc, yerr=maml_sem,
            marker="s", color="#C0392B", linewidth=2.0,
            markersize=6, capsize=3, capthick=1.2, elinewidth=1.0,
            linestyle="-", label="BNT-MAML-TTA (ours, k=3)", zorder=4)
ax.fill_between(N_values,
                [v-e for v,e in zip(maml_auc, maml_sem)],
                [v+e for v,e in zip(maml_auc, maml_sem)],
                color="#C0392B", alpha=0.10)

# Annotation N=4 (notre setting principal)
ax.axvspan(3.7, 4.3, alpha=0.06, color="#555555", zorder=0)
ax.annotate("Our setting\n(N=4)",
            xy=(4, 71.5), xytext=(4, 74.5),
            fontsize=20, color="#888888",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", color="#AAAAAA", lw=1.0))

ax.set_xlabel("Number of training sites (N)", fontsize=20)
ax.set_ylabel("Inter-site AUC (%)", fontsize=20)
ax.set_xlim(1, 17)
ax.set_ylim(65, 83)
ax.set_xticks(N_values)
ax.tick_params(axis="both", labelsize=15)
ax.legend(fontsize=16, loc="lower right",
          frameon=True, framealpha=1.0, edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")



plt.tight_layout()
plt.savefig("figures/fig_scaling_sites.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_scaling_sites.png", dpi=300, bbox_inches="tight")
print("Figure sauvegardee!")
