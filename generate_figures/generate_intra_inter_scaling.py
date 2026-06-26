import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

N_SEEDS = 5
def sem(std): return std / np.sqrt(N_SEEDS)

N_values = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]

intra_auc = [84.67, 84.91, 84.85, 89.43, 88.78, 87.46, 89.97, 87.41, 90.40, 87.67, 84.95]
intra_std = [5.05,  4.99,  5.57,  5.57,  3.72,  1.70,  1.97,  3.06,  1.20,  2.29,  3.46]
intra_sem = [sem(s) for s in intra_std]

inter_auc = [71.59, 71.22, 71.61, 71.88, 70.47, 70.54, 73.01, 74.78, 72.56, 75.04, 79.42]
inter_std = [0.19,  1.29,  1.68,  0.73,  0.44,  0.97,  1.06,  1.75,  1.52,  2.30,  2.25]
inter_sem = [sem(s) for s in inter_std]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# Intra-site
ax.errorbar(N_values, intra_auc, yerr=intra_sem,
            marker="o", color="#C0392B", linewidth=2.0,
            markersize=7, capsize=4, capthick=1.5, elinewidth=1.5,
            linestyle="-", label="Intra-site AUC (source sites)", zorder=4)
ax.fill_between(N_values,
                [v-e for v,e in zip(intra_auc, intra_sem)],
                [v+e for v,e in zip(intra_auc, intra_sem)],
                color="#C0392B", alpha=0.10)

# Inter-site
ax.errorbar(N_values, inter_auc, yerr=inter_sem,
            marker="s", color="#4472C4", linewidth=2.0,
            markersize=7, capsize=4, capthick=1.5, elinewidth=1.5,
            linestyle="--", label="Inter-site AUC (unseen sites)", zorder=3)
ax.fill_between(N_values,
                [v-e for v,e in zip(inter_auc, inter_sem)],
                [v+e for v,e in zip(inter_auc, inter_sem)],
                color="#4472C4", alpha=0.10)

# Zone our setting N=4
ax.axvspan(3.7, 4.3, alpha=0.06, color="#555555", zorder=0)
ax.annotate("Our setting (N=4)",
            xy=(4, 89.43), xytext=(6, 92),
            fontsize=16, color="#555555",
            ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->",
                           color="#AAAAAA", lw=1.0))

ax.set_xlabel("Number of training sites (N)", fontsize=20)
ax.set_ylabel("AUC (%)", fontsize=20)
ax.set_xlim(1, 17)
ax.set_ylim(65, 97)
ax.set_xticks(N_values)
ax.tick_params(axis="both", labelsize=15)

ax.legend(fontsize=16, loc="lower right",
          frameon=True, framealpha=1.0,
          edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")

plt.tight_layout()
plt.savefig("figures/fig_intra_inter_scaling.pdf",
            dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_intra_inter_scaling.png",
            dpi=300, bbox_inches="tight")
print("Figure sauvegardee!")
