import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

N_SEEDS = 5
def sem(std): return std / np.sqrt(N_SEEDS)

steps    = [0,     1,     2,     3,     5,     7,     10]
aucs     = [70.55, 72.26, 71.74, 72.39, 72.22, 72.20, 70.97]
stds     = [0.75,  1.26,  1.24,  1.30,  1.45,  1.27,  2.24]
sems     = [sem(s) for s in stds]

fig, ax = plt.subplots(figsize=(7, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.errorbar(steps, aucs, yerr=sems,
            marker="s", color="#C0392B", linewidth=2.0,
            markersize=7, capsize=4, capthick=1.5, elinewidth=1.5,
            linestyle="-", label="BNT-MAML-FewShot", zorder=4)
ax.fill_between(steps,
                [v-e for v,e in zip(aucs, sems)],
                [v+e for v,e in zip(aucs, sems)],
                color="#C0392B", alpha=0.12)

ax.axvspan(2.7, 3.3, alpha=0.06, color="#555555", zorder=0)

ax.set_xlabel("Number of inner loop steps", fontsize=20)
ax.set_ylabel("Inter-site AUC (%)", fontsize=20)
ax.set_xlim(0, 11)
ax.set_ylim(69.5, 75.5)
ax.set_xticks([0, 1, 2, 3, 5, 7, 10])
ax.tick_params(axis="both", labelsize=15)
ax.set_xticklabels(["steps={}".format(s) for s in steps],
                   rotation=30, ha="right")
ax.legend(fontsize=16, loc="lower left",
          frameon=True, framealpha=1.0, edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")

plt.tight_layout()
plt.savefig("figures/fig_auc_steps.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_auc_steps.png", dpi=300, bbox_inches="tight")
print("Figure sauvegardee!")