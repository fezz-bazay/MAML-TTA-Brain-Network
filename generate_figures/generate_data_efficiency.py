import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

N_SEEDS = 5
def sem(std): return std / np.sqrt(N_SEEDS)

k_sgd   = [0,     1,     3,     5,     10,    15]
bnt_auc = [70.66, 68.22, 69.50, 70.38, 70.50, 72.65]
bnt_std = [0.86,  1.28,  2.95,  1.47,  1.47,  1.65]
bnt_sem = [sem(s) for s in bnt_std]

k_maml   = [0,     1,     3,     5,     10,    15]
maml_auc = [70.55, 72.26, 72.39, 72.31, 72.36, 73.24]
maml_std = [0.75,  1.47,  1.30,  1.31,  1.37,  0.53]
maml_sem = [sem(s) for s in maml_std]

fig, ax = plt.subplots(figsize=(8, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

ax.errorbar(k_sgd, bnt_auc, yerr=bnt_sem,
            marker="o", color="#4472C4", linewidth=2.0,
            markersize=7, capsize=4, capthick=1.5, elinewidth=1.5,
            linestyle="--", label="BNT-SGD fine-tuning", zorder=3)
ax.fill_between(k_sgd,
                [v-e for v,e in zip(bnt_auc, bnt_sem)],
                [v+e for v,e in zip(bnt_auc, bnt_sem)],
                color="#4472C4", alpha=0.10)

ax.errorbar(k_maml, maml_auc, yerr=maml_sem,
            marker="s", color="#C0392B", linewidth=2.0,
            markersize=7, capsize=4, capthick=1.5, elinewidth=1.5,
            linestyle="-", label="BNT-MAML-FewShot", zorder=4)
ax.fill_between(k_maml,
                [v-e for v,e in zip(maml_auc, maml_sem)],
                [v+e for v,e in zip(maml_auc, maml_sem)],
                color="#C0392B", alpha=0.10)

ax.axvspan(2.7, 3.3, alpha=0.06, color="#555555", zorder=0)

ax.set_xlabel("Support examples per site (k)", fontsize=20)
ax.set_ylabel("Inter-site AUC (%)", fontsize=20)
ax.set_xlim(0, 16)
ax.set_ylim(66, 76)
ax.set_xticks([0, 1, 3, 5, 10, 15])
ax.set_xticklabels(["k=0", "k=1", "k=3", "k=5", "k=10", "k=15"])
ax.tick_params(axis="both", labelsize=15)
ax.legend(fontsize=16, loc="lower right",
          frameon=True, framealpha=1.0, edgecolor="#CCCCCC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.spines["bottom"].set_color("#CCCCCC")

plt.tight_layout()
plt.savefig("figures/fig_data_efficiency_final.pdf", dpi=300, bbox_inches="tight")
plt.savefig("figures/fig_data_efficiency_final.png", dpi=300, bbox_inches="tight")
print("Figure sauvegardee!")