import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


sns.set_style("whitegrid")        # cleaner look
plt.rcParams.update({
    'font.size': 12,         # base font size
    'axes.titlesize': 12,    # title size
    'axes.labelsize': 10,    # x/y label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12,
    # 'figure.dpi': 120,
})

root_dir = Path("/home/fzhcis/mylab/gdrive/papers/rse-2025-through-lidars-eye/feature_ablation")
df = pd.read_csv(root_dir / "eval_feature_ablation.csv") 
metrics_fig_path = root_dir / "feature_ablation_metrics.png"


# --------------  2. Dotted-line plot for OA / MA / mIoU  --------------
metrics = ["Overall Accuracy", "Mean Accuracy", "Mean IoU"]
x_labels = df["Feature Name"]

fig1, ax = plt.subplots(figsize=(12, 4))
for m in metrics:
    ax.plot(x_labels,
            df[m],
            marker="o",
            linestyle="--",
            label=m)

# ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_xlabel("Feature configuration")
ax.set_xticks(range(len(x_labels)))
ax.set_xticklabels(x_labels, rotation=45, ha="right")
ax.set_title("Global segmentation metrics across feature sets")
ax.legend(frameon=False)
plt.tight_layout()
fig1.savefig(metrics_fig_path, dpi=300)
plt.close(fig1)
# plt.show()

# --------------  3. Heat-map for per-class IoU  --------------
class_cols = ["Void",
              "Ground & Water",
              "Stem",
              "Canopy",
              "Roots",
              "Objects"]

heat_data = (df[class_cols]
             .set_index(df["Feature Name"]))      # rows = feature sets

# Transpose the data so features are columns and classes are rows
heat_data = heat_data.T

heat_data_range = [heat_data.min().min(),
                  heat_data.max().max()]  # global min/max for color scale

cmap = 'YlOrRd'
iou_fig_path = root_dir / f"feature_ablation_per_class_iou_{cmap}.png"

# Adjust figure size based on data dimensions for better grid scale
n_features = heat_data.shape[1]  # number of feature configurations
n_classes = heat_data.shape[0]   # number of semantic classes
fig_width = max(6, min(12, n_features * 0.8))  # adaptive width, min 8, max 16
fig_height = max(4, n_classes * 0.5)           # adaptive height based on classes

fig2, ax2 = plt.subplots(figsize=(fig_width, fig_height))
sns.heatmap(heat_data,
            annot=False,
            fmt=".3f",
            cmap=cmap,
            vmin=heat_data_range[0], 
            vmax=heat_data_range[1],
            cbar_kws={"label": "IoU"},
            ax=ax2)

ax2.set_title("Per-class IoU by feature configuration")
ax2.set_xlabel(None)
ax2.set_ylabel("Semantic class")


# # Mute all x-tick labels
ax2.set_xticklabels([])
# plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")

# Keep Y-axis labels horizontal (rotation=0)
plt.setp(ax2.get_yticklabels(), rotation=0)


plt.tight_layout()
fig2.savefig(iou_fig_path, dpi=300)
plt.close(fig2)  # Free memory after saving
