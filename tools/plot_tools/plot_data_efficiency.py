import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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


# ------ load data ------
root   = Path("/home/fzhcis/mylab/gdrive/papers/rse-2025-through-lidars-eye/data_efficiency")                               # <- adjust path
df = pd.read_csv(root / "data_efficiency_eval_metrics_summary.csv")
LINE_PLOT = True
HEATMAP = False
# print(df.head())
subset_nums = np.array([4, 8, 12, 16, 20, 24, 28])

feature_names = df["Feature_name"].unique().tolist()
print("Feature names:", feature_names)
accu_metrics = ["oAcc", "mAcc", "mIoU"]
entropy_metrics = ["error_map_entropy", "total_uncertainty_entropy", "mutual_info_entropy"]
auprc_metrics = ["total_uncertainty_auprc", "mutual_info_auprc"]


# # -----------Plot accuracy metrics heatmap and line plot-----------
# metrics = accu_metrics
# vmin = np.array(df[metrics]).min()
# vmax = np.array(df[metrics]).max()

# fig_key_str = metrics[0].split("_")[-1]
# for i, feature_name in enumerate(feature_names):
#     print(f"Plotting heatmap for feature set: {feature_name} ({i+1}/{len(feature_names)})")
#     # Filter the DataFrame for the current feature set
#     plot_df = df[df["Feature_name"] == feature_name][metrics].copy()
#     plot_df["subset_nums"] = subset_nums
#     if HEATMAP:
#         fig, ax = plt.subplots(figsize=(6, 3))
#         sns.heatmap(plot_df.set_index("subset_nums"),
#                     annot=True, fmt=".2f", cmap="YlOrRd",
#                     vmin=vmin, vmax=vmax,
#                     # cbar_kws={"label": "Score"},
#                     cbar=False,
#                     ax=ax)
#         # ax.set_title("Data Efficiency Metrics for Different Feature Sets")
#         ax.set_xlabel(None)
#         ax.set_ylabel(None)
#         ax.set_yticklabels(["Overall Accuracy", "Mean Accuracy", "Mean IoU"], rotation=0)
#         ax.set_xticklabels(subset_nums, rotation=0, ha="right")
#         plt.tight_layout()
#         fig.savefig(root / f"data_efficiency_heatmap_{i:02d}_{feature_name}.png", dpi=300)
#         plt.close(fig)

#     if LINE_PLOT:
#         fig, ax = plt.subplots(figsize=(5, 4))
#         for metric in metrics:
#             sns.lineplot(data=plot_df,
#                         x="subset_nums", 
#                         y=metric,
#                         marker="o", 
#                         label=metric, 
#                         ax=ax)

#         # ax.set_title(f"Performance vs. Training Set Size for {feature_name}")
#         ax.set_xlabel("Number of train/val scans")
#         ax.set_xticks(subset_nums)
#         ax.set_xticklabels(subset_nums, rotation=0, ha="right")
#         ax.set_ylabel(None)
#         ax.set_ylim(vmin*0.8, 1.0)
#         ax.legend()
#         ax.grid(True)
#         fig.savefig(root / f"data_efficiency_lineplot_{fig_key_str}_{i:02d}_{feature_name}.png", dpi=300)
#         plt.close(fig)


# -----------Plot entropy metrics heatmap and line plot-----------
# metrics = entropy_metrics
# fig_key_str = metrics[0].split("_")[-1]
# for i, feature_name in enumerate(feature_names):
#     print(f"Plotting heatmap for feature set: {feature_name} ({i+1}/{len(feature_names)})")
#     # Filter the DataFrame for the current feature set
#     plot_df = df[df["Feature_name"] == feature_name][metrics].copy()
#     plot_df["subset_nums"] = subset_nums
#     if HEATMAP:
#         vmin = np.array(df[metrics]).min()
#         vmax = np.array(df[metrics]).max()
#         fig, ax = plt.subplots(figsize=(6, 3))
#         sns.heatmap(plot_df.set_index("subset_nums"),
#                     annot=True, fmt=".2f", cmap="YlOrRd",
#                     vmin=vmin, vmax=vmax,
#                     # cbar_kws={"label": "Score"},
#                     cbar=False,
#                     ax=ax)
#         # ax.set_title("Data Efficiency Metrics for Different Feature Sets")
#         ax.set_xlabel(None)
#         ax.set_ylabel(None)
#         ax.set_yticklabels(["Overall Accuracy", "Mean Accuracy", "Mean IoU"], rotation=0)
#         ax.set_xticklabels(subset_nums, rotation=0, ha="right")
#         plt.tight_layout()
#         fig.savefig(root / f"data_efficiency_heatmap_{i:02d}_{feature_name}.png", dpi=300)
#         plt.close(fig)

#     if LINE_PLOT:
#         fig, ax1 = plt.subplots(figsize=(6, 4))
        
#         # Left y-axis: total_uncertainty_entropy and mutual_info_entropy
#         left_metrics = ["total_uncertainty_entropy", "mutual_info_entropy"]
#         markers = ["o", "s"]  # Different markers for left axis metrics
#         vmin = np.array(df[left_metrics]).min()
#         vmax = np.array(df[left_metrics]).max()
#         for j, metric in enumerate(left_metrics):
#             if metric in metrics:
#                 ax1.plot(plot_df["subset_nums"], plot_df[metric],
#                         marker=markers[j], 
#                         label=metric.replace('_', ' ').title(),
#                         color='tab:blue',
#                         linestyle='-',
#                         linewidth=2,
#                         markersize=6)
        
#         ax1.set_xlabel("Number of train/val scans")
#         ax1.set_xticks(subset_nums)
#         ax1.set_xticklabels(subset_nums, rotation=0, ha="right")
#         ax1.set_ylabel("Uncertainty Entropy", color='tab:blue')
#         ax1.tick_params(axis='y', labelcolor='tab:blue')
#         ax1.set_ylim(vmin*0.95, vmax*1.01)
#         # Right y-axis: error_map_entropy
#         ax2 = ax1.twinx()
#         if "error_map_entropy" in metrics:
#             ax2.plot(plot_df["subset_nums"], plot_df["error_map_entropy"],
#                     marker="^", 
#                     label="Error Map Entropy",
#                     color='tab:red',
#                     linestyle='--',
#                     linewidth=2,
#                     markersize=6)
        
#         ax2.set_ylabel("Error Map Entropy", color='tab:red')
#         ax2.tick_params(axis='y', labelcolor='tab:red')
#         vmin = np.array(df['error_map_entropy']).min()
#         vmax = np.array(df['error_map_entropy']).max()
#         ax2.set_ylim(vmin*0.95, vmax*1.01)
        
#         # Create combined legend
#         lines1, labels1 = ax1.get_legend_handles_labels()
#         lines2, labels2 = ax2.get_legend_handles_labels()
#         ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
#         # Turn off grid completely
#         # ax1.grid(False)
#         ax2.grid(False)
#         # ax1.set_axisbelow(False)
        
#         fig.savefig(root / f"data_efficiency_lineplot_{fig_key_str}_{i:02d}_{feature_name}.png", dpi=300)
#         plt.close(fig)


# -----------Plot AUPRC metrics heatmap and line plot-----------
metrics = auprc_metrics
vmin = np.array(df[metrics]).min()
vmax = np.array(df[metrics]).max()
fig_key_str = metrics[0].split("_")[-1]
for i, feature_name in enumerate(feature_names):
    print(f"Plotting heatmap for feature set: {feature_name} ({i+1}/{len(feature_names)})")
    # Filter the DataFrame for the current feature set
    plot_df = df[df["Feature_name"] == feature_name][metrics].copy()
    plot_df["subset_nums"] = subset_nums
    if HEATMAP:
        fig, ax = plt.subplots(figsize=(6, 3))
        sns.heatmap(plot_df.set_index("subset_nums"),
                    annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=vmin, vmax=vmax,
                    # cbar_kws={"label": "Score"},
                    cbar=False,
                    ax=ax)
        # ax.set_title("Data Efficiency Metrics for Different Feature Sets")
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_yticklabels(["Overall Accuracy", "Mean Accuracy", "Mean IoU"], rotation=0)
        ax.set_xticklabels(subset_nums, rotation=0, ha="right")
        plt.tight_layout()
        fig.savefig(root / f"data_efficiency_heatmap_{i:02d}_{feature_name}.png", dpi=300)
        plt.close(fig)

    if LINE_PLOT:
        fig, ax = plt.subplots(figsize=(6, 4))
        for metric in metrics:
            sns.lineplot(data=plot_df,
                        x="subset_nums", 
                        y=metric,
                        marker="o", 
                        label=metric, 
                        ax=ax)

        # ax.set_title(f"Performance vs. Training Set Size for {feature_name}")
        ax.set_xlabel("Number of train/val scans")
        ax.set_xticks(subset_nums)
        ax.set_xticklabels(subset_nums, rotation=0, ha="right")
        ax.set_ylabel("Area under PR Curve")
        ax.set_ylim(vmin*0.9, vmax*1.01)
        ax.legend()
        ax.grid(True)
        fig.savefig(root / f"data_efficiency_lineplot_{fig_key_str}_{i:02d}_{feature_name}.png", dpi=300)
        plt.close(fig)