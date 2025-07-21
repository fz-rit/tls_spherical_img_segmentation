import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
plt.rcParams.update({
    'font.size': 12,         # base font size
    'axes.titlesize': 12,    # title size
    'axes.labelsize': 10,    # x/y label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})
CLASS_MAP = {
    0: 'Void',
    1: 'Ground&Water',
    2: 'Stem',
    3: 'Canopy',
    4: 'Roots',
    5: 'Object'
}


def read_class_id_csv(file_path):
    """
    Read a CSV file and return a DataFrame with the class_id column.
    """
    df = pd.read_csv(file_path)
    if 'class_id' not in df.columns:
        raise ValueError(f"CSV file {file_path} does not contain 'class_id' column.")
    class_id_col = df[['class_id']].to_numpy()
    print(f"Shape of class_id column: {class_id_col.shape}")
    print(f"Type of class_id column: {class_id_col.dtype}")
    print(f"Unique values in class_id column: {np.unique(class_id_col)}")
    return class_id_col


def grab_class_id(file_paths):
    """
    Read all CSV files and concatenate the class_id columns into a single array.
    """
    class_id_ls = []
    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        if file_path.suffix == '.csv':
            class_id_col = read_class_id_csv(file_path)
        elif file_path.suffix == '.label':
            class_id_col = np.loadtxt(file_path, dtype=int)
        elif file_path.suffix == '.png':
            mask = Image.open(file_path)
            class_id_col = np.array(mask).flatten()
            class_id_col = class_id_col.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}. Only .csv, .label and .png files are supported.")
        class_0_num = (class_id_col == 0).sum()
        class_0_ratio = class_0_num / len(class_id_col) * 100
        if class_0_num > 10:
            print(f"File: {file_path}, Number of class 0 points: {class_0_num}, Ratio: {class_0_ratio:.2f}%")
        class_id_ls.append(class_id_col)
    class_id_all = np.concatenate(class_id_ls)
    return class_id_ls, class_id_all

def plot_class_id_hist(class_id_vec, class_map, save_path):
    """
    Plot histogram for all class_id values with a second y-axis showing class ratios.

    Parameters:
    - class_id_vec: np.ndarray of class ids (1D array)
    - class_map: dict mapping class id to class name (e.g., {0: 'Ground', 1: 'Tree'})
    - save_path: path to save the figure
    """
    FONTSIZE = 16

    class_ids = np.unique(class_id_vec)
    counts = np.array([(class_id_vec == cid).sum() for cid in class_ids])
    ratios = counts / counts.sum()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Histogram (left y-axis)
    bars = ax1.bar(class_ids, counts, width=0.6, edgecolor='black', align='center', color='skyblue')
    ax1.set_ylabel('Number of Points', color='blue', fontsize=FONTSIZE)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONTSIZE)
    ax1.set_xticks(class_ids)
    ax1.set_xticklabels([class_map[i] for i in class_ids], rotation=0, fontsize=FONTSIZE)
    # ax1.set_title(f'Histogram of Classes ({prefix})\n ', fontsize=FONTSIZE)
    ax1.grid(axis='y', alpha=0.3)

    # Add counts to top of each bar
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:,}',
                     xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 5),
                     textcoords="offset points",
                     ha='center', va='bottom', fontsize=FONTSIZE)

    # Ratio line (right y-axis)
    ax2 = ax1.twinx()
    ax2.plot(class_ids, ratios, 'o', color='darkred', label='Ratio')
    ax2.set_ylabel('Ratio (%)', color='darkred', fontsize=FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=FONTSIZE)
    ax2.set_ylim(0, max(ratios) * 1.2)
    ax2.set_yticks(np.linspace(0, max(ratios) * 1.2, 5))
    ax2.set_yticklabels([f"{r*100:.1f}%" for r in np.linspace(0, max(ratios) * 1.2, 5)], fontsize=FONTSIZE)

    # Add ratio text next to each dot
    for x, y in zip(class_ids, ratios):
        ax2.annotate(f'{y*100:.1f}%',
                     xy=(x, y),
                     xytext=(5, 0),
                     textcoords='offset points',
                     ha='left', va='center', fontsize=FONTSIZE, color='darkred')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ----- For pcd files -----
# root_dir = Path("/home/fzhcis/data/mangrove3d_pcd/test/")
# label_paths = list(root_dir.glob('**/*_refined.label'))


# ----- For image mask files -----
root_dir = Path("/home/fzhcis/data/palau_2024_for_rc/mangrove3d/train_val/")
label_paths = list(root_dir.glob('**/*_segmk_refined.png'))
if len(label_paths) != 9:
    raise ValueError(f"Expected 9 label files, found {len(label_paths)}. Please check the directory structure.")
class_id_ls, class_id_all = grab_class_id(label_paths)
save_path = root_dir / "class_id_histogram.png"
plot_class_id_hist(class_id_all, CLASS_MAP, save_path=save_path)

# print(f"Label file paths: {label_paths}")