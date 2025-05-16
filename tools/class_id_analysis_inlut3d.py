import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pathlib import Path




# def read_class_id_csv(file_path):
#     """
#     Read a CSV file and return a DataFrame with the class_id column.
#     """
#     df = pd.read_csv(file_path)
#     if 'class_id' not in df.columns:
#         raise ValueError(f"CSV file {file_path} does not contain 'class_id' column.")
#     class_id_col = df[['class_id']].to_numpy()
#     print(f"Shape of class_id column: {class_id_col.shape}")
#     print(f"Type of class_id column: {class_id_col.dtype}")
#     print(f"Unique values in class_id column: {np.unique(class_id_col)}")
#     return class_id_col

def read_class_id_from_pts(file_path):
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # First line is the number of points, skip it
    data_lines = lines[1:]

    # Convert to DataFrame
    from io import StringIO
    data_str = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep='\s+', header=None)
    df.columns = ["X", "Y", "Z", "r", "g", "b", "class_id", "instance_id"]
    class_id_col = df[['class_id']].to_numpy()
    print(f"Shape of class_id column: {class_id_col.shape}")
    print(f"Type of class_id column: {class_id_col.dtype}")
    print(f"Unique values in class_id column: {np.unique(class_id_col)}")
    return class_id_col


def read_class_id_from_mask_png(file_path):
    """
    Read a PNG file and return a DataFrame with the class_id column.
    """
    from PIL import Image
    img = Image.open(file_path)
    img = img.convert('L')  # Convert to grayscale
    class_id_col = np.array(img).flatten()
    print(f"Shape of class_id column: {class_id_col.shape}")
    print(f"Type of class_id column: {class_id_col.dtype}")
    print(f"Unique values in class_id column: {np.unique(class_id_col)}")
    return class_id_col

def grab_class_id(file_paths, labels):
    """
    Read all CSV files and concatenate the class_id columns into a single array.
    """
    count_dict = {label: 0 for label in labels}
    for file_path in file_paths:
        print(f"Reading file: {file_path}")
        class_id_col = read_class_id_from_pts(file_path)
        class_ids, counts = np.unique(class_id_col, return_counts=True)
        for class_id, count in zip(class_ids, counts):
            if class_id == -1:
                count_dict[18] += count
            else:
                count_dict[class_id] += count
    count_dict = {k: int(v) for k, v in count_dict.items()}
    return count_dict

def plot_class_id_hist(count_dict, class_map, save_path="class_id_histogram.png"):
    """
    Plot histogram for all class_id values with a second y-axis showing class ratios.

    Parameters:
    - class_id_vec: np.ndarray of class ids (1D array)
    - class_map: dict mapping class id to class name (e.g., {0: 'Ground', 1: 'Tree'})
    - save_path: path to save the figure
    """
    FONTSIZE = 16

    class_ids = list(class_map.keys())
    counts = np.array([count_dict[class_id] for class_id in class_ids])
    ratios = counts / counts.sum()

    x_ticks = np.arange(len(class_ids))
    fig, ax1 = plt.subplots(figsize=(20, 8))
    # Histogram (left y-axis)
    bars = ax1.bar(x_ticks, counts, width=0.6, edgecolor='black', align='center', color='skyblue')
    ax1.set_ylabel('Number of Points', color='blue', fontsize=FONTSIZE)
    ax1.tick_params(axis='y', labelcolor='blue', labelsize=FONTSIZE)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([class_map[i] for i in class_ids], rotation=45, fontsize=FONTSIZE)
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
    ax2.plot(x_ticks, ratios, 'o--', color='darkred', label='Ratio')
    ax2.set_ylabel('Ratio (%)', color='darkred', fontsize=FONTSIZE)
    ax2.tick_params(axis='y', labelcolor='darkred', labelsize=FONTSIZE)
    ax2.set_ylim(0, max(ratios) * 1.2)
    ax2.set_yticks(np.linspace(0, max(ratios) * 1.2, 5))
    ax2.set_yticklabels([f"{r*100:.1f}%" for r in np.linspace(0, max(ratios) * 1.2, 5)], fontsize=FONTSIZE)

    # Add ratio text next to each dot
    for x, y in zip(x_ticks, ratios):
        ax2.annotate(f'{y*100:.2f}%',
                     xy=(x, y),
                     xytext=(5, 0),
                     textcoords='offset points',
                     ha='left', va='center', fontsize=FONTSIZE, color='darkred')

    fig.tight_layout()
    plt.savefig(f"outputs/{save_path}", dpi=150)
    plt.show()


import json
label_json_path = Path(f"/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/inlut3d/labels.json")
with open(label_json_path, 'r') as f:
    label_json = json.load(f)

label_map = {label_dict['code']:label_dict["label"] for label_dict in label_json}
pts_file_dir = Path(f"/media/fzhcis/Seagate Expansion Drive/point_cloud_data/inlut3d")
pts_file_paths = list(pts_file_dir.rglob("*.pts"))
pts_file_paths = sorted(pts_file_paths, key=lambda x: int(x.stem.split("_")[-1]))
pts_file_paths = pts_file_paths[:120]
# print(len(pts_file_paths))
# print(pts_file_paths[:10])
count_dict = grab_class_id(pts_file_paths, list(label_map.keys()))
plot_class_id_hist(count_dict, label_map, save_path=f"inlut3d_class_id_histogram_{len(pts_file_paths):03d}_merged.png")

# save count_dict to json
count_dict_path = Path(f"outputs/inlut3d_class_id_count_merged.json")
with open(count_dict_path, 'w') as f:
    out_dict = {
        "total_count": sum(count_dict.values()),
        "class_id_count": count_dict,
        "class_id_map": label_map
    }
    json.dump(count_dict, f, indent=4)