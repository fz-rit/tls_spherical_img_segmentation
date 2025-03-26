from collections import Counter
from prepare_dataset import load_data
import json
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Tuple
from matplotlib.colors import ListedColormap

config_file = 'params/paths_zmachine.json'
with open(config_file, 'r') as f:
    config = json.load(f)


def checkout_class_freq(config):
    train_dataset, _, _ = load_data(config)
    labels_map = train_dataset.labels_map
    num_classes = 5

    all_labels = []
    for img_patch, mask_patch in train_dataset:
        # mask_patch is of shape (H, W), containing class indices
        flat_mask = mask_patch.view(-1)
        all_labels.extend(flat_mask.tolist())

    class_counts = Counter(all_labels)
    total_pixels = sum(class_counts.values())

    print("Class Frequencies:")
    class_freq = {}
    for c in range(num_classes):
        freq = class_counts.get(c, 0) / total_pixels
        class_freq[c] = freq
        print(f"Class {c}-{labels_map[c]}: {freq*100:.2f}% of pixels")



# Class Frequencies:
# Class 0-Void: 23.98% of pixels
# Class 1-Miscellaneous: 2.34% of pixels
# Class 2-Leaves: 36.00% of pixels
# Class 3-Bark: 21.54% of pixels
# Class 4-Soil: 16.14% of pixels

def drop_zero_in_cm(cm: np.ndarray, verbose=False) -> np.ndarray:
    """
    Drop rank i if it is all zeros.

    Args:
    cm (numpy.ndarray): Confusion matrix.

    Returns:
    cm_zero_dropped (numpy.ndarray): Confusion matrix with rank i dropped if it is all zeros.
    """
    for idx in range(cm.shape[0]):
        row = cm[idx]
        col = cm[:, idx]
        if not np.any(col) or not np.any(row):
            if verbose:
                print("⚠️Warning: Dropping rank", idx, "from confusion matrix because it is all zeros.")
                print("Before dropping:")
                print(cm)
            cm = np.delete(cm, idx, axis=0)
            cm = np.delete(cm, idx, axis=1)
            if verbose:
                print("After dropping:")
                print(cm)
    return cm

def calc_metrics(true_flat: np.ndarray, 
                 pred_flat: np.ndarray, 
                 num_classes: int) -> Tuple[np.ndarray, float, float, float, float, float]:
    """
    Calculate evaluation metrics for semantic segmentation.

    Args:
    true_flat (numpy.ndarray): Flattened ground truth mask (1D array).
    pred_flat (numpy.ndarray): Flattened predicted mask (1D array).
    num_classes (int): Number of classes in the dataset.

    Returns:
    cm (numpy.ndarray): Confusion matrix.
    overall_accuracy (float): Overall accuracy.
    mAcc (float): Mean class accuracy.
    mIoU (float): Mean intersection over union.
    FWIoU (float): Frequency weighted intersection over union.
    dice_coefficient (float): Dice coefficient.
    """

    # Compute confusion matrix
    conf_mtx = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))
    # print("Confusion Matrix:")
    # print(cm)
    conf_mtx_zero_dropped = drop_zero_in_cm(conf_mtx)

    intersection = np.diag(conf_mtx_zero_dropped)
    # Overall Accuracy
    overall_accuracy = intersection.sum() / conf_mtx_zero_dropped.sum()

    # Mean class Accuracy
    class_accuracy = intersection / conf_mtx_zero_dropped.sum(axis=1)
    mAcc = np.nanmean(class_accuracy)

    # Intersection over Union (IoU) for each class
    union = conf_mtx_zero_dropped.sum(axis=1) + conf_mtx_zero_dropped.sum(axis=0) - np.diag(conf_mtx_zero_dropped)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    # Frequency Weighted IoU
    freq = conf_mtx_zero_dropped.sum(axis=1) / conf_mtx_zero_dropped.sum()
    FWIoU = (freq * IoU).sum()

    # Dice Coefficient for each class
    dice = 2 * intersection / (conf_mtx_zero_dropped.sum(axis=1) + conf_mtx_zero_dropped.sum(axis=0))
    dice_coefficient = np.nanmean(dice)

    return conf_mtx_zero_dropped, overall_accuracy, mAcc, mIoU, FWIoU, dice_coefficient


def custom_cmap():
    # # Define your custom colors
    # COLOR_TO_INDEX = {
    #     "0,0,0": 0,
    #     "128,0,128": 1,
    #     "165,42,42": 2,
    #     "0,128,0": 3,
    #     "255,165,0": 4,
    #     "255,255,0": 5
    # }

    # # Convert color strings to RGB tuples normalized to [0, 1]
    # color_list = []
    # for rgb_str in sorted(COLOR_TO_INDEX, key=COLOR_TO_INDEX.get):  # sort by index
    #     rgb = [int(x)/255 for x in rgb_str.split(',')]
    #     color_list.append(rgb)
    color_list = [
                    [0.0, 0.0, 0.0],           # index 0
                    [0.502, 0.0, 0.502],       # index 1
                    [0.647, 0.165, 0.165],     # index 2
                    [0.0, 0.502, 0.0],         # index 3
                    [1.0, 0.647, 0.0],         # index 4
                    [1.0, 1.0, 0.0]            # index 5
                ]


    # Create the colormap
    custom_cmap = ListedColormap(color_list)
    return custom_cmap
