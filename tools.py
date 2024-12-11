from collections import Counter
from prepare_dataset import load_data
import json
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Tuple

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
    pixel_accuracy (float): Pixel accuracy.
    mPA (float): Mean pixel accuracy.
    mIoU (float): Mean intersection over union.
    FWIoU (float): Frequency weighted intersection over union.
    dice_coefficient (float): Dice coefficient.
    """

    # Compute confusion matrix
    cm = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

    # Pixel Accuracy
    pixel_accuracy = np.diag(cm).sum() / cm.sum()

    # Mean Pixel Accuracy
    class_accuracy = np.diag(cm) / cm.sum(axis=1)
    mPA = np.nanmean(class_accuracy)

    # Intersection over Union (IoU) for each class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
    IoU = intersection / union
    mIoU = np.nanmean(IoU)

    # Frequency Weighted IoU
    freq = cm.sum(axis=1) / cm.sum()
    FWIoU = (freq * IoU).sum()

    # Dice Coefficient for each class
    dice = 2 * intersection / (cm.sum(axis=1) + cm.sum(axis=0))
    dice_coefficient = np.nanmean(dice)

    return cm, pixel_accuracy, mPA, mIoU, FWIoU, dice_coefficient