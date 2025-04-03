from collections import Counter
from prepare_dataset import load_data
import json
# from sklearn.metrics import confusion_matrix
import numpy as np
from typing import Tuple
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import torch
import time
from pathlib import Path



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


# def calc_metrics(true_flat: np.ndarray, 
#                  pred_flat: np.ndarray, 
#                  num_classes: int) -> dict:
#     """
#     Calculate evaluation metrics for semantic segmentation.
#     """
#     conf_mtx = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

#     # Keep only the classes that appear in either true or pred
#     mask = (conf_mtx.sum(axis=0) + conf_mtx.sum(axis=1)) > 0
#     conf_mtx = conf_mtx[mask][:, mask]

#     intersection = np.diag(conf_mtx)
#     total = conf_mtx.sum()
#     oAccu = intersection.sum() / total if total > 0 else 0

#     # Mean class Accuracy
#     class_accuracy = intersection / np.maximum(conf_mtx.sum(axis=1), 1)
#     mAcc = np.nanmean(class_accuracy)

#     # IoU
#     union = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0) - intersection
#     IoU = intersection / np.maximum(union, 1)
#     mIoU = np.nanmean(IoU)

#     # Frequency Weighted IoU
#     freq = conf_mtx.sum(axis=1) / total if total > 0 else np.zeros_like(conf_mtx.sum(axis=1))
#     FWIoU = np.sum(freq * IoU)

#     # Dice Coefficient
#     denom = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0)
#     dice = 2 * intersection / np.maximum(denom, 1)
#     dice_coefficient = np.nanmean(dice)

#     return {
#         'confusion_matrix': conf_mtx,
#         'oAcc': oAccu,
#         'mAcc': mAcc,
#         'mIoU': mIoU,
#         'FWIoU': FWIoU,
#         'dice_coefficient': dice_coefficient
#     }



# def calc_oAccu_mIoU(true_flat: np.ndarray,
#                     pred_flat: np.ndarray,
#                     num_classes: int) -> Tuple[float, float]:
#     """
#     Calculate overall accuracy and mean intersection over union (IoU).

#     Args:
#     true_flat (numpy.ndarray): Flattened ground truth mask (1D array).
#     pred_flat (numpy.ndarray): Flattened predicted mask (1D array).
#     num_classes (int): Number of classes in the dataset.

#     Returns:
#     oAccu (float): Overall accuracy.
#     mIoU (float): Mean intersection over union.
#     """

#     metrics_dict = calc_metrics(true_flat, pred_flat, num_classes)
#     oAcc = metrics_dict['oAcc']
#     mIoU = metrics_dict['mIoU']
#     return oAcc, mIoU

    


def custom_cmap():
    color_list = [
                    [0.0, 0.0, 0.0],           # index 0
                    [0.502, 0.0, 0.502],       # index 1
                    [0.647, 0.165, 0.165],     # index 2
                    [0.0, 0.502, 0.0],         # index 3
                    [1.0, 0.647, 0.0],         # index 4
                    [1.0, 1.0, 0.0]            # index 5
                ]


    custom_cmap = ListedColormap(color_list)
    return custom_cmap


def get_pil_palette():
    color_list = [
        [0.0, 0.0, 0.0],           # black
        [0.502, 0.0, 0.502],       # purple
        [0.647, 0.165, 0.165],     # brown
        [0.0, 0.502, 0.0],         # green
        [1.0, 0.647, 0.0],         # orange
        [1.0, 1.0, 0.0]            # yellow
    ]

    # Convert to 0–255 and flatten
    flat_palette = [int(x * 255) for rgb in color_list for x in rgb]

    # Pad with zeros to length 768 (PIL expects full 256 colors x 3 channels)
    flat_palette += [0] * (768 - len(flat_palette))

    return flat_palette


# def visualize_losses(train_losses, val_losses, plt_save_path, clip_val_loss=True):
#     """
#     Plot and save training and validation losses.

#     Parameters:
#     - train_losses (list or np.array): Training loss values over epochs.
#     - val_losses (list or np.array): Validation loss values over epochs.
#     - plt_save_path (str): Path to save the plot image.
#     - clip_val_loss (bool): Whether to clip validation loss for clearer plots.
#     """

#     train_losses = np.array(train_losses)
#     val_losses = np.array(val_losses)

#     # Prevent empty input or division by zero
#     if len(train_losses) == 0 or len(val_losses) == 0:
#         print("Error: Empty loss lists.")
#         return

#     if clip_val_loss:
#         max_train = max(train_losses)
#         val_losses = val_losses.clip(max=3*max_train)
#         print(f"Clipping validation loss to max of 3 times the max training loss: {3*max_train}")

#     plt.figure(figsize=(8, 5))
#     plt.plot(train_losses, label='Train Loss', marker='o')
#     plt.plot(val_losses, label='Val Loss' + (' (clipped)' if clip_val_loss else ''), marker='x')

#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Losses')
#     plt.legend()
#     plt.grid(True)

#     # Ensure save directory exists
#     plt_save_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(plt_save_path)
#     plt.close()

#     print(f"---- Loss plot saved at: {plt_save_path} ----")


# def visualize_metrics(train_oAccus, val_oAccus, train_mIoUs, val_mIoUs, plt_save_path):
#     """
#     Plot and save training and validation metrics.

#     Parameters:
#     - train_oAccus (list or np.array): Training overall accuracy values over epochs.
#     - val_oAccus (list or np.array): Validation overall accuracy values over epochs.
#     - train_mIoUs (list or np.array): Training mean IoU values over epochs.
#     - val_mIoUs (list or np.array): Validation mean IoU values over epochs.
#     - plt_save_path (str): Path to save the plot image.
#     """
#     train_oAccus = np.array(train_oAccus)
#     val_oAccus = np.array(val_oAccus)
#     train_mIoUs = np.array(train_mIoUs)
#     val_mIoUs = np.array(val_mIoUs)

#     # Prevent empty input or division by zero
#     if len(train_oAccus) == 0 or len(val_oAccus) == 0:
#         print("Error: Empty metric lists.")
#         return

#     plt.figure(figsize=(8, 5))
#     plt.plot(train_oAccus, label='Train Overall Accuracy', marker='o')
#     plt.plot(val_oAccus, label='Val Overall Accuracy', marker='o')
#     plt.plot(train_mIoUs, label='Train Mean IoU', marker='x')
#     plt.plot(val_mIoUs, label='Val Mean IoU', marker='x')

#     plt.xlabel('Epoch')
#     plt.ylabel('Metrics')
#     plt.title('Training and Validation Metrics')
#     plt.legend()
#     plt.grid(True)

#     # Ensure save directory exists
#     plt_save_path.parent.mkdir(parents=True, exist_ok=True)
#     plt.savefig(plt_save_path)
#     plt.close()

#     print(f"---- Metrics plot saved at: {plt_save_path} ----")



def save_model_locally(model, model_dir, model_name_prefix, dummy_shape):
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"----Created directory {model_dir}----")

    save_model_path = model_dir / f'{model_name_prefix}.pth'
    torch.save(model.state_dict(), save_model_path)
    print(f"----Model saved at {save_model_path}----")

    onnx_model_path = model_dir / f'{model_name_prefix}.onnx'

    # Create a dummy input with the same shape as your input
    dummy_input = torch.randn(dummy_shape).to('cuda')  # replace H, W as needed
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,  # common version; increase if needed
        do_constant_folding=True
    )
    print(f"----ONNX model saved at {onnx_model_path}----")


