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


