from collections import Counter
import torch
from prepare_dataset import load_data


train_dataset, _, _ = load_data('params/paths_zmachine.json')

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
    print(f"Class {c}: {freq*100:.2f}% of pixels")

