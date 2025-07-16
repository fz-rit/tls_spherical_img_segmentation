from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import json
mask_dir = Path("/home/fzhcis/mylab/data/forestsemantic_seg2d/seg2d/test/mask")
mask_files = list(mask_dir.glob("*mask.png"))
mask_files.sort()

label_json = Path("/home/fzhcis/Downloads/ForestSemantic/output/test/input/labels.json")

def get_label_map(label_file: Path) -> dict:
    """Get label map from config."""
    with open(label_file, 'r') as f:
        label_json = json.load(f)
    label_map = {label_dict['code']: label_dict["label"] for label_dict in label_json}
    label_map = {k: v for k, v in label_map.items() if k < 18}
    return label_map

def get_histogram(mask_file: Path) -> dict:
    mask = Image.open(mask_file)
    mask = np.array(mask)
    unique, counts = np.unique(mask, return_counts=True)
    histogram = {"class": unique, "count": counts}
    return histogram

label_map = get_label_map(label_json)
for mask_file in mask_files:
    histogram = get_histogram(mask_file)
    print(f"Processing {mask_file.name} with shape {mask_file.stat().st_size} bytes")
    print(f"Histogram: {histogram}")
    # histogram['class'] = histogram['class'].map(label_map)
    # histogram.to_csv(mask_file.with_suffix('.csv'), index=False)
    # print(f"Processed {mask_file.name}, saved histogram to {mask_file.with_suffix('.csv')}")