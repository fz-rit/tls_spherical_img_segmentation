from pathlib import Path
import numpy as np
from PIL import Image
import json
import argparse

# label_json = Path("/home/fzhcis/Downloads/ForestSemantic/output/test/input/labels.json")

# def get_label_map(label_file: Path) -> dict:
#     """Get label map from config."""
#     with open(label_file, 'r') as f:
#         label_json = json.load(f)
#     label_map = {label_dict['code']: label_dict["label"] for label_dict in label_json}
#     label_map = {k: v for k, v in label_map.items() if k < 18}
#     return label_map
# label_map = get_label_map(label_json)

def read_mask(mask_file: Path) -> np.ndarray:
    """Read mask image and convert to numpy array."""
    mask = Image.open(mask_file)
    mask = np.array(mask)
    return mask

def mergeclass(mask: np.ndarray, src_class: int, target_class: int) -> np.ndarray:
    mask[mask == src_class] = target_class
    return mask

def get_histogram(mask: np.ndarray) -> dict:
    unique, counts = np.unique(mask, return_counts=True)
    histogram = {"class": unique, "count": counts}
    return histogram

def save_merged_mask(merged_map: np.ndarray, out_file: Path):
    """
    Save the class-merged segmentation map to a file.
    """
    Image.fromarray(merged_map.astype(np.uint8)).save(out_file)

    print(f"Class merged map saved to {out_file.name}")



parser = argparse.ArgumentParser(description='Prepare dataset for segmentation')
parser.add_argument('--mask_dir', type=str, 
                    help='Path to the directory containing mask images')
args = parser.parse_args()

mask_dir = Path(args.mask_dir)
mask_files = list(mask_dir.glob("*mask.png"))
mask_files.sort()

merged_mask_dir = mask_dir.parent / "merged_mask"
merged_mask_dir.mkdir(exist_ok=True)

for mask_file in mask_files:
    print(f"Processing {mask_file.name} with shape {mask_file.stat().st_size} bytes")
    mask = read_mask(mask_file)
    histogram = get_histogram(mask)
    print(f"Histogram before merge: {histogram}")
    mask = mergeclass(mask, 7, 0)
    histogram = get_histogram(mask)
    print(f"Histogram after merge: {histogram}")

    # save_merged_mask(mask, merged_mask_dir / f"{mask_file.stem}_merged.png")