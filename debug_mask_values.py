#!/usr/bin/env python3

import numpy as np
from PIL import Image
from pathlib import Path
import json

# Load config
config_file = 'params/config_zmachine_forestsemantic.json'
with open(config_file, 'r') as f:
    CONFIG = json.load(f)

def check_mask_values():
    """Check the range of values in mask files"""
    root_dir = Path(CONFIG["root_dir"])
    dataset_name = CONFIG['dataset_name']
    num_classes = CONFIG['num_classes']
    
    print(f"Dataset: {dataset_name}")
    print(f"Expected number of classes: {num_classes}")
    print(f"Valid class indices should be: 0 to {num_classes-1}")
    print()
    
    # Check training masks
    train_mask_dir = root_dir / "train_val/mask"
    mask_paths = sorted(train_mask_dir.glob("*.png"))
    
    if not mask_paths:
        print(f"No mask files found in {train_mask_dir}")
        return
    
    print(f"Found {len(mask_paths)} mask files in training set")
    
    all_values = set()
    problematic_files = []
    
    for i, mask_path in enumerate(mask_paths[:5]):  # Check first 5 files
        print(f"\nChecking: {mask_path.name}")
        mask = np.array(Image.open(mask_path))
        unique_values = np.unique(mask)
        all_values.update(unique_values)
        
        print(f"  Shape: {mask.shape}")
        print(f"  Unique values: {unique_values}")
        print(f"  Min value: {mask.min()}, Max value: {mask.max()}")
        
        # Check for out-of-bounds values
        invalid_values = unique_values[(unique_values < 0) | (unique_values >= num_classes)]
        if len(invalid_values) > 0:
            problematic_files.append((mask_path.name, invalid_values))
            print(f"  ⚠️  PROBLEM: Invalid values found: {invalid_values}")
        else:
            print(f"  ✅ All values are valid (0 to {num_classes-1})")
    
    print(f"\n" + "="*50)
    print(f"SUMMARY:")
    print(f"All unique values across checked masks: {sorted(all_values)}")
    print(f"Expected range: 0 to {num_classes-1}")
    
    if problematic_files:
        print(f"\n⚠️  PROBLEMATIC FILES:")
        for filename, invalid_vals in problematic_files:
            print(f"  {filename}: invalid values {invalid_vals}")
    else:
        print(f"\n✅ All checked masks have valid values")

if __name__ == "__main__":
    check_mask_values()
