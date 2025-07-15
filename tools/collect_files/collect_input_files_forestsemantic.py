"""
Summary:
This script collects input files for the Mangrove3D project from a specified root directory.
It organizes the files into train/val and test directories based on their parent directory names.
It copies image cube files, YAML metadata files, and mask files into corresponding directories.
Author: fzhcis
Date: 2025-06-23
"""

import shutil
from pathlib import Path


train_val_dirs = ['plot1', 'plot3']
test_dirs = ['plot5']
root_dir = Path('/home/fzhcis/data/forest_semantic_preprocessed/output')
copy_root_dir = Path('/home/fzhcis/data/forest_semantic_preprocessed/seg2d/')

copy_test_dir = copy_root_dir / 'test'
copy_train_val_dir = copy_root_dir / 'train_val'


image_cube_files = list(root_dir.glob('**/img/*_image_cube.npy'))
yaml_files = list(root_dir.glob('**/img/*_image_cube_meta.yaml'))
mask_files = list(root_dir.glob('**/img/*_segmk_refined.png'))

image_cube_files.sort()
yaml_files.sort()
mask_files.sort()

print(f'Found {len(image_cube_files)} image cube files, {len(yaml_files)} yaml files, and {len(mask_files)} mask files.')

for i in range(39):
    image_cube_file = image_cube_files[i]
    yaml_file = yaml_files[i]
    mask_file = mask_files[i]

    parent_dir = image_cube_file.parent.parent.parent.parent
    assert yaml_file.parent.parent.parent.parent == parent_dir, \
        f'Yaml file {yaml_file.name} does not match image cube parent directory {parent_dir}'
    assert mask_file.parent.parent.parent.parent == parent_dir, \
        f'Mask file {mask_file.name} does not match image cube parent directory {parent_dir}'

    if any(x in str(parent_dir) for x in test_dirs):
        copy_dir = copy_test_dir
    elif any(x in str(parent_dir) for x in train_val_dirs):
        copy_dir = copy_train_val_dir
    else:
        raise ValueError(f'Unknown parent directory: {parent_dir}')

    image_cube_dest_dir = copy_dir / 'img_cube' 
    yaml_dest_dir = copy_dir / 'meta'
    mask_dest_dir = copy_dir / 'mask'

    image_cube_dest_dir.mkdir(parents=True, exist_ok=True)
    yaml_dest_dir.mkdir(parents=True, exist_ok=True)
    mask_dest_dir.mkdir(parents=True, exist_ok=True)
    image_cube_dest = image_cube_dest_dir / f"proj{i:03d}_{image_cube_file.name}"
    yaml_dest = yaml_dest_dir / f"proj{i:03d}_{yaml_file.name}"
    mask_dest = mask_dest_dir / f"proj{i:03d}_{mask_file.name}"


    shutil.copy2(image_cube_file, image_cube_dest)
    shutil.copy2(yaml_file, yaml_dest)
    shutil.copy2(mask_file, mask_dest)

    print(f'Copied {image_cube_file.name} to {image_cube_dest}')
    print(f'Copied {yaml_file.name} to {yaml_dest}')
    print(f'Copied {mask_file.name} to {mask_dest}')