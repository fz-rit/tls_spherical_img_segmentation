

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.load_tools import CONFIG
from tools.metrics_tools import compute_band_correlation, compute_pca_components, compute_mnf, compute_ica
from tools.visualize_tools import plot_correlation_matrix, plot_pca_components, plot_rgb_permutations
from pathlib import Path
import numpy as np
from prepare_dataset import load_image_cube_and_metadata
import matplotlib.pyplot as plt
from PIL import Image

image_cube_path = Path(CONFIG['root_dir']) / CONFIG['image_dir'] / 'SICK_1642331006_image_cube.npy'
image_meta_path = image_cube_path.parent / 'SICK_1642331006_image_cube_metadata.json'

output_stem = image_cube_path.stem
image_cube, metadata = load_image_cube_and_metadata(image_cube_path, image_meta_path)
image_cube = image_cube.astype(np.float32) # (H, W, C)
image_cube = image_cube.transpose(2, 0, 1) # Change to (C, H, W)


image_cube = np.concatenate([image_cube[[0,1,2], :, :], image_cube[[5, 6, 7], :, :]], axis=0) # delete 3rd and 4th bands; Curvature and Roughness
# print(image_cube.shape)
band_names = ['Intensity', 
            'Z Map Inverse', 
            'Range', 
            # 'Curvature', 
            # 'Roughness', 
            'Rn',
            'Gn',
            'Bn']


corr_matrix = compute_band_correlation(image_cube)
plot_correlation_matrix(corr_matrix, band_names = band_names, output_stem=output_stem)


pcs, pca = compute_pca_components(image_cube, n_components=3)
mnf_components = compute_mnf(image_cube, n_components=3)
ica_components = compute_ica(image_cube, n_components=3)

for components, name in zip([pcs, mnf_components, ica_components], ['PCA', 'MNF', 'ICA']):
    out_file = f"{output_stem}_{name}"
    plot_pca_components(components, output_stem=out_file)
    plot_rgb_permutations(components, output_stem=out_file)

plt.show()