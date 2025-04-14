from prepare_dataset import load_image_cube_and_metadata
import matplotlib.pyplot as plt
from PIL import Image
from tools.load_tools import config
from pathlib import Path
import numpy as np
from tools.visualize_tools import plot_channel_histograms

image_cube_path = Path(config['root_dir']) / config['image_dir'] / 'UMBCBL009_1830507489_image_cube.npy'
image_meta_path = image_cube_path.parent / 'UMBCBL009_1830507489_image_cube_metadata.json'

output_stem = image_cube_path.stem
image_cube, metadata = load_image_cube_and_metadata(image_cube_path, image_meta_path)
image_cube = image_cube.astype(np.float32) # (H, W, C)
image_cube = image_cube.transpose(2, 0, 1) # Change to (C, H, W)


channel_names = ['Intensity', 
            'Z Map Inverse', 
            'Range', 
            'Curvature', 
            'Roughness', 
            'Rn',
            'Gn',
            'Bn']
plot_channel_histograms(image_cube, channel_names=channel_names, bins=50)
plt.show()