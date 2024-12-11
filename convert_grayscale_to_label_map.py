import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
import json
from pathlib import Path


def convert_grayscale_to_label_map(gray_image: np.ndarray) -> np.ndarray:
    """
    Convert a grayscale image to a segmentation label map with labels ranging from 0 to 4.
    Outlier pixels are assigned labels based on the spatially nearest labeled pixel to promote
    connectivity of regions. This implementation uses OpenCV for efficiency and conciseness.

    Parameters
    ----------
    gray_image : np.ndarray
        Input grayscale image as a NumPy array.

    Returns
    -------
    label_map : np.ndarray
        Segmentation label map with integer labels from 0 to 4.
    """
    # Define the label-to-grayscale-values mapping with approximate values
    label_to_gray_values = {
        0: [0, 1], # Void
        1: [99, 100, 101], # Miscellaneous
        2: [121, 122, 123], # Leaves
        3: [140, 141, 142], # Bark
        4: [233, 234, 235], # Soil
    }

    # Build a grayscale value to label mapping
    gray_value_to_label = {}
    for label, gray_values in label_to_gray_values.items():
        for gray_value in gray_values:
            gray_value_to_label[gray_value] = label

    # Initialize the label map with -1, using a signed integer type
    label_map = np.full_like(gray_image, fill_value=-1, dtype=np.int32)

    # Apply the mapping to the image
    for gray_value, label in gray_value_to_label.items():
        label_map[gray_image == gray_value] = label

    # Create a mask of valid labels
    valid_mask = label_map >= 0

    # Check if there are any outliers to process
    if np.any(~valid_mask):
        # Invert the valid mask for OpenCV distance transform (foreground pixels are non-zero)
        inverted_mask = (~valid_mask).astype(np.uint8)

        # Perform distance transform and get labels of nearest valid pixels
        distance, labels = cv2.distanceTransformWithLabels(
            inverted_mask,
            distanceType=cv2.DIST_L2,
            maskSize=5,
            labelType=cv2.DIST_LABEL_PIXEL
        )

        # Adjust labels to get indices of nearest valid pixels
        nearest_labels = labels - 1  # OpenCV labels start from 1

        # Get coordinates of all valid pixels
        valid_coords = np.column_stack(np.nonzero(valid_mask))

        # Map labels to outlier pixels based on nearest valid pixel
        outlier_coords = np.column_stack(np.nonzero(~valid_mask))
        nearest_valid_indices = nearest_labels[~valid_mask].astype(np.int32)

        # Ensure indices are within valid range
        nearest_valid_indices = np.clip(nearest_valid_indices, 0, len(valid_coords) - 1)

        # Assign labels from nearest valid pixels to outlier pixels
        nearest_valid_coords = valid_coords[nearest_valid_indices]
        label_map[~valid_mask] = label_map[tuple(nearest_valid_coords.T)]

    return label_map.astype(np.uint8)  # Convert to uint8 for consistent data type


if __name__ == '__main__':

    json_path = Path('./input_params/convert_grayscale_to_label_map_zmachine.json')
    with open(json_path, 'r') as file:
        config = json.load(file)

    root_dir = Path(config["root_dir"])
    seg_map_mono_path = root_dir / config["seg_map_mono_filename"]
    output_dir = Path(config["output_dir"])

    seg_map_gray = io.imread(seg_map_mono_path)  # (height, width)
    seg_map_label = convert_grayscale_to_label_map(seg_map_gray)
    output_file = output_dir / config["seg_map_label_filename"]
    io.imsave(output_file, seg_map_label)
    print(f"Saved label map to: {output_file}") 
    # To view the output in imagej: 
    # open the file, go to image -> Adjust -> Brightness/Contrast (hotkey Ctrl+Shift+C)
    # and click on Auto.