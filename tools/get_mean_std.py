import numpy as np
import yaml
from pathlib import Path
import json


def normalize_img_per_channel(img):

    for channel in range(img.shape[2]):
        img[:, :, channel] = (img[:, :, channel] - img[:, :, channel].min()) / (img[:, :, channel].max() - img[:, :, channel].min())
    
    return img

def get_mean_std_from_img_paths(img_paths, input_channels):
    """
    Compute mean and std per channel over the entire dataset.
    
    Args:
        img_paths (list): List of image file paths.
        input_channels (list): List of input channels.
    
    Returns:
        Tuple of (mean, std) as lists.
    """
    mean_ls, std_ls = [], []
    for img_path in img_paths:
        image_cube = np.load(img_path)
        image_cube = image_cube[:, :, input_channels]
        image_cube = normalize_img_per_channel(image_cube)
        mean = np.mean(image_cube, axis=(0, 1)) # (C,)
        std = np.std(image_cube, axis=(0, 1))

        mean_ls.append(mean)
        std_ls.append(std)
    
    mean = np.round(np.mean(np.vstack(mean_ls), axis=0), 4) # (C,)
    std = np.round(np.mean(np.vstack(std_ls), axis=0), 4) # (C,)

    return mean.tolist(), std.tolist()
    

def read_mean_std_from_yaml(file_path):
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File {file_path} does not exist. Run the tools/get_mean_std.py script to generate it.")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    mean_std_dict = data['mean_std']
    return mean_std_dict

if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    root_dir = Path(config['root_dir'])
    train_val_test_split_file = root_dir / config['train_val_test_split_file']

    with open(train_val_test_split_file, 'r') as f:
        split_paths = yaml.safe_load(f)

    image_file_path_dict = split_paths['img']
    input_channels = config['input_channels']
    in_ch_str = "_".join([str(ch) for ch in input_channels])
    output_file = Path(config['root_dir']) / f'mean_std_{in_ch_str}.yaml'
    mean_std_dict = {}
    image_file_paths = image_file_path_dict['train']
    mean, std = get_mean_std_from_img_paths(image_file_paths, input_channels)
    mean_std_dict['mean'] = mean
    mean_std_dict['std'] = std
    output_dict = {
        'input_channels': input_channels,
        'mean_std': mean_std_dict
    }
    with open(output_file, 'w') as f:
        yaml.dump(output_dict, f)
    print(f"Mean and std values of the train data saved to {output_file}")