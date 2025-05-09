import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Any
import json
import yaml
from pprint import pprint
from tools.get_mean_std import read_mean_std_from_yaml, normalize_img_per_channel

PATCH_PER_IMAGE = 5
# PATCH_HEIGHT = 544
# PATCH_WIDTH = 256
NUM_CLASSES = 6



def pad_img_or_mask(arr):
    """
    Padding a 2D mask or 3D image to make it divisible by 32.
    
    Args:
        arr (np.ndarray): Input array of shape (H, W) or (H, W, C)
    
    Returns:
        np.ndarray: Padded array with shape:
                    - (H + pad_h, W + pad_w) if input was 2D
                    - (H + pad_h, W + pad_w, C) if input was 3D
    """
    h, w = arr.shape[:2]
    pad_h = (32 - (h % 32)) % 32 // 2 # handle special case when h % 32 == 0
    pad_w = (32 - (w % 32)) % 32 // 2

    if arr.ndim == 2:
        # It's a 2D mask
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif arr.ndim == 3:
        # It's a 3D image cube
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    else:
        raise ValueError(f"Input array must be 2D or 3D, got shape {arr.shape}")
    return padded.astype(np.float32)
    
    
def pad_img_and_mask(image, mask):
    """
    Padding an image and its corresponding mask to the target size.
    
    Args:
        image (np.ndarray): Input image of shape (H, W, C)
        mask (np.ndarray): Input mask of shape (H, W)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Padded image and mask.
    """
    padded_image = pad_img_or_mask(image)
    padded_mask = pad_img_or_mask(mask)
    return padded_image, padded_mask

def depad_img_or_mask(padded_arr, original_shape):
    """
    Remove symmetric padding from a padded array.
    
    Args:
        padded_arr (np.ndarray): Padded image or mask, shape (H + pad_h*2, W + pad_w*2) or (H + pad_h*2, W + pad_w*2, C)
        original_shape (Tuple[int, int] or Tuple[int, int, int]): Original shape of the array before padding.
    
    Returns:
        np.ndarray: Depadded array of shape original_shape.
    """
    orig_h, orig_w = original_shape[:2]
    pad_h = (32 - (orig_h % 32)) % 32 // 2
    pad_w = (32 - (orig_w % 32)) % 32 // 2

    if padded_arr.ndim == 2:
        return padded_arr[pad_h:pad_h + orig_h, pad_w:pad_w + orig_w]
    elif padded_arr.ndim == 3:
        return padded_arr[pad_h:pad_h + orig_h, pad_w:pad_w + orig_w, :]
    else:
        raise ValueError(f"Input array must be 2D or 3D, got shape {padded_arr.shape}")


class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_file_paths, 
                 mask_paths=None,  # Ground truth mask may not be available.
                 input_channels=None,
                 patch_splits=PATCH_PER_IMAGE, 
                 transform=None, 
                 labels_map=None):
        self.image_file_paths = image_file_paths
        self.input_channels = input_channels
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.labels_map = labels_map
        self.patch_idx_ls = [(i, p) for i in range(len(self.image_file_paths)) for p in range(patch_splits)]

        
    def __len__(self):
        return len(self.patch_idx_ls)
    
    def __getitem__(self, idx):
        i, p = self.patch_idx_ls[idx]

        image_cube = np.load(self.image_file_paths[i])
        image_cube = image_cube[:, :, self.input_channels]  # Select only the input channels

        image_cube = normalize_img_per_channel(image_cube)
        mask = np.array(Image.open(self.mask_paths[i]))

        patch_width = image_cube.shape[1] // self.patch_splits
        x_start = p * patch_width
        x_end = x_start + patch_width
        image_patch = image_cube[:, x_start:x_end, :]
        mask_patch = mask[:, x_start:x_end]

        image_patch, mask_patch = pad_img_and_mask(image_patch, mask_patch)

        # Apply Albumentations transform if provided
        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed["image"]  # shape: (C, H, W) after ToTensorV2
            mask_patch = transformed["mask"]    # shape: (H, W) or (1, H, W)
            mask_patch = mask_patch.long()
        else:
            print("⚠️ WARNING: No transform provided. Converting to tensor manually.")
            image_patch = torch.from_numpy(image_patch).permute(2, 0, 1).float() / 255.0
            mask_patch = torch.from_numpy(mask_patch).long()

        return image_patch, mask_patch



def get_data_paths(config: Dict, verbose: bool = False) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Get image and mask paths from the config dictionary.

    Args:
    config (dict): Configuration dictionary.
    """
    root_dir = Path(config['root_dir'])
    train_val_test_split_file = root_dir / config['train_val_test_split_file']

    with open(train_val_test_split_file, 'r') as f:
        split_paths = yaml.safe_load(f)

    image_file_path_dict = split_paths['img']
    mask_path_dict = split_paths['mask']

    
    if verbose:
        pprint(image_file_path_dict)
        pprint(mask_path_dict)

    return image_file_path_dict, mask_path_dict


# class BrightnessContrastOnlyFirst3Channels(A.ImageOnlyTransform):
#     def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
#         super().__init__(p=p)
#         self.brightness_limit = brightness_limit
#         self.contrast_limit = contrast_limit
#         self.bc_transform = A.RandomBrightnessContrast(
#             brightness_limit=self.brightness_limit,
#             contrast_limit=self.contrast_limit,
#             p=1.0  # always apply inside
#         )

#     def apply(self, img, **params):
#         # img shape [H, W, C]
#         # separate first 3 channels from geometry
#         first3 = img[:, :, :3]
#         last3 = img[:, :, 3:]

#         # Albumentations needs an image with shape [H, W, C]
#         # apply bc_transform to the first3 only
#         first3_aug = self.bc_transform(image=first3)['image']
#         # re-concat
#         return np.concatenate([first3_aug, last3], axis=2)


class ChannelShuffleGroups(A.ImageOnlyTransform):
    def __init__(self, groups, p=0.5):
        """
        groups: list of lists, each inner list contains channel indices to be shuffled among themselves.
                e.g., [[0,1,2], [3,4,5]] means shuffle (0,1,2) among themselves, and shuffle (3,4,5) among themselves.
        p: probability of applying the transform
        """
        super(ChannelShuffleGroups, self).__init__(p=p)
        self.groups = groups

    def apply(self, img, **params):
        """
        img: A NumPy array of shape (H, W, C)
        """
        # Make a copy so we can reassign channels safely.
        out = img.copy()

        # Shuffle channels within each group.
        for group in self.groups:
            # Create a random permutation of the channel indices in this group.
            shuffled = np.random.permutation(group)

            # Assign them back in place:
            for old_ch, new_ch in zip(group, shuffled):
                out[:, :, old_ch] = img[:, :, new_ch]

        return out


def trasform_by_channls(input_channels:list, p=0.5, mean_std_dict=None):
    """
    
    p: probability of applying the transform
    """
    # channel_groups = [
    #     [0, 1, 2], # Intensity - Z - Range
    #     [3, 4, 5], # Adjusted Intensity - ZInv - Range
    #     [6, 7, 8], # Pseudo-RGB from normals
    #     [9, 10, 11], # PCA
    #     [12, 13, 14], # MNF
    #     [15, 16, 17], # ICA
    # ]
    if len(input_channels) == 3:
        shuffle_groups = [[0, 1, 2]]
        # norm_mean = [0.485, 0.456, 0.406]
        # norm_std = [0.229, 0.224, 0.225]
    elif len(input_channels) == 6:
        shuffle_groups = [[0, 1, 2], [3, 4, 5]]
        # norm_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        # norm_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    elif len(input_channels) == 9:
        shuffle_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        # norm_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406, 0.485, 0.456, 0.406]
        # norm_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225, 0.229, 0.224, 0.225]
    elif len(input_channels) == 15:
        shuffle_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], 
                          [9, 10, 11], [12, 13, 14]]
        # norm_mean = [0.485, 0.456, 0.406, 0.485, 0.456, 0.406,
        #                 0.485, 0.456, 0.406, 0.485, 0.456, 0.406,
        #                 0.485, 0.456, 0.406]
        # norm_std = [0.229, 0.224, 0.225, 0.229, 0.224, 0.225,
        #                 0.229, 0.224, 0.225, 0.229, 0.224, 0.225,
        #                 0.229, 0.224, 0.225]

    norm_mean = mean_std_dict['mean']
    norm_std = mean_std_dict['std']

    transform = A.Compose([
        ChannelShuffleGroups(groups=shuffle_groups, p=p),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=10, p=0.5),
        # BrightnessContrastOnlyFirst3Channels(p=0.5), # # !!!!put it aside for now.
        # A.Normalize(mean=norm_mean, std=norm_std),  ## Due to the distribution difference, training without normalization is better.
        A.pytorch.ToTensorV2()
    ],
    additional_targets={'mask': 'mask'})

    return transform


def load_data(config, input_channels=None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Get image and mask paths
    image_file_path_dict, mask_path_dict = get_data_paths(config)
    num_workers = config['num_workers']
    mean_std_yaml_file = Path(config['root_dir']) / config['mean_std_file']
    mean_std_dict = read_mean_std_from_yaml(mean_std_yaml_file)
    labels_map = {
        0: 'Void',
        1: 'Ground & Water',
        2: 'Stem',
        3: 'Canopy',
        4: 'Roots',
        5: 'Objects'
    }

    # ---- Define Albumentations transforms ----
    train_transform = trasform_by_channls(input_channels=input_channels, p=0.5, mean_std_dict=mean_std_dict)

    val_transform = A.Compose([
                                # A.Normalize(
                                #     mean=mean_std_dict['mean'],
                                #     std=mean_std_dict['std']
                                # ),
                                ToTensorV2()
                            ], additional_targets={'mask': 'mask'})

    # ---- Create dataloaders for train, val, and test ----
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        image_file_paths = image_file_path_dict[split]
        mask_paths = mask_path_dict[split]

        dataset = SegmentationPatchDataset(
            image_file_paths=image_file_paths,
            input_channels=input_channels,
            mask_paths=mask_paths,
            patch_splits=PATCH_PER_IMAGE,
            transform=train_transform if split == 'train' else val_transform,
            labels_map=labels_map
        )

        bt_size = PATCH_PER_IMAGE if split == 'test' else config['train_batch_size']
        dataloaders[split] = DataLoader(dataset, 
                                        batch_size=bt_size, 
                                        shuffle=True if split == 'train' else False, 
                                        num_workers=num_workers)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    return train_loader, val_loader, test_loader




if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    input_channels = config['input_channels_ls'][0]
    train_loader, val_loader, test_loader = load_data(config, input_channels=input_channels)
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")
    print("Data loaders created successfully!")
    print("Sample batch from training loader:")
    imgs, masks = next(iter(train_loader))
    print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")

    print("Sample batch from validation loader:")
    imgs, masks = next(iter(val_loader))
    print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")

    print("Sample batch from test loader:")
    imgs, masks = next(iter(test_loader))
    print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
    
    