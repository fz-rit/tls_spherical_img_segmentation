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
from pprint import pprint

PATCH_PER_IMAGE = 6
PATCH_WIDTH = 256
NUM_CLASSES = 6

def load_image_cube_and_metadata(image_cube_path: Path, metadata_path: Path) -> Dict[str, Any]:
    """
    Loads an image cube and its metadata from saved .npy files.

    Parameters:
    - image_cube_path: The path to the saved image cube file.
    - metadata_path: The path to the saved metadata file.

    Returns:
    - A dictionary containing the image cube and metadata.
    """
    
    # Load the image cube (8-channel data)
    image_cube = np.load(image_cube_path, allow_pickle=True)
    
    # Load the metadata from .json file
    with open(metadata_path, 'r') as file:
        metadata = json.load(file)

    return image_cube, metadata


def resize_image_or_mask(arr, target_size):
    """
    Resize a 2D mask or 3D image cube using PIL.
    
    Args:
        arr (np.ndarray): Input array of shape (H, W) or (H, W, C)
        target_size (tuple): (target_height, target_width)
    
    Returns:
        np.ndarray: Resized array with shape:
                    - (target_H, target_W) if input was 2D
                    - (target_H, target_W, C) if input was 3D
    """
    target_h, target_w = target_size

    # Ensure dtype is supported by PIL
    if arr.dtype == np.int64:
        arr = arr.astype(np.int32)
    elif arr.dtype == np.float64:
        arr = arr.astype(np.float32)

    if arr.ndim == 2:
        # It's a 2D mask → use NEAREST
        img = Image.fromarray(arr)
        resized = img.resize((target_w, target_h), resample=Image.NEAREST)
        return np.array(resized)

    elif arr.ndim == 3:
        # It's an image cube → resize each channel using BILINEAR
        resized_channels = []
        for c in range(arr.shape[2]):
            img_c = Image.fromarray(arr[:, :, c])
            img_resized = img_c.resize((target_w, target_h), resample=Image.BILINEAR)
            resized_channels.append(np.array(img_resized))
        return np.stack(resized_channels, axis=2)

    else:
        raise ValueError(f"Input array must be 2D or 3D, got shape {arr.shape}")
    
def resize_image_and_mask(image, mask, target_size):
    """
    Resize an image and its corresponding mask to the target size.
    
    Args:
        image (np.ndarray): Input image of shape (H, W, C)
        mask (np.ndarray): Input mask of shape (H, W)
        target_size (tuple): (target_height, target_width)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Resized image and mask.
    """
    resized_image = resize_image_or_mask(image, target_size)
    resized_mask = resize_image_or_mask(mask, target_size)
    return resized_image, resized_mask



class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_file_paths, 
                 image_meta_paths,
                 mask_paths=None,  # Ground truth mask may not be available.
                 patch_splits=PATCH_PER_IMAGE, 
                 transform=None, 
                 mask_transform=None,
                 labels_map=None):
        self.image_file_paths = image_file_paths
        self.image_meta_paths = image_meta_paths
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.mask_transform = mask_transform
        self.labels_map = labels_map
        self.patch_coords = []  # Will store tuples: [(img_index, x_start, x_end)...]

        patch_width = PATCH_WIDTH
        
        # Create patch coordinates for all images
        for i in range(len(self.image_file_paths)):
            for p in range(patch_splits):
                x_start = p * patch_width
                x_end = x_start + patch_width
                self.patch_coords.append((i, x_start, x_end))
        
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, x_start, x_end = self.patch_coords[idx]

        image_cube, _ = load_image_cube_and_metadata(self.image_file_paths[i], self.image_meta_paths[i])
        mask = np.array(Image.open(self.mask_paths[i]))
        # Resize the input image and mask to be divisible by 32: (540, 1440) -> (512, 1536)
        image_cube_resized, mask_resized = resize_image_and_mask(image_cube, mask, (512, 1536))

        image_patch = image_cube_resized[:, x_start:x_end, :]
        mask_patch = mask_resized[:, x_start:x_end]

        # Apply Albumentations transform if provided
        if self.transform:
            transformed = self.transform(image=image_patch, mask=mask_patch)
            image_patch = transformed["image"]  # shape: (C, H, W) after ToTensorV2
            mask_patch = transformed["mask"]          # shape: (H, W) or (1, H, W)
            mask_patch = mask_patch.long()
        else:
            # If no transform, convert to torch.Tensor manually
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
    image_dir = root_dir / Path(config['image_dir'])
    mask_dir = root_dir / Path(config['mask_dir'])
    train_val_test_split_file = Path(config['root_dir']) / config['train_val_test_split_file']

    with open(train_val_test_split_file, 'r') as file:
        tvt_splits = json.load(file)

    image_file_path_dict = {}
    image_meta_path_dict = {}
    mask_path_dict = {}

    for split in ['train', 'val', 'test']:
        image_file_path_dict[split] = []
        image_meta_path_dict[split] = []
        mask_path_dict[split] = []

        for num_str in sorted(tvt_splits[split]):
            for image_file_path in image_dir.glob(config['image_file_pattern']):
                if num_str in image_file_path.stem:
                    image_file_path_dict[split].append(image_file_path)
                    break

            for image_meta_path in image_dir.glob(config['image_meta_pattern']):
                if num_str in image_meta_path.stem:
                    image_meta_path_dict[split].append(image_meta_path)
                    break

            for mask_path in mask_dir.glob(config['mask_file_pattern']):
                if num_str in mask_path.stem:
                    mask_path_dict[split].append(mask_path)
                    break 
    
    if verbose:
        pprint(image_file_path_dict)
        pprint(image_meta_path_dict)
        pprint(mask_path_dict)

    return image_file_path_dict, image_meta_path_dict, mask_path_dict


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

def load_data(config):
    # Get image and mask paths
    image_file_path_dict, image_meta_path_dict, mask_path_dict = get_data_paths(config)

    labels_map = {
        0: 'Void',
        1: 'Ground & Water',
        2: 'Stem',
        3: 'Canopy',
        4: 'Roots',
        5: 'Objects'
    }

    # ---- Define Albumentations transforms ----
    train_transform = A.Compose([
                                    ChannelShuffleGroups(groups=[[0, 1, 2, 3, 4], [5, 6, 7]], p=0.5),
                                    A.Resize(height=512, width=512),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=30, p=0.5),
                                    # BrightnessContrastOnlyFirst3Channels(p=0.5), # # !!!!put it aside for now.
                                    # A.Normalize(mean=(0.485, 0.456, 0.406, 0.0, 0.0, 0.0), # !!!!put mean and std aside for now.
                                    #             std=(0.229, 0.224, 0.225, 1.0, 1.0, 1.0)),
                                    A.pytorch.ToTensorV2()
                                ],
                                additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
                                A.Resize(height=512, width=512),
                                # A.Normalize(
                                #     mean=(0.485, 0.456, 0.406, 0.0, 0.0, 0.0),
                                #     std=(0.229, 0.224, 0.225, 1.0, 1.0, 1.0)
                                # ),
                                ToTensorV2()
                            ], additional_targets={'mask': 'mask'})

    # ---- Create dataloaders for train, val, and test ----
    dataloaders = {}
    for split in ['train', 'val', 'test']:
        image_file_paths = image_file_path_dict[split]
        image_meta_paths = image_meta_path_dict[split]
        mask_paths = mask_path_dict[split]

        dataset = SegmentationPatchDataset(
            image_file_paths=image_file_paths,
            image_meta_paths=image_meta_paths,
            mask_paths=mask_paths,
            patch_splits=PATCH_PER_IMAGE,
            transform=train_transform if split == 'train' else val_transform,
            labels_map=labels_map
        )

        dataloaders[split] = DataLoader(dataset, batch_size=PATCH_PER_IMAGE, shuffle=True if split == 'train' else False, num_workers=2)

    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    train_loader, val_loader, test_loader = load_data(config)
    print(f"Training data: {len(train_loader.dataset)} samples")
    print(f"Validation data: {len(val_loader.dataset)} samples")
    print(f"Test data: {len(test_loader.dataset)} samples")
    print("Data loaders created successfully!")
    print("Sample batch from training loader:")
    for imgs, masks in train_loader:
        print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
        break
    print("Sample batch from validation loader:")
    for imgs, masks in val_loader:
        print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
        break
    print("Sample batch from test loader:")
    for imgs, masks in test_loader:
        print(f"Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
        break

