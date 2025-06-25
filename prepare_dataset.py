import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import json
import yaml
from pprint import pprint
from tools.get_mean_std import normalize_img_per_channel
from sklearn.model_selection import train_test_split

def collect_image_mask_pairs(img_dir: Path, mask_dir: Path) -> Tuple[List[Path], List[Path]]:
    img_paths = sorted(img_dir.glob("*.npy"))
    mask_paths = sorted(mask_dir.glob("*.png"))
    assert len(img_paths) == len(mask_paths), "Mismatch between images and masks"
    assert len(img_paths) > 0, f"No images found in {img_dir}"
    assert len(mask_paths) > 0, f"No masks found in {mask_dir}"
    for img_path, mask_path in zip(img_paths, mask_paths):
        match = img_path.stem.split("_image_cube")[0][-4:] == mask_path.stem.split("_segmk")[0][-4:]
        assert match, f"Image {img_path.name} does not match mask {mask_path.name}"
    return img_paths, mask_paths


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
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    elif arr.ndim == 3:
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
                 patch_splits=None, 
                 transform=None, 
                 ):
        self.image_file_paths = image_file_paths
        self.input_channels = input_channels
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
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
        out = img.copy()

        for group in self.groups:
            shuffled = np.random.permutation(group)

            for old_ch, new_ch in zip(group, shuffled):
                out[:, :, old_ch] = img[:, :, new_ch]

        return out


def trasform_by_channls(input_channels:list, p=0.5):
    """
    
    p: probability of applying the transform
    """
    if len(input_channels) == 3:
        shuffle_groups = [[0, 1, 2]]
    elif len(input_channels) == 6:
        shuffle_groups = [[0, 1, 2], [3, 4, 5]]
    elif len(input_channels) == 9:
        shuffle_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    elif len(input_channels) == 15:
        shuffle_groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], 
                          [9, 10, 11], [12, 13, 14]]

    transform = A.Compose([
        ChannelShuffleGroups(groups=shuffle_groups, p=p),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.0625, scale=(1 - 0.15, 1 + 0.15), rotate=(-10, 10), p=0.5),
        A.pytorch.ToTensorV2()
    ],
    additional_targets={'mask': 'mask'})

    return transform



def load_data(config, input_channels=None) -> Tuple[DataLoader, DataLoader, DataLoader]:

    root_dir = Path(config["root_dir"])
    num_workers = config['num_workers']
    patches_per_image = config['patches_per_image']
    batch_size = config['train_batch_size']

    train_transform = trasform_by_channls(input_channels=input_channels)
    val_transform = A.Compose([ToTensorV2()], additional_targets={'mask': 'mask'})

    trainval_img_paths, trainval_mask_paths = collect_image_mask_pairs(
        root_dir / "train_val/img_cube", root_dir / "train_val/mask"
    )


    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        trainval_img_paths,
        trainval_mask_paths,
        test_size=config.get("val_ratio", 0.1),
        shuffle=True,
        random_state=42
    )


    train_dataset = SegmentationPatchDataset(
        train_img_paths, train_mask_paths, input_channels, patches_per_image, train_transform
    )
    val_dataset = SegmentationPatchDataset(
        val_img_paths, val_mask_paths, input_channels, patches_per_image, val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # --- Test loader ---
    test_img_paths, test_mask_paths = collect_image_mask_pairs(
        root_dir / "test/img_cube", root_dir / "test/mask"
    )
    test_dataset = SegmentationPatchDataset(
        test_img_paths, test_mask_paths, input_channels, patches_per_image, val_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=patches_per_image, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    config_file = 'params/paths_zmachine_mangrove3d.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
        
    input_channels = config['input_channels_ls'][0]

    train_loader, val_loader, test_loader = load_data(config, input_channels=input_channels)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Data loaders created successfully!")
    
    imgs, masks = next(iter(train_loader))
    print(f"training loader Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")

    imgs, masks = next(iter(val_loader))
    print(f"validation loader Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")

    imgs, masks = next(iter(test_loader))
    print(f"test loader Image batch shape: {imgs.shape}, Mask batch shape: {masks.shape}")
    
    