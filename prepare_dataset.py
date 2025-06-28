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
from sklearn.utils import resample


def subset_dataset_by_count(img_paths, mask_paths, count, seed=42):
    assert count <= len(img_paths), f"Requested {count} samples, but only {len(img_paths)} available"
    img_sub, mask_sub = resample(img_paths, mask_paths, n_samples=count, random_state=seed, replace=False)
    return img_sub, mask_sub


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

def pad_img_or_mask_vertical_only(arr):
    """
    Pad only the vertical dimension (height) to make it divisible by 32.
    """
    h, w = arr.shape[:2]
    pad_h = (32 - (h % 32)) % 32
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if arr.ndim == 2:
        return np.pad(arr, ((pad_top, pad_bottom), (0, 0)), mode='constant').astype(np.float32)
    elif arr.ndim == 3:
        return np.pad(arr, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant').astype(np.float32)
    else:
        raise ValueError(f"Input array must be 2D or 3D, got shape {arr.shape}")


def pad_img_and_mask_vertical_only(image, mask):
    return pad_img_or_mask_vertical_only(image), pad_img_or_mask_vertical_only(mask)

def depad_tensor_vertical_only(tensor: torch.Tensor, original_height: int) -> torch.Tensor:
    """
    Remove vertical padding from a tensor (e.g., image, mask, or logits).

    Args:
        tensor: Tensor of shape [H, W], [C, H, W], or [B, C, H, W]
        original_height: Height before vertical padding

    Returns:
        Tensor cropped back to original height.
    """
    if tensor.dim() == 2:
        # [H, W]
        H = tensor.shape[0]
        pad_total = H - original_height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return tensor[pad_top:H - pad_bottom, :]

    elif tensor.dim() == 3:
        # [C, H, W]
        H = tensor.shape[1]
        pad_total = H - original_height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return tensor[:, pad_top:H - pad_bottom, :]

    elif tensor.dim() == 4:
        # [B, C, H, W]
        H = tensor.shape[2]
        pad_total = H - original_height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return tensor[:, :, pad_top:H - pad_bottom, :]

    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")



class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_file_paths, 
                 mask_paths=None,
                 input_channels=None,
                 patch_splits=None, 
                 transform=None,
                 buffer_size: int = 0,  # <--- NEW
                 ):
        self.image_file_paths = image_file_paths
        self.input_channels = input_channels
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.buffer_size = buffer_size
        self.patch_idx_ls = [(i, p) for i in range(len(self.image_file_paths)) for p in range(patch_splits)]

    def __len__(self):
        return len(self.patch_idx_ls)
    
    def __getitem__(self, idx):
        i, p = self.patch_idx_ls[idx]

        # Load image and mask
        image_cube = np.load(self.image_file_paths[i])
        image_cube = image_cube[:, :, self.input_channels]
        image_cube = normalize_img_per_channel(image_cube)
        mask = np.array(Image.open(self.mask_paths[i]))

        full_h, full_w = image_cube.shape[:2]
        tile_w = full_w // self.patch_splits
        in_size = tile_w + 2 * self.buffer_size

        x_center = p * tile_w + tile_w // 2
        x_start = x_center - in_size // 2
        x_end = x_start + in_size

        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - full_w)
        crop_l = max(0, x_start)
        crop_r = min(full_w, x_end)

        img_patch = image_cube[:, crop_l:crop_r, :]
        mask_patch = mask[:, crop_l:crop_r]

        if pad_left or pad_right:
            img_patch = np.pad(img_patch, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
            mask_patch = np.pad(mask_patch, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)

        img_patch, mask_patch = pad_img_and_mask_vertical_only(img_patch, mask_patch)

        # Construct buffer mask
        H, W = mask_patch.shape
        buffer_mask = np.zeros((H, W), dtype=np.uint8)
        buffer_mask[:, self.buffer_size:self.buffer_size + tile_w] = 1

        # Apply transforms
        if self.transform:
            tr = self.transform(image=img_patch, mask=mask_patch)
            img_patch = tr["image"]
            mask_patch = tr["mask"].long()
        else:
            img_patch = torch.from_numpy(img_patch).permute(2, 0, 1).float() / 255.
            mask_patch = torch.from_numpy(mask_patch).long()

        buffer_mask = torch.from_numpy(buffer_mask).unsqueeze(0).float()
        return img_patch, mask_patch, buffer_mask


def choose_buffer_size(tile_size, multiple_of=32):
    for buf in range(32, 65, 4):  # reasonable range
        if (tile_size + 2 * buf) % multiple_of == 0:
            return buf
    raise ValueError("No suitable buffer size found to make input divisible by {}".format(multiple_of))




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
    if len(input_channels) % 3 != 0:
        raise ValueError(f"Number of input channels must be divisible by 3, got {len(input_channels)}")
    
    shuffle_groups = [[i, i+1, i+2] for i in range(0, len(input_channels), 3)]

    transform = A.Compose([
        ChannelShuffleGroups(groups=shuffle_groups, p=p),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=0.0625, scale=(1 - 0.15, 1 + 0.15), rotate=(-10, 10), p=0.5),
        A.pytorch.ToTensorV2()
    ],
    additional_targets={'mask': 'mask'})

    return transform



def load_data(config, input_channels=None, train_subset_cnt=30) -> Tuple[DataLoader, DataLoader, DataLoader]:

    root_dir = Path(config["root_dir"])
    num_workers = config['num_workers']
    patches_per_image = config['patches_per_image']
    batch_size = config['train_batch_size']
    input_size = config['input_size']

    train_transform = trasform_by_channls(input_channels=input_channels)
    val_transform = A.Compose([ToTensorV2()], additional_targets={'mask': 'mask'})

    trainval_img_paths, trainval_mask_paths = collect_image_mask_pairs(
        root_dir / "train_val/img_cube", root_dir / "train_val/mask"
    )

    if train_subset_cnt < len(trainval_img_paths):
        print(f"Subsetting dataset to {train_subset_cnt} samples")
        trainval_img_paths, trainval_mask_paths = subset_dataset_by_count(
            trainval_img_paths, trainval_mask_paths, train_subset_cnt
        )

    train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = train_test_split(
        trainval_img_paths,
        trainval_mask_paths,
        test_size=config.get("val_ratio", 0.1),
        shuffle=True,
        random_state=42
    )
    tile_w = input_size[1] // patches_per_image  
    buffer_size = choose_buffer_size(tile_w, multiple_of=32)
    print(f"Using adaptive buffer size horizontally: {buffer_size}")
    train_dataset = SegmentationPatchDataset(
        train_img_paths, train_mask_paths, input_channels, patches_per_image, train_transform, buffer_size=buffer_size
    )
    val_dataset = SegmentationPatchDataset(
        val_img_paths, val_mask_paths, input_channels, patches_per_image, val_transform, buffer_size=buffer_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    test_img_paths, test_mask_paths = collect_image_mask_pairs(
        root_dir / "test/img_cube", root_dir / "test/mask"
    )
    test_dataset = SegmentationPatchDataset(
        test_img_paths, test_mask_paths, input_channels, patches_per_image, val_transform, buffer_size=buffer_size
    )
    test_loader = DataLoader(test_dataset, batch_size=patches_per_image, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    from tools.load_tools import CONFIG
    input_channels = CONFIG['input_channels_ls'][0]
    train_subset_cnt = CONFIG.get('train_subset_cnt', 30)
    train_loader, val_loader, test_loader = load_data(CONFIG, input_channels=input_channels, train_subset_cnt=train_subset_cnt)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print("Data loaders created successfully!")
    
    imgs, masks, buffer_masks = next(iter(train_loader))
    print(f"training loader Image batch shape: {imgs.shape}, \nMask batch shape: {masks.shape}, \nBuffer Mask batch shape: {buffer_masks.shape}")

    imgs, masks, buffer_masks = next(iter(val_loader))
    print(f"validation loader Image batch shape: {imgs.shape}, \nMask batch shape: {masks.shape}, \nBuffer Mask batch shape: {buffer_masks.shape}")

    imgs, masks, buffer_masks = next(iter(test_loader))
    print(f"test loader Image batch shape: {imgs.shape}, \nMask batch shape: {masks.shape}, \nBuffer Mask batch shape: {buffer_masks.shape}")
    
    