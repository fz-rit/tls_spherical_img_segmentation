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
from typing import Tuple, Dict, Any

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
    # print(f"Image cube loaded from {image_cube_path}, shape: {image_cube.shape}")
    
    # Load the metadata
    metadata = np.load(metadata_path, allow_pickle=True).item()  # use .item() to load it as a dictionary
    # print(f"Metadata loaded from {metadata_path}")

    return {
        'image_cube': image_cube,
        'metadata': metadata
    }


class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_file_paths, 
                 image_meta_paths,
                 mask_paths, 
                 patch_splits=5, 
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
        self.patch_coords = []  # Will store tuples: (img_index, x_start, x_end)

        # Open a sample image to determine dimensions
        data = load_image_cube_and_metadata(self.image_file_paths[0], self.image_meta_paths[0])
        image_cube = data['image_cube']
        metadata = data['metadata']
        w = image_cube.shape[1]
        patch_width = w // patch_splits
        
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

        data = load_image_cube_and_metadata(self.image_file_paths[i], self.image_meta_paths[i])
        image_cube = data['image_cube']
        metadata = data['metadata']
        mask = np.array(Image.open(self.mask_paths[i]))

        top_crop, bottom_crop = 14, 526  # cut off the top and bottom 14 pixels
        combined_img = image_cube[top_crop:bottom_crop, x_start:x_end, ...]

        # Crop the mask to match
        mask = mask[top_crop:bottom_crop, x_start:x_end]

        # Apply Albumentations transform if provided
        if self.transform:
            transformed = self.transform(image=combined_img, mask=mask)
            combined_img = transformed["image"]  # shape: (C, H, W) after ToTensorV2
            mask = transformed["mask"]          # shape: (H, W) or (1, H, W)
            mask = mask.long()
        else:
            # If no transform, convert to torch.Tensor manually
            combined_img = torch.from_numpy(combined_img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return combined_img, mask



def get_image_maskt_paths(config: dict) -> Tuple[list, list, list]:
    """
    Get image and mask paths from the config dictionary.

    Args:
    config (dict): Configuration dictionary.
    """
    root_dir = Path(config['root_dir'])
    image_dir = root_dir / Path(config['image_dir'])
    mask_dir = root_dir / Path(config['mask_dir'])

    image_file_paths = sorted(image_dir.glob(config['image_file_pattern']))
    image_meta_paths = sorted(image_dir.glob(config['image_meta_pattern']))
    mask_paths = sorted(mask_dir.glob(config['mask_file_pattern']))

    return image_file_paths, image_meta_paths, mask_paths


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
    image_file_paths, image_meta_paths, mask_paths = get_image_maskt_paths(config)

    labels_map = {
        0: 'Void',
        1: 'Ground & Water',
        2: 'Stem',
        3: 'Canopy',
        4: 'Roots',
        5: 'Objects'
    }

    # Split into training and validation
    train_image_file_paths = image_file_paths[:2]
    train_image_meta_paths = image_meta_paths[:2]
    train_mask_paths = mask_paths[:2]

    val_image_file_paths = [image_file_paths[2]]
    val_image_meta_paths = [image_meta_paths[2]]
    val_mask_paths = [mask_paths[2]]

    test_image_file_paths = [image_file_paths[3]]
    test_image_meta_paths = [image_meta_paths[3]]
    test_mask_paths = [mask_paths[3]]

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

    # ---- Create dataset and dataloader ----
    train_dataset = SegmentationPatchDataset(
        image_file_paths=train_image_file_paths,
        image_meta_paths=train_image_meta_paths,
        mask_paths=train_mask_paths,
        patch_splits=5,
        transform=train_transform,  # Albumentations for training
        labels_map=labels_map
    )

    val_dataset = SegmentationPatchDataset(
        image_file_paths=val_image_file_paths,
        image_meta_paths=val_image_meta_paths,
        mask_paths=val_mask_paths,
        patch_splits=5,
        transform=val_transform,  
        labels_map=labels_map
    )

    test_dataset = SegmentationPatchDataset(
        image_file_paths=test_image_file_paths,
        image_meta_paths=test_image_meta_paths,
        mask_paths=test_mask_paths,
        patch_splits=5,
        transform=val_transform,  
        labels_map=labels_map
    )



    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
