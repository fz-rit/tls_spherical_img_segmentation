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


class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_irr_paths, 
                 image_nxnynz_paths,
                 mask_paths, 
                 patch_splits=5, 
                 transform=None, 
                 mask_transform=None,
                 labels_map=None):
        self.image_irr_paths = image_irr_paths
        self.image_nxnynz_paths = image_nxnynz_paths
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.mask_transform = mask_transform
        self.labels_map = labels_map
        self.patch_coords = []  # Will store tuples: (img_index, x_start, x_end)

        # Open a sample image to determine dimensions
        sample_img = Image.open(self.image_irr_paths[0])
        w, h = sample_img.size
        patch_width = w // patch_splits
        
        # Create patch coordinates
        for i in range(len(image_irr_paths)):
            for p in range(patch_splits):
                x_start = p * patch_width
                x_end = x_start + patch_width
                self.patch_coords.append((i, x_start, x_end))
        
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, x_start, x_end = self.patch_coords[idx]

        # Load the IRZ and NxNyNz images
        img_irz = Image.open(self.image_irr_paths[i])         # e.g., Intensity-Range-Zvalue
        img_nxnynz = Image.open(self.image_nxnynz_paths[i])   # e.g., Nx-Ny-Nz

        # Load the mask
        mask = Image.open(self.mask_paths[i])

        # Convert them to NumPy arrays
        img_irz = np.array(img_irz)
        img_nxnynz = np.array(img_nxnynz)
        mask = np.array(mask)

        # Crop the IRZ and NxNyNz arrays
        top_crop, bottom_crop = 14, 526  # Example
        img_irz = img_irz[top_crop:bottom_crop, x_start:x_end, ...]
        img_nxnynz = img_nxnynz[top_crop:bottom_crop, x_start:x_end, ...]

        # Crop the mask to match
        mask = mask[top_crop:bottom_crop, x_start:x_end]

        # Concatenate IRZ + NxNyNz => (H, W, 6)
        # Make sure img_irz and img_nxnynz each have 3 channels
        # shape after concatenation: (height, width, 6)
        combined_img = np.concatenate([img_irz, img_nxnynz], axis=-1)

        # Apply Albumentations transform if provided
        if self.transform:
            transformed = self.transform(image=combined_img, mask=mask)
            combined_img = transformed["image"]  # shape: (6, H, W) after ToTensorV2
            mask = transformed["mask"]          # shape: (H, W) or (1, H, W)
            mask = mask.long()
        else:
            # If no transform, convert to torch.Tensor manually
            combined_img = torch.from_numpy(combined_img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return combined_img, mask



def get_image_maskt_paths(config: dict):


    root_dir = Path(config['root_dir'])
    image_dir = root_dir / Path(config['image_dir'])
    mask_dir = root_dir / Path(config['mask_dir'])

    image_irr_paths = sorted(image_dir.glob(config['image_irr_pattern'])) # Intensity-Range-Zvalue pseudo-color images
    image_nxnynz_paths = sorted(image_dir.glob(config['image_nxnynz_pattern'])) # nx-ny-nz HSV-converted-color images
    mask_paths = sorted(mask_dir.glob(config['mask_file_pattern']))

    return image_irr_paths, image_nxnynz_paths, mask_paths


class BrightnessContrastOnlyFirst3Channels(A.ImageOnlyTransform):
    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, p=0.5):
        super().__init__(p=p)
        self.brightness_limit = brightness_limit
        self.contrast_limit = contrast_limit
        self.bc_transform = A.RandomBrightnessContrast(
            brightness_limit=self.brightness_limit,
            contrast_limit=self.contrast_limit,
            p=1.0  # always apply inside
        )

    def apply(self, img, **params):
        # img shape [H, W, 6]
        # separate first 3 channels from geometry
        first3 = img[:, :, :3]
        last3 = img[:, :, 3:]

        # Albumentations needs an image with shape [H, W, C]
        # apply bc_transform to the first3 only
        first3_aug = self.bc_transform(image=first3)['image']
        # re-concat
        return np.concatenate([first3_aug, last3], axis=2)


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
    image_irr_paths, image_nxnynz_paths, mask_paths = get_image_maskt_paths(config)

    labels_map = {
        0: 'Void',
        1: 'Miscellaneous',
        2: 'Leaves',
        3: 'Bark',
        4: 'Soil'
    }

    # Split into training and validation
    train_image_irr_paths = image_irr_paths[:-1]
    train_image_nxnynz_paths = image_nxnynz_paths[:-1]
    train_mask_paths = mask_paths[:-1]

    val_image_irr_paths = [image_irr_paths[-1]]
    val_image_nxnynz_paths = [image_nxnynz_paths[-1]]
    val_mask_paths = [mask_paths[-1]]

    # ---- Define Albumentations transforms ----
    train_transform = A.Compose([
                                    ChannelShuffleGroups(groups=[[0,1,2], [3,4,5]], p=0.5),
                                    A.Resize(height=512, width=512),
                                    A.HorizontalFlip(p=0.5),
                                    A.VerticalFlip(p=0.5),
                                    A.RandomRotate90(p=0.5),
                                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=30, p=0.5),
                                    BrightnessContrastOnlyFirst3Channels(p=0.5),
                                    A.Normalize(mean=(0.485, 0.456, 0.406, 0.0, 0.0, 0.0),
                                                std=(0.229, 0.224, 0.225, 1.0, 1.0, 1.0)),
                                    A.pytorch.ToTensorV2()
                                ],
                                additional_targets={'mask': 'mask'})

    val_transform = A.Compose([
                                A.Resize(height=512, width=512),
                                A.Normalize(
                                    mean=(0.485, 0.456, 0.406, 0.0, 0.0, 0.0),
                                    std=(0.229, 0.224, 0.225, 1.0, 1.0, 1.0)
                                ),
                                ToTensorV2()
                            ], additional_targets={'mask': 'mask'})

    # ---- Create dataset and dataloader ----
    train_dataset = SegmentationPatchDataset(
        image_irr_paths=train_image_irr_paths,
        image_nxnynz_paths=train_image_nxnynz_paths,
        mask_paths=train_mask_paths,
        patch_splits=5,
        transform=train_transform,  # Albumentations for training
        labels_map=labels_map
    )

    val_dataset = SegmentationPatchDataset(
        image_irr_paths=val_image_irr_paths,
        image_nxnynz_paths=val_image_nxnynz_paths,
        mask_paths=val_mask_paths,
        patch_splits=5,
        transform=val_transform,  # Albumentations for validation
        labels_map=labels_map
    )

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=2)

    return train_dataset, train_loader, val_loader
