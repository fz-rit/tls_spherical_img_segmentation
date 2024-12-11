import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

class SegmentationPatchDataset(Dataset):
    def __init__(self, 
                 image_irz_paths, 
                 image_nxnynz_paths,
                 mask_paths, 
                 patch_splits=5, 
                 transform=None, 
                 mask_transform=None,
                 labels_map=None):
        self.image_irz_paths = image_irz_paths
        self.image_nxnynz_paths = image_nxnynz_paths
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.mask_transform = mask_transform
        self.labels_map = labels_map
        self.patch_coords = []  # Will store tuples: (img_index, x_start, x_end)

        # Open a sample image to determine dimensions
        sample_img = Image.open(self.image_irz_paths[0])
        w, h = sample_img.size
        patch_width = w // patch_splits
        
        # Create patch coordinates
        for i in range(len(image_irz_paths)):
            for p in range(patch_splits):
                x_start = p * patch_width
                x_end = x_start + patch_width
                self.patch_coords.append((i, x_start, x_end))
        
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, x_start, x_end = self.patch_coords[idx]

        # Load image and mask
        img_irz = Image.open(self.image_irz_paths[i])
        img_nxnynz = Image.open(self.image_nxnynz_paths[i])
        mask = Image.open(self.mask_paths[i])

        # Convert mask if needed: 
        # Placeholder for converting from RGB mask to class indices, do so here.

        # Crop the images to make them divisible by 32
        top_crop = 14
        bottom_crop = 526
        img_irz = img_irz.crop((x_start, top_crop, x_end, bottom_crop))
        img_nxnynz = img_nxnynz.crop((x_start, top_crop, x_end, bottom_crop))

        # Convert mask to np array and crop
        mask = np.array(mask)
        mask = mask[top_crop:bottom_crop, x_start:x_end]  # Crop mask array to match image patch

        # Apply transforms to images
        if self.transform:
            img_irz = self.transform(img_irz)  # Apply transforms to img_irz
            img_nxnynz = self.transform(img_nxnynz)  # Apply transforms to img_nxnynz
        else:
            # Convert PIL image to tensor if no transform provided
            img_irz = torch.from_numpy(np.array(img_irz)).permute(2, 0, 1).float() / 255.0
            img_nxnynz = torch.from_numpy(np.array(img_nxnynz)).permute(2, 0, 1).float() / 255.0

        # Stack the images along the channel dimension
        img = torch.cat((img_irz, img_nxnynz), dim=0)

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return img, mask
    

def get_image_maskt_paths(config: dict):


    root_dir = Path(config['root_dir'])
    image_dir = root_dir / Path(config['image_dir'])
    mask_dir = root_dir / Path(config['mask_dir'])

    image_irz_paths = sorted(image_dir.glob(config['image_irz_pattern'])) # Intensity-Range-Zvalue pseudo-color images
    image_nxnynz_paths = sorted(image_dir.glob(config['image_nxnynz_pattern'])) # nx-ny-nz HSV-converted-color images
    mask_paths = sorted(mask_dir.glob(config['mask_file_pattern']))

    return image_irz_paths, image_nxnynz_paths, mask_paths

def load_data(config):
    # Get image and mask paths
    image_irz_paths, image_nxnynz_paths, mask_paths = get_image_maskt_paths(config)

    labels_map = {
                    0: 'Void',
                    1: 'Miscellaneous',
                    2: 'Leaves',
                    3: 'Bark',
                    4: 'Soil'
                }
    # Split into training and validation
    train_image_irz_paths = image_irz_paths[:-1]
    train_image_nxnynz_paths = image_nxnynz_paths[:-1]
    train_mask_paths  = mask_paths[:-1]

    val_image_irz_paths = [image_irz_paths[-1]]
    val_image_nxnynz_paths = [image_nxnynz_paths[-1]]
    val_mask_paths  = [mask_paths[-1]]

    # Define transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create dataset and dataloader
    train_dataset = SegmentationPatchDataset(
        image_irz_paths=train_image_irz_paths,
        image_nxnynz_paths=train_image_nxnynz_paths,
        mask_paths=train_mask_paths,
        patch_splits=5,
        transform=img_transform,
        labels_map=labels_map
    )

    val_dataset = SegmentationPatchDataset(
        image_irz_paths=val_image_irz_paths,
        image_nxnynz_paths=val_image_nxnynz_paths,
        mask_paths=val_mask_paths,
        patch_splits=5,
        transform=img_transform,
        labels_map=labels_map
    )

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=2)

    return train_dataset, train_loader, val_loader