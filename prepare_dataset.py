import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import json
from pathlib import Path

class SegmentationPatchDataset(Dataset):
    def __init__(self, image_paths, mask_paths, patch_splits=5, transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_splits = patch_splits
        self.transform = transform
        self.mask_transform = mask_transform
        self.patch_coords = []  # Will store tuples: (img_index, x_start, x_end)

        # Open a sample image to determine dimensions
        sample_img = Image.open(self.image_paths[0])
        w, h = sample_img.size
        patch_width = w // patch_splits
        
        # We assume height = 540 initially, we will crop to 512
        # Make sure h >= 512 before cropping
        if h < 512:
            raise ValueError("Image height is smaller than 512, cannot crop down.")

        for i in range(len(image_paths)):
            for p in range(patch_splits):
                x_start = p * patch_width
                x_end = x_start + patch_width
                self.patch_coords.append((i, x_start, x_end))
        
    def __len__(self):
        return len(self.patch_coords)
    
    def __getitem__(self, idx):
        i, x_start, x_end = self.patch_coords[idx]

        # Load image and mask
        img = Image.open(self.image_paths[i]).convert('RGB')
        mask = Image.open(self.mask_paths[i])

        # Convert mask if needed: 
        # If your mask is already single-channel with proper class indices, skip this step.
        # If you need to convert from RGB mask to class indices, do so here.

        # Crop the image to the top-left (0,0) and bottom-right (x_end, 512)
        img = img.crop((x_start, 0, x_end, 512))

        # Convert mask to np array and crop
        mask = np.array(mask)
        mask = mask[:512, x_start:x_end]  # Crop mask array to match image patch

        # Apply transforms
        if self.transform:
            img = self.transform(img)  # Usually transforms.ToTensor() + normalization

        else:
            # Convert PIL image to tensor if no transform provided
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        if self.mask_transform:
            mask = self.mask_transform(mask)
        else:
            mask = torch.from_numpy(mask).long()

        return img, mask
    

def get_image_maskt_paths(config_file):
    # Load config file
    with open(config_file) as f:
        config = json.load(f)

    root_dir = Path(config['root_dir'])
    image_dir = root_dir / Path(config['image_dir'])
    mask_dir = root_dir / Path(config['mask_dir'])

    image_paths = sorted(image_dir.glob(config['image_file_pattern']))
    mask_paths = sorted(mask_dir.glob(config['mask_file_pattern']))

    return image_paths, mask_paths

def load_data(config_file):
    # Get image and mask paths
    image_paths, mask_paths = get_image_maskt_paths(config_file)

    # Split into training and validation
    train_image_paths = image_paths[:-1]
    train_mask_paths  = mask_paths[:-1]

    val_image_paths = [image_paths[-1]]
    val_mask_paths  = [mask_paths[-1]]

    # Define transforms
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Create dataset and dataloader
    train_dataset = SegmentationPatchDataset(
        image_paths=train_image_paths,
        mask_paths=train_mask_paths,
        patch_splits=5,
        transform=img_transform
    )

    val_dataset = SegmentationPatchDataset(
        image_paths=val_image_paths,
        mask_paths=val_mask_paths,
        patch_splits=5,
        transform=img_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=2)

    return train_dataset, train_loader, val_loader