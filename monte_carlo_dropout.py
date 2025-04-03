import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import torch.nn.functional as F
from prepare_dataset import resize_image_or_mask
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder


class DecoderBlockWithDropout(nn.Module):
    def __init__(self, decoder_block, dropout_p=0.3):
        super().__init__()
        self.block = decoder_block
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x, skip=None):
        x = self.block(x, skip)
    # def forward(self, x):
    #     x = self.block(x)
        x = self.dropout(x)
        return x
    

class SegBlockWithDropout(nn.Module): # For FPNDecoder
    def __init__(self, seg_block: nn.Module, dropout_p: float):
        super().__init__()
        self.seg_block = seg_block
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.seg_block(x)
        x = self.dropout(x)
        return x


def add_dropout_to_decoder(model, p=0.3):
    """
    Wrap decoder blocks with dropout. Supports UNet (ModuleList)
    and UNet++ (nested ModuleDict of lists).
    """
    decoder = model.decoder

    # Case 1: UNet++ — decoder.blocks is a ModuleDict of lists
    if hasattr(decoder, "blocks") and isinstance(decoder.blocks, nn.ModuleDict):
        for key in decoder.blocks:
            decoder.blocks[key] = DecoderBlockWithDropout(decoder.blocks[key], dropout_p=p)


    # Case 2: UNet — decoder.blocks is a ModuleList
    elif hasattr(decoder, "blocks") and isinstance(decoder.blocks, nn.ModuleList):
        for i in range(len(decoder.blocks)):
            decoder.blocks[i] = DecoderBlockWithDropout(decoder.blocks[i], dropout_p=p)


    # Case 3: FPN - decoder.blocks is an instance of FPNDecoder
    # Note: FPNDecoder also has a dropout layer, but we can add more dropout
    # to the seg_blocks if needed.
    # ref: https://github.com/qubvel-org/segmentation_models.pytorch/blob/main/segmentation_models_pytorch/decoders/fpn/decoder.py
    elif isinstance(decoder, FPNDecoder):
        for i in range(len(decoder.seg_blocks)):
            block = decoder.seg_blocks[i]
            decoder.seg_blocks[i] = SegBlockWithDropout(block, dropout_p=p)
        # decoder.dropout = nn.Dropout2d(p=p)

    else:
        raise TypeError(f"Unknown decoder structure: {type(decoder)}")



class MonteCarloDropoutUncertainty(nn.Module):
    def __init__(self, model, imgs):
        super().__init__()
        self.model = model
        self.imgs = imgs
        self.num_classes = 6

    def perform_mc_dropout_inference(self, mc_iterations=20):

        self.model.train()  # Keep dropout ON during inference
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        preds = []

        with torch.no_grad():
            for _ in range(mc_iterations):
                pred = self.model(self.imgs) # (N, C, H, W), N - batch size, 5; C - number of classes; W - width of image patch, 360.
                pred_masks = torch.softmax(pred, dim=1)  # (N, C, H, W)
                
                pred_masks_resized = F.interpolate(pred_masks, 
                                                   size=(512, 360), # Resize width from 512 → 360, keep height at 512
                                                   mode='bilinear', 
                                                   align_corners=False)
                combined_pred_mask = torch.cat([mask for mask in pred_masks_resized], dim=2)  # shape: (C, H, W * N)
                preds.append(combined_pred_mask.unsqueeze(0))  # shape: [1, C, H, W * N]

        self.preds = torch.stack(preds) # [mc_iterations, 1, C, H, WN]


    def predictive_entropy(self, mc_preds):
        mean_preds = mc_preds.mean(dim=0)  # average predictions across samples
        entropy = -torch.sum(mean_preds * torch.log(mean_preds + 1e-8), dim=1) # Max entropy is log(num_classes)
        entropy_normalized = (entropy - entropy.min()) / (entropy.max() - entropy.min())
        return entropy_normalized # [1, H, WN]; range 0-1

    def mutual_information(self, mc_preds):
        mean_preds = mc_preds.mean(dim=0)  # shape: [1, num_classes, H, W]
        
        # entropy of averaged predictions
        entropy_mean = -torch.sum(mean_preds * torch.log(mean_preds + 1e-8), dim=1)

        # average entropy across samples
        entropy_each_sample = -torch.sum(mc_preds * torch.log(mc_preds + 1e-8), dim=2)
        mean_entropy = entropy_each_sample.mean(dim=0)

        # mutual information (epistemic uncertainty)
        mi = entropy_mean - mean_entropy # Max MI is log(num_classes)
        mi_normalized = (mi - mi.min()) / (mi.max() - mi.min())
        return mi_normalized  # [1, H, WN]; range 0-1


    def estimate_epistemic_uncertainty(self, mutual_information=False):
        """
        Estimate the epistemic uncertainty using the predictive entropy or mutual information.
        """
        if mutual_information:
            ep_uncertainty = self.mutual_information(self.preds)
        else:
            ep_uncertainty = self.predictive_entropy(self.preds)

        self.uncertainty_map = ep_uncertainty.squeeze().cpu().numpy() # [H, WN]
        self.uncertainty_map = resize_image_or_mask(self.uncertainty_map, (540, 1440))


    def binarize_uncertainty_map(self):

        self.thresholds = np.linspace(0.2, 0.5, 4)
        self.binarized_uncertainty_maps = []
        for threshold in self.thresholds:
            binarized_map = np.zeros_like(self.uncertainty_map)
            binarized_map[self.uncertainty_map >= threshold] = 1
            self.binarized_uncertainty_maps.append(binarized_map)

    def plot_uncertainty_map(self, output_path):
        """
        Plot the uncertainty map along with the binarized uncertainty maps. 
        Stack them vertically in subplots.
        
        """
        subplot_count = 1 + len(self.binarized_uncertainty_maps)
        fig, axs = plt.subplots(subplot_count, 1, figsize=(10, 4*subplot_count))
        unctt_map = axs[0].imshow(self.uncertainty_map, cmap='inferno')  # 'hot', 'jet', 'viridis', 'inferno', 'magma', 'cividis'
        axs[0].set_title("Epistemic Uncertainty Map (normalized)")


        for i, binarized_map in enumerate(self.binarized_uncertainty_maps):
            axs[i+1].imshow(binarized_map, cmap='gray')
            axs[i+1].set_title(f"Binarized Uncertainty Map (Threshold: {self.thresholds[i]:.2f})")
            axs[i+1].axis('off')

        fig.colorbar(unctt_map, ax=axs[0], orientation='horizontal', fraction=0.05, pad=0.05)
        fig.savefig(output_path)
        print(f"🌻Uncertainty map saved to {output_path}.")
        # plt.show()

        # Save each image separately
        titles = [f'Binarized Uncertainty Map_Threshold_{t:.2f}' for t in self.thresholds]
        subplot_save_dir = output_path.parent / f'uncertainty_maps_{str(output_path.stem).split("_map_")[1]}'
        subplot_save_dir.mkdir(exist_ok=True)
        unctt_map_img = Image.fromarray((self.uncertainty_map * 255).astype(np.uint8))
        output_path = subplot_save_dir / 'uncertainty_map.png'
        unctt_map_img.save(output_path)
        
        for i, binarized_map in enumerate(self.binarized_uncertainty_maps):
            output_path = subplot_save_dir / f'{titles[i]}.png'
            binarized_map_img = Image.fromarray((binarized_map * 255).astype(np.uint8))   
            binarized_map_img.save(output_path)
        print(f"🍀Individual maps saved to {subplot_save_dir}.")


    def execute(self, mc_iterations=20, 
                mutual_information=True, 
                output_path='outputs/uncertainty_map.png'):
        self.perform_mc_dropout_inference(mc_iterations)
        self.estimate_epistemic_uncertainty(mutual_information)
        self.binarize_uncertainty_map()
        self.plot_uncertainty_map(output_path)