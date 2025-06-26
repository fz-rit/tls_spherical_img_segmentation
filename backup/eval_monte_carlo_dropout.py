import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, resize_image_or_mask
from backup.training import train_unet, create_unet_multi_channels
from tools.load_tools import calc_metrics, get_color_map, get_pil_palette
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from monte_carlo_dropout import MonteCarloDropoutUncertainty
from PIL import Image
import numpy as np

def load_model(config: dict, device: str) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    
    model_dir = Path(config['root_dir']) / config['model_dir']
    model_file = model_dir / config['model_file']
    # Load the model if there is a saved model, otherwise train a new model
    if model_file.exists():
        model = create_unet_multi_channels()
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"======Loaded model from disk: {model_file.stem}.======")
    else:
        model, _, _  = train_unet(config)
        print("####Trained a new model.####")
    
    model = model.to(device)

    return model


def save_mask_as_image(mask: np.ndarray, mono_path: Path, color_path: Path):
    """
    Save the mask as a monochrome and color image.

    Args:
    mask (np.ndarray): Mask array.
    mono_path (Path): Path to save the monochrome image.
    color_path (Path): Path to save the color image.
    """
    mask = mask.astype(np.uint8)
    mask_mono = Image.fromarray(mask)
    mask_mono.save(mono_path)
    print(f"📸Monochrome mask saved to {mono_path}")

    mask_color = Image.fromarray(mask)
    mask_color.putpalette(get_pil_palette())
    mask_color.save(color_path)
    print(f"🎨Color mask saved to {color_path}")


def visualize_eval_output(imgs, true_masks, pred_masks, output_path: Path = None):
    """"
    Visualize the image and masks.
    """
    N_CLASSES = 6
    num_samples = len(imgs)  # Let's visualize up to 5 samples
    fig, axs = plt.subplots(3, num_samples, figsize=(10, 6*num_samples))

    for i in range(num_samples):
        # Original image: convert from tensor [C,H,W] to [H,W,C] and un-normalize if needed
        img = imgs[i].permute(1, 2, 0).numpy()  
        true_mask = true_masks[i].numpy()
        pred_mask = pred_masks[i].numpy()

        img = resize_image_or_mask(img, (540, 1440)) 
        true_mask = resize_image_or_mask(true_mask, (540, 1440))
        pred_mask = resize_image_or_mask(pred_mask, (540, 1440))

        # Compute metrics between true_mask and pred_mask
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()
        
        metric_dict = calc_metrics(true_flat, pred_flat, N_CLASSES)
        
        oAccu, mAccu, mIoU, FWIoU, dice_coefficient = metric_dict['oAccu'], metric_dict['mAccu'], metric_dict['mIoU'], metric_dict['FWIoU'], metric_dict['dice_coefficient']
        if num_samples == 1:
            axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]
            pred_title = ' '.join(['Predicted Mask:',
                        f'oAccu: {oAccu:.4f};',
                        f'mAccu: {mAccu:.4f};',
                        f'mIoU: {mIoU:.4f};',
                        f'FWIoU: {FWIoU:.4f};',
                        f'dice_coeff: {dice_coefficient:.4f}'])
        else:
            pred_title = f'Predicted Mask'
            axs_img, axs_true, axs_pred = axs[0, i], axs[1, i], axs[2, i]

        display_channels = [4, 0, 2] # Roughness, Intensity, Range
        axs_img.imshow(img[:, :, display_channels])
        axs_img.set_title('Original Image')
        axs_img.axis('off')

        # For masks, use a discrete colormap to distinguish classes
        axs_true.imshow(true_mask, cmap=get_color_map(), interpolation='nearest')
        axs_true.set_title('Ground Truth Mask')
        axs_true.axis('off')

        axs_pred.imshow(pred_mask, cmap=get_color_map(), interpolation='nearest')
        axs_pred.set_title(pred_title)
        axs_pred.axis('off')

    plt.tight_layout()
    plt.show()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"outputs/output_{timestamp}.png") if output_path is None else output_path
    fig.savefig(output_path)
    print(f"㊗️Segmentation map saved to {output_path}")

    # Save the pred_mask in rbg image.
    pred_mask_mono_path = output_path.parent / f"pred_mask_mono_{timestamp}.png"
    pred_mask_color_path = output_path.parent / f"pred_mask_color_{timestamp}.png"
    save_mask_as_image(pred_mask, pred_mask_mono_path, pred_mask_color_path)



def evaluate(imgs, true_masks, config, 
             device:str = 'cuda', 
             output_path_1: Path = None, 
             output_path_2: Path = None,
             output_path_3: Path = None):
    """
    Evaluate the model on the test set.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to run the evaluation on.
    """
    model = load_model(config, device)

    imgs = imgs.to(device)                  # (N, C, H, W)
    true_masks = true_masks.to(device)      # (N, H, W)

    

    # ---------Evaluate model in Monte Carlo Dropout mode and estimate uncertainty.------
    print("🔮Evaluating the model in Bayesian mode...")
    mcdu = MonteCarloDropoutUncertainty(model, imgs)
    mcdu.execute(mc_iterations=40, 
                mutual_information=True, 
                output_path=output_path_3)

    # -----Evaluate model in normal mode.--------------
    print("🙂Evaluating the model in normal mode...")
    model.eval()
    with torch.no_grad():
        preds = model(imgs)                 # (N, C, H, W)
        pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

    imgs = imgs.cpu()
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()
    
    # visualize_eval_output(imgs, true_masks, pred_masks, output_path_1) # Visualize the outputs by patch.

    # Images: (N, 3, H, W) -> concatenate along width
    combined_img = torch.cat([img for img in imgs], dim=2)  # shape: (3, H, W * N)
    # Masks: (N, H, W) -> concatenate along width
    combined_true_mask = torch.cat([mask for mask in true_masks], dim=1)  # shape: (H, W * N)
    combined_pred_mask = torch.cat([mask for mask in pred_masks], dim=1)  # shape: (H, W * N)
    
    visualize_eval_output([combined_img], 
                          [combined_true_mask], 
                          [combined_pred_mask],
                          output_path_2) # Visualize the outputs by image.
    
    

def main():
    config_file = 'params/paths_zmachine.json'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(config_file, 'r') as f:
        config = json.load(f)

    _, _, test_loader = load_data(config)
    test_img_idx = config['test_img_idx'] 
    imgs, true_masks = list(test_loader)[test_img_idx]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    key_str = str(test_loader.dataset.image_file_paths[test_img_idx].stem).split('_')[1][-4:]
    output_path_1 = Path(config['root_dir']) / 'outputs' / key_str / f'combined_output_{key_str}_{timestamp}_split.png'
    output_path_2 = Path(config['root_dir']) / 'outputs' / key_str /f'combined_output_{key_str}_{timestamp}.png'
    output_path_3 = Path(config['root_dir']) / 'outputs' / key_str /f'uncertainty_map_{key_str}_{timestamp}.png'

    evaluate(imgs, true_masks, config, device, output_path_1, output_path_2, output_path_3)  

if __name__ == '__main__':
    main()