import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, load_image_cube_and_metadata
from training import train_unet
from tools import calc_metrics, custom_cmap
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from training import create_unet_multi_channels
from pprint import pprint


def load_model(config: dict, device: str) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    
    model_dir = Path(config['root_dir']) / config['model_dir']
    # Load the model if there is a saved model, otherwise train a new model
    if model_dir.exists() and any(model_dir.glob('*.pth')):
        model = create_unet_multi_channels()
        model_path = next(model_dir.glob('model_epoch*.pth'))
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"======Loaded model from disk: {model_path}.======")
    else:
        model, _, _  = train_unet(config)
        print("####Trained a new model.####")
    
    model = model.to(device)

    return model



def visualize_eval_output(imgs, true_masks, pred_masks, output_path: Path = None):
    """"
    Visualize the image and masks by patch.


    """
    N_CLASSES = 6
    num_samples = min(5, len(imgs))  # Let's visualize up to 5 samples
    fig, axs = plt.subplots(3, num_samples, figsize=(10, 6*num_samples))

    for i in range(num_samples):
        # Original image: convert from tensor [C,H,W] to [H,W,C] and un-normalize if needed
        img = imgs[i].permute(1, 2, 0).numpy()  
        
        # True mask and predicted mask are [H,W] arrays with class indices.
        # For visualization, we show them as a simple colormapped image.
        true_mask = true_masks[i].numpy()
        pred_mask = pred_masks[i].numpy()

        # Compute metrics between true_mask and pred_mask
        true_flat = true_mask.flatten()
        pred_flat = pred_mask.flatten()
        
        cm, OverallAccu, mAccu, mIoU, FWIoU, dice_coefficient = calc_metrics(true_flat, pred_flat, N_CLASSES)
        
        
        if num_samples == 1:
            axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]
            pred_title = ' '.join(['Predicted Mask:',
                        f'oAccu: {OverallAccu:.4f};',
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
        axs_true.imshow(true_mask, cmap=custom_cmap(), interpolation='nearest')
        axs_true.set_title('Ground Truth Mask')
        axs_true.axis('off')

        axs_pred.imshow(pred_mask, cmap=custom_cmap(), interpolation='nearest')
        axs_pred.set_title(pred_title)
        axs_pred.axis('off')

    plt.tight_layout()
    plt.show()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/output_{timestamp}.png" if output_path is None else output_path
    fig.savefig(output_path)
    print(f"Saved {output_path}")


def evaluate(imgs, true_masks, model, 
             device:str = 'cuda', 
             output_path_1: Path = None, 
             output_path_2: Path = None):
    """
    Evaluate the model on the test set.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to run the evaluation on.
    """
    imgs = imgs.to(device)                  # (N, C, H, W)
    true_masks = true_masks.to(device)      # (N, H, W)
    with torch.no_grad():
        preds = model(imgs)                 # (N, C, H, W)
        # Convert logits to predicted class indices
        pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

    # Move data back to CPU for visualization
    imgs = imgs.cpu()
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()
    
    visualize_eval_output(imgs, true_masks, pred_masks, output_path_1) # Visualize the outputs by patch.


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


    # Load test image and mask from test_loader
        _, _, test_loader = load_data(config)
    model = load_model(config, device)
    model.eval()

    # Get one batch from the validation loader
    # imgs, true_masks = next(iter(test_loader))
    desired_index = 1 
    imgs, true_masks = list(test_loader)[desired_index]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    key_str = str(test_loader.dataset.image_file_paths[0].stem).split('_')[1][-4:]
    output_path_1 = Path(config['root_dir']) / 'outputs' / f'combined_output_{key_str}_{timestamp}_split.png'
    output_path_2 = Path(config['root_dir']) / 'outputs' / f'combined_output_{key_str}_{timestamp}.png'

    evaluate(imgs, true_masks, model, device, output_path_1, output_path_2)

if __name__ == '__main__':
    main()