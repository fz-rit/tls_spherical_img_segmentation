import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data
from training import train_unet
from tools import calc_metrics
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime


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
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=6,
            classes=5
        )
        model.load_state_dict(torch.load(model_dir / 'model_epoch200.pth', weights_only=True))
        print("======Loaded model from disk.======")
    else:
        model = train_unet(config)
        print("####Trained a new model.####")
    
    model = model.to(device)

    return model

config_file = 'params/paths_zmachine.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(config_file, 'r') as f:
    config = json.load(f)


_, _, val_loader = load_data(config)
model = load_model(config, device)
model.eval()


# Get one batch from the validation loader
imgs, true_masks = next(iter(val_loader))

imgs = imgs.to(device)                  # (N, 3, H, W)
true_masks = true_masks.to(device)      # (N, H, W)

with torch.no_grad():
    preds = model(imgs)                 # (N, C, H, W)
    # Convert logits to predicted class indices
    pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

# Move data back to CPU for visualization
imgs = imgs.cpu()
true_masks = true_masks.cpu()
pred_masks = pred_masks.cpu()

# Number of samples to visualize (up to your batch size)
num_samples = max(5, imgs.size(0))  # Let's visualize up to 5 samples

fig, axs = plt.subplots(num_samples, 3, figsize=(10, 3*num_samples))

for i in range(num_samples):
    # Original image: convert from tensor [C,H,W] to [H,W,C] and un-normalize if needed
    # If you normalized your images earlier, you might have to denormalize here.
    img = imgs[i].permute(1, 2, 0).numpy()  
    img = (img - img.min()) / (img.max() - img.min())  # Simple normalization for display
    
    # True mask and predicted mask are [H,W] arrays with class indices.
    # For visualization, we show them as a simple colormapped image.
    true_mask = true_masks[i].numpy()
    pred_mask = pred_masks[i].numpy()

    # Compute metrics between true_mask and pred_mask
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    num_classes = preds.shape[1]
    cm, pixel_accuracy, mPA, mIoU, FWIoU, dice_coefficient = calc_metrics(true_flat, pred_flat, num_classes)
    
    
    if num_samples == 1:
        axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]
    else:
        axs_img, axs_true, axs_pred = axs[i, 0], axs[i, 1], axs[i, 2]

    axs_img.imshow(img[:, :, :3])
    axs_img.set_title('Original Image')
    axs_img.axis('off')

    # For masks, use a discrete colormap to distinguish classes
    axs_true.imshow(true_mask, cmap='tab20', interpolation='nearest')
    axs_true.set_title('Ground Truth Mask')
    axs_true.axis('off')

    axs_pred.imshow(pred_mask, cmap='tab20', interpolation='nearest')
    axs_pred.set_title(f'Predicted Mask'
                       f'\noverallPA: {pixel_accuracy:.4f};'
                       f'\nmPA: {mPA:.4f};'
                       f'\nmIoU: {mIoU:.4f};'
                       f'\nFWIoU: {FWIoU:.4f};'
                       f'\ndice_coeff: {dice_coefficient:.4f}')
    axs_pred.axis('off')

plt.tight_layout()
plt.show()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig.savefig(f"output_{timestamp}.png")
print(f"Saved output_{timestamp}.png")
