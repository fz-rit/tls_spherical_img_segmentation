import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data
from backup.training import train_unet
from tools.load_tools import calc_metrics
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from backup.training import create_unet_multi_channels


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


def enable_dropout(model):
    """
    Enable dropout layers in the model while keeping the rest in evaluation mode.
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()

def predict_with_uncertainty(model, image, num_samples=50):
    # Ensure the model is in evaluation mode
    model.eval()
    # Enable dropout layers for MC sampling without affecting BatchNorm layers
    enable_dropout(model)
    
    preds_samples = []
    aleatoric_vars_samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            preds, log_vars = model(image)
            preds_samples.append(preds)
            aleatoric_vars_samples.append(torch.exp(log_vars))
    
    preds_samples = torch.stack(preds_samples)  # Shape: (num_samples, B, num_classes, H, W)
    aleatoric_vars_samples = torch.stack(aleatoric_vars_samples)
    
    # Epistemic uncertainty: variance over predictions
    epistemic_var = preds_samples.var(dim=0)
    # Aleatoric uncertainty: mean predicted variance
    aleatoric_var = aleatoric_vars_samples.mean(dim=0)
    
    predictive_variance = epistemic_var + aleatoric_var
    mean_preds = preds_samples.mean(dim=0)
    
    return mean_preds, predictive_variance





config_file = 'params/paths_zmachine.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with open(config_file, 'r') as f:
    config = json.load(f)


_, _, test_loader = load_data(config)
model = load_model(config, device)
# model.eval()
# enable_dropout(model)


# Get one batch from the validation loader
imgs, true_masks = next(iter(test_loader))

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
    img = imgs[i].permute(1, 2, 0).numpy()  
    
    # True mask and predicted mask are [H,W] arrays with class indices.
    # For visualization, we show them as a simple colormapped image.
    true_mask = true_masks[i].numpy()
    pred_mask = pred_masks[i].numpy()

    # Compute metrics between true_mask and pred_mask
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    num_classes = preds.shape[1]
    cm, OverallAccu, mAccu, mIoU, FWIoU, dice_coefficient = calc_metrics(true_flat, pred_flat, num_classes)
    
    
    if num_samples == 1:
        axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]
    else:
        axs_img, axs_true, axs_pred = axs[i, 0], axs[i, 1], axs[i, 2]

    display_channels = [4, 0, 2] # Roughness, Intensity, Range
    axs_img.imshow(img[:, :, display_channels])
    axs_img.set_title('Original Image')
    axs_img.axis('off')

    # For masks, use a discrete colormap to distinguish classes
    axs_true.imshow(true_mask, cmap='tab20', interpolation='nearest')
    axs_true.set_title('Ground Truth Mask')
    axs_true.axis('off')

    axs_pred.imshow(pred_mask, cmap='tab20', interpolation='nearest')
    axs_pred.set_title(f'Predicted Mask'
                       f'\noAccu: {OverallAccu:.4f};'
                       f'\nmAccu: {mAccu:.4f};'
                       f'\nmIoU: {mIoU:.4f};'
                       f'\nFWIoU: {FWIoU:.4f};'
                       f'\ndice_coeff: {dice_coefficient:.4f}')
    axs_pred.axis('off')

plt.tight_layout()
plt.show()
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
fig.savefig(f"outputs/output_{timestamp}.png")
print(f"Saved output_{timestamp}.png")
