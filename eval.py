import torch
import matplotlib.pyplot as plt
from training import train_unet

# Load the trained model and validation loader
model, val_loader = train_unet('params/paths_zmachine.json')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Assuming model, device, and val_loader are already defined
# val_loader should provide (images, masks)
model.eval()

# Get one batch from the validation loader
imgs, true_masks = next(iter(val_loader))

# Move data to the appropriate device
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
num_samples = max(5, imgs.size(0))  # Let's visualize up to 4 samples

fig, axs = plt.subplots(num_samples, 3, figsize=(10, 3*num_samples))

for i in range(num_samples):
    # Original image: convert from tensor [C,H,W] to [H,W,C] and un-normalize if needed
    # If you normalized your images earlier, you might have to denormalize here.
    img = imgs[i].permute(1, 2, 0).numpy()  
    img = (img - img.min()) / (img.max() - img.min())  # Simple normalization for display
    
    # True mask and predicted mask are [H,W] arrays with class indices
    # For visualization, you can either display them as is (if few classes)
    # or use a color mapping. Here we show them as a simple colormapped image.
    true_mask = true_masks[i].numpy()
    pred_mask = pred_masks[i].numpy()
    
    if num_samples == 1:
        # If only one sample, axs is not a 2D array
        axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]
    else:
        axs_img, axs_true, axs_pred = axs[i, 0], axs[i, 1], axs[i, 2]

    axs_img.imshow(img)
    axs_img.set_title('Original Image')
    axs_img.axis('off')

    # For masks, use a discrete colormap to distinguish classes
    axs_true.imshow(true_mask, cmap='tab20', interpolation='nearest')
    axs_true.set_title('Ground Truth Mask')
    axs_true.axis('off')

    axs_pred.imshow(pred_mask, cmap='tab20', interpolation='nearest')
    axs_pred.set_title('Predicted Mask')
    axs_pred.axis('off')

plt.tight_layout()
plt.show()
