import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt

class UnetWithUncertainty(nn.Module):
    def __init__(self, base_model, num_classes):
        """
        Wrap an SMP UNet to output (preds, log_vars).
        Args:
            base_model: the original SMP UNet
            num_classes: number of segmentation classes
        """
        super(UnetWithUncertainty, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        
        # Replace the segmentation head so that it outputs 2 * num_classes channels.
        # Here we assume that the original segmentation head has an attribute `in_channels`.
        # in_channels = self.base_model.segmentation_head.in_channels
        
        # Create a new segmentation head: a simple conv that outputs 2 * num_classes channels.
        self.base_model.segmentation_head = nn.Conv2d(
            16, num_classes * 2, kernel_size=3, padding=1
        )

         # Initialize weights for log_vars to be close to zero
        with torch.no_grad():
            self.base_model.segmentation_head.weight[num_classes:].fill_(0)
    
    def forward(self, x):
        """
        Forward pass returns a tuple: (preds, log_vars)
        """
        # Get output from the base model. Expected shape: (batch, num_classes * 2, H, W)
        out = self.base_model(x)
        # Split the output along the channel dimension into predictions and log variance.
        preds, log_vars = torch.chunk(out, chunks=2, dim=1)
        return preds, log_vars
    


class JointLoss(nn.Module):
    def __init__(self, first_loss, second_loss, first_weight=0.5, second_weight=0.5):
        super(JointLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, y_pred, y_true):
        loss1 = self.first_loss(y_pred, y_true)
        loss2 = self.second_loss(y_pred, y_true)
        return self.first_weight * loss1 + self.second_weight * loss2

class HeteroscedasticLoss(nn.Module):
    def __init__(self, base_loss):
        super(HeteroscedasticLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, outputs, targets):
        # outputs is a tuple: (preds, log_vars)
        preds, log_vars = outputs
        
        # Compute per-pixel base loss.
        # For cross entropy, preds should be of shape (B, C, H, W)
        # and targets of shape (B, H, W), and we set reduction='none'
        ce_loss = self.base_loss(preds, targets)  # Shape: (B, H, W)

        # Clamp log_vars to avoid extreme values
        log_vars = torch.clamp(log_vars, min=-2, max=2)
        
        # If log_vars is of shape (B, C, H, W), we assume class-independent uncertainty.
        # Average log_vars over the class dimension.
        log_vars_avg = log_vars.mean(dim=1)  # Now shape: (B, H, W)
        
        # Weight the per-pixel loss by exp(-log_vars) and add the log variance as a regularizer.
        weighted_loss = torch.exp(-log_vars_avg) * ce_loss + log_vars_avg
        
        # Return the mean loss over all pixels and examples.
        return weighted_loss.mean()



def expand_first_conv_to_multi_channels(model):
    """
    model: The SMP model after loading with 3-channel pretrained weights.
    This function:
      1. Reads the existing pretrained conv1 (shape [64, 3, 7, 7])
      2. Creates a new conv1 (shape [64, 6, 7, 7])
      3. Copies the first 3 channel weights
      4. Randomly initializes channels 4-6
      5. Assigns the new conv1 back
    """

    # 1. Get the pretrained conv1
    pretrained_conv1 = model.encoder.conv1  # [64, 3, 7, 7]
    pretrained_weights = pretrained_conv1.weight.data  # Tensor of shape [64, 3, 7, 7]

    # 2. Create a new Conv2d for multi channels (8 in, 64 out)
    new_conv1 = nn.Conv2d(
        in_channels=8,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    
    with torch.no_grad():
        # 3. Copy pretrained weights for pretrain channels 
        pretrain_channels = [4, 2, 0] # Roughness, Intensity, Range
        new_conv1.weight[:, pretrain_channels, :, :] = pretrained_weights

        # 4. Randomly init the remaining 3 channels (channels 4-6)
        non_pretrain_channels = [1, 3, 5, 6, 7]
        nn.init.kaiming_normal_(new_conv1.weight[:, non_pretrain_channels, :, :], mode='fan_out', nonlinearity='relu')

    # 5. Assign the new conv back
    model.encoder.conv1 = new_conv1


def create_unet_multi_channels():
    # 1) Load a 3-channel model with pretrained resnet34
    model_unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,   # SMP default (RGB)
        classes=6
    )

    # 2) Expand the first conv to 6 channels, partially preserving pretrained weights
    expand_first_conv_to_multi_channels(model_unet)

    # 3) Wrap the model so it outputs both predictions and uncertainty.
    model_with_uncertainty = UnetWithUncertainty(model_unet, num_classes=6)
    return model_with_uncertainty




def visualize_losses(train_losses, val_losses):
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_unet(config, save_model=False):
    train_loader, val_loader, _ = load_data(config)
    model = create_unet_multi_channels()

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # loss_fn = JointLoss(
    #     smp.losses.DiceLoss(mode='multiclass'),
    #     torch.nn.CrossEntropyLoss(),
    #     first_weight=0.5,
    #     second_weight=0.5
    # )
    loss_fn = HeteroscedasticLoss(JointLoss(
                                        smp.losses.DiceLoss(mode='multiclass'),
                                        torch.nn.CrossEntropyLoss(),
                                        first_weight=0.5,
                                        second_weight=0.5
                                            )
                                 )

    epoch_num = config['epoch_num']
    train_losses = []
    val_losses = []

    
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            # preds, log_vars  = model(imgs)
            outputs = model(imgs) # outputs is a tuple: (preds, log_vars)
            loss = loss_fn(outputs, masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{epoch_num}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if save_model:
        model_dir = Path(config['root_dir']) / config['model_dir']
        if not model_dir.exists():
            model_dir.mkdir(parents=True)
            print(f"----Created directory {model_dir}----")
        save_model_path = model_dir / f'model_epoch{epoch_num:03d}.pth'
        torch.save(model.state_dict(), save_model_path)
        print(f"----Model saved at {save_model_path}----")

    visualize_losses(train_losses, val_losses)
    return model, train_losses, val_losses



if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    model, train_losses, val_losses = train_unet(config, save_model=True)
    


