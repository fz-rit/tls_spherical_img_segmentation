import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt

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

def expand_first_conv_to_6_channels(model):
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

    # 2. Create a new Conv2d for 6 channels
    new_conv1 = nn.Conv2d(
        in_channels=6,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    # 3. Copy pretrained weights for first 3 channels
    with torch.no_grad():
        new_conv1.weight[:, :3, :, :] = pretrained_weights

        # 4. Randomly init the remaining 3 channels (channels 4-6)
        nn.init.kaiming_normal_(new_conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

    # 5. Assign the new conv back
    model.encoder.conv1 = new_conv1


def create_unet_6channels():
    # 1) Load a 3-channel model with pretrained resnet34
    model_unet = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,   # SMP default (RGB)
        classes=5
    )

    # 2) Expand the first conv to 6 channels, partially preserving pretrained weights
    expand_first_conv_to_6_channels(model_unet)

    return model_unet


def visualize_losses(train_losses, val_losses):
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def train_unet(config, save_model=False):
    _, train_loader, val_loader = load_data(config)
    model = create_unet_6channels()

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # loss_fn = smp.losses.DiceLoss(mode='multiclass')
    loss_fn = JointLoss(
        smp.losses.DiceLoss(mode='multiclass'),
        torch.nn.CrossEntropyLoss(),
        first_weight=0.5,
        second_weight=0.5
    )
    epoch_num = 150
    train_losses = []
    val_losses = []

    
    for epoch in range(epoch_num):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)
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
    


