import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data
import torch.nn as nn
from pathlib import Path
import json
import matplotlib.pyplot as plt
import torch.onnx

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
        # new_conv1.weight[:, :3, :, :] = pretrained_weights
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

    return model_unet


def visualize_losses(train_losses, val_losses):
    
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def save_model_locally(model, model_dir, epoch_num, dummy_shape, device):
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"----Created directory {model_dir}----")

    # Save .pth (PyTorch state dict)
    save_model_path = model_dir / f'model_epoch{epoch_num:03d}.pth'
    torch.save(model.state_dict(), save_model_path)
    print(f"----Model saved at {save_model_path}----")

    # Save as ONNX
    onnx_model_path = model_dir / f'model_epoch{epoch_num:03d}.onnx'

    # Create a dummy input with the same shape as your input
    dummy_input = torch.randn(dummy_shape).to(device)  # replace H, W as needed

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_model_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,  # common version; increase if needed
        do_constant_folding=True
    )
    print(f"----ONNX model saved at {onnx_model_path}----")

def train_unet(config, save_model=False):
    train_loader, val_loader, _ = load_data(config)
    model = create_unet_multi_channels()

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointLoss(
        smp.losses.DiceLoss(mode='multiclass'),
        torch.nn.CrossEntropyLoss(),
        first_weight=0.5,
        second_weight=0.5
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
        dummy_shape = (5, 8, 512, 1440)
        save_model_locally(model=model, 
                        model_dir=model_dir,
                        epoch_num=epoch_num, 
                        dummy_shape=dummy_shape,
                        device=device)

    visualize_losses(train_losses, val_losses)
    return model, train_losses, val_losses



if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    model, train_losses, val_losses = train_unet(config, save_model=True)
    


