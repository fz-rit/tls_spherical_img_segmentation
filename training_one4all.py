import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data, PATCH_WIDTH, NUM_CLASSES
import torch.nn as nn
from pathlib import Path
import json
import torch.onnx
from monte_carlo_dropout import add_dropout_to_decoder
import numpy as np
from tools import calc_oAccu_mIoU, visualize_losses, visualize_metrics, save_model_locally, EarlyStopping
import time


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
      3. Copies weights of channels 0, 2, 4 to the new conv1
      4. Randomly initializes channels 1, 3, 5, 6, 7
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




def build_model_for_multi_channels(model_name, encoder_name='resnet34', dropout_rate=0.3):

    if model_name == 'unetpp':
        model_class = smp.UnetPlusPlus
    elif model_name == 'unet':
        model_class = smp.Unet
    elif model_name == 'fpn':
        model_class = smp.FPN
    elif model_name == 'linknet':
        model_class = smp.Linknet
    elif model_name == 'pspnet':
        model_class = smp.PSPNet
    elif model_name == 'deeplabv3':
        model_class = smp.DeepLabV3
    elif model_name == 'deeplabv3plus':
        model_class = smp.DeepLabV3Plus
    else:
        raise ValueError(f"Unknown model name: {model_name}")


    encoder_name_list = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'efficientnet-b6', 'efficientnet-b7'
    ]
    if encoder_name not in encoder_name_list:
        raise ValueError(f"Unknown encoder name: {encoder_name}. Supported encoders are: {encoder_name_list}")

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,   # SMP default (RGB)
        classes=NUM_CLASSES
    )

    expand_first_conv_to_multi_channels(model)
    add_dropout_to_decoder(model, p=dropout_rate)

    return model


def train_model(config, pretrained_model_source=False, save_model=False):
    train_loader, val_loader, _ = load_data(config)
    model = build_model_for_multi_channels(model_name=config['model_name'],
                                           encoder_name=config['encoder_name'])
    pretrained_epoch = 0

    if pretrained_model_source:
        model_dir = Path(config['root_dir']) / config['model_dir']
        model_file = model_dir / config['model_file']
        pretrained_epoch = int(model_file.stem.split('_')[-1])
        if model_file.exists():
            model.load_state_dict(torch.load(model_file, weights_only=True))
            print(f"======🌞Loaded model from disk: {model_file.stem}.======")
        else:
            raise FileNotFoundError(f"Pretrained model not found at {model_file}")
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointLoss(
        smp.losses.DiceLoss(mode='multiclass'),
        torch.nn.CrossEntropyLoss(),
        first_weight=0.5,
        second_weight=0.5
    )

    model_dir = Path(config['root_dir']) / config['model_dir'] / config['model_name']
    dummy_shape = (3, 8, 512, PATCH_WIDTH)
    epoch_num = config['epoch_num']
    train_losses = []
    val_losses = []
    train_oAccus, val_oAccus = [], []
    train_mIoUs, val_mIoUs = [], []
    early_stopper = EarlyStopping(patience=10, mode='loss')

    for epoch in range(pretrained_epoch, epoch_num):
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = loss_fn(preds, masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Collect predictions and labels
            preds_labels = torch.argmax(preds, dim=1)
            y_true_train.append(masks.cpu().numpy().ravel())
            y_pred_train.append(preds_labels.cpu().numpy().ravel())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Compute train metrics
        y_true_flat = np.concatenate(y_true_train)
        y_pred_flat = np.concatenate(y_pred_train)
        train_oAccu, train_mIoU = calc_oAccu_mIoU(y_true_flat, y_pred_flat, NUM_CLASSES)
        train_oAccus.append(train_oAccu)
        train_mIoUs.append(train_mIoU)

        # Validation
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = loss_fn(preds, masks)
                val_loss += loss.item()

                preds_labels = torch.argmax(preds, dim=1)
                y_true_val.append(masks.cpu().numpy().ravel())
                y_pred_val.append(preds_labels.cpu().numpy().ravel())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Compute val metrics
        y_true_flat = np.concatenate(y_true_val)
        y_pred_flat = np.concatenate(y_pred_val)
        val_oAccu, val_mIoU = calc_oAccu_mIoU(y_true_flat, y_pred_flat, NUM_CLASSES)
        val_oAccus.append(val_oAccu)
        val_mIoUs.append(val_mIoU)

        print(f"Epoch {epoch + 1}/{epoch_num} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train oAcc: {train_oAccu:.4f}, mIoU: {train_mIoU:.4f} | "
              f"Val oAcc: {val_oAccu:.4f}, mIoU: {val_mIoU:.4f}")
        
        early_stopper(val_loss)
        should_save_by_interval = (epoch + 1) % config['model_save_interval'] == 0
        should_save_due_to_early_stop = early_stopper.early_stop

        # Save model if conditions are met
        if save_model and (should_save_by_interval or should_save_due_to_early_stop):
            save_model_locally(model=model, 
                        model_dir=model_dir,
                        epoch=epoch+1, 
                        dummy_shape=dummy_shape,
                        )
            
        if should_save_due_to_early_stop:
            print("⏹ Early stopping triggered!")
            break

    # Save loss figure
    timestr = time.strftime("%Y%m%d_%H%M%S")
    plt_save_path = Path(config['root_dir']) / 'outputs' / config['model_name'] / f'losses_{timestr}.png'
    visualize_losses(train_losses, val_losses, plt_save_path)

    # Save metrics figure
    metrics_save_path = Path(config['root_dir']) / 'outputs' / config['model_name'] / f'metrics_{timestr}.png'
    visualize_metrics(train_oAccus, val_oAccus, train_mIoUs, val_mIoUs, metrics_save_path)

    out_dict = {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_oAccus': train_oAccus,
        'val_oAccus': val_oAccus,
        'train_mIoUs': train_mIoUs,
        'val_mIoUs': val_mIoUs
    }
    return out_dict




if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    out_dict = train_model(config, pretrained_model_source=False, save_model=True)