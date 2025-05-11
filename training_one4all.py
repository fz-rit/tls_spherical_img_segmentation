import gc
import copy
import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data, NUM_CLASSES
import torch.nn as nn
from pathlib import Path
import json
import numpy as np
from tools.metrics_tools import calc_oAccu_mIoU
from tools.visualize_tools import plot_training_validation_losses, plot_training_validation_metrics
from tools.load_tools import save_model_locally, dump_dict_to_yaml
import time
from pprint import pformat
from tools.earlystopping import EarlyStopping
from tools.logger_setup import Logger

log = Logger()

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


def expand_first_conv_to_multi_channels(model, expand_channels=6):
    """
    Expands the first convolutional layer of the model to accept more input channels.
    Args:
        model (torch.nn.Module): The model to modify.
        expand_channels (int): The number of input channels to expand to.
    """

    # 1. Get the pretrained conv1
    pretrained_conv1 = model.encoder.conv1  # [64, 3, 7, 7]
    pretrained_weights = pretrained_conv1.weight.data  # Tensor of shape [64, 3, 7, 7]

    # 2. Create a new Conv2d for multi channels (8 in, 64 out)
    new_conv1 = nn.Conv2d(
        in_channels=expand_channels,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False
    )

    # Copy pretrained weights for pretrain channels 
    with torch.no_grad():  
        new_conv1.weight[:, :3, :, :] = pretrained_weights
        if expand_channels == 6:
            new_conv1.weight[:, 3:, :, :] = pretrained_weights
        elif expand_channels == 9:
            new_conv1.weight[:, 3:6, :, :] = pretrained_weights
            new_conv1.weight[:, 6:, :, :] = pretrained_weights
        else:
            raise ValueError(f"Unsupported number of input channels: {expand_channels}, only 3/6/9 are supported.")

    model.encoder.conv1 = new_conv1



def build_model_for_multi_channels(model_name, encoder_name='resnet34', in_channels=3):

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
    elif model_name == 'segformer':
        model_class = smp.Segformer
    else:
        raise ValueError(f"Unknown model name: {model_name}")


    encoder_name_list = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
        'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d',
        'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
        'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
        'efficientnet-b6', 'efficientnet-b7', 'mit_b2', 'mit_b3', 
        'timm-efficientnet-b5'
    ]
    if encoder_name not in encoder_name_list:
        raise ValueError(f"Unknown encoder name: {encoder_name}. Supported encoders are: {encoder_name_list}")

    model = model_class(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,   # SMP default (RGB)
        classes=NUM_CLASSES
    )

    if in_channels > 3:
        expand_first_conv_to_multi_channels(model, expand_channels=in_channels)
    # add_dropout_to_decoder(model, p=dropout_rate)

    return model


def train_model(config, input_channels, model_name, pretrained_model_source=False, save_model=False):
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    train_loader, val_loader, test_loader = load_data(config, input_channels=input_channels)
    model = build_model_for_multi_channels(
        model_name=model_name,
        encoder_name=config['encoder_name'],
        in_channels=len(input_channels)
    )
    pretrained_epoch = 0
    stop_early = config['stop_early']
    dummy_shape = list(next(iter(test_loader))[0].shape)
    epoch_num = config['epoch_num']
    channel_info_str = '_'.join([str(ch) for ch in input_channels])
    # If loading from a pretrained model
    if pretrained_model_source:
        model_dir = Path(config['root_dir']) / config['model_dir'] / model_name
        model_file = model_dir / config['model_file']
        pretrained_epoch = int(model_file.stem.split('_')[-1])
        if model_file.exists():
            model.load_state_dict(torch.load(model_file, weights_only=True))
            log.info(f"======🌞Loaded model from disk: {model_file.stem}.======")
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

    
    # Tracking containers
    train_losses, val_losses = [], []
    train_oAccus, val_oAccus = [], []
    train_mIoUs, val_mIoUs = [], []

    early_stopper = EarlyStopping(patience=config['early_stop_patience'], mode='loss')

    # -------------------------------------------------------------------------
    # 2. Keep track of the best model
    # -------------------------------------------------------------------------
    best_val_loss = float('inf')
    best_model_state = None  # For storing state_dict of the best model

    # -------------------------------------------------------------------------
    # 3. Training loop
    # -------------------------------------------------------------------------
    for epoch in range(pretrained_epoch, epoch_num):
        # ---------------------
        #   Training phase
        # ---------------------
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

        # ---------------------
        #   Validation phase
        # ---------------------
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

        # ---------------------
        #   Logging
        # ---------------------
        log.info(f"Epoch {epoch + 1}/{epoch_num} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train oAcc: {train_oAccu:.4f}, mIoU: {train_mIoU:.4f} | "
              f"Val oAcc: {val_oAccu:.4f}, mIoU: {val_mIoU:.4f}")

        # ---------------------------------------------------------------------
        #  Update best model if the current validation loss is better
        # ---------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        # ---------------------------------------------------------------------
        #  Early stopping
        # ---------------------------------------------------------------------
        early_stopper(val_loss)
        save_by_interval = (epoch + 1) % config['model_save_interval'] == 0
        save_due_to_early_stop = early_stopper.early_stop and stop_early

        # ---------------------------------------------------------------------
        #  Save at interval or if triggered by early stopping
        # ---------------------------------------------------------------------
        if save_model and (save_by_interval or save_due_to_early_stop):
            model_dir = Path(config['root_dir']) / config['model_dir'] / model_name
            timestr = time.strftime("%Y%m%d_%H%M%S")
            model_name_prefix = f"model_name_{config['encoder_name']}_epoch_{epoch+1:03d}__{channel_info_str}_{timestr}"
            save_model_locally(
                model=model,
                model_dir=model_dir,
                model_name_prefix=model_name_prefix,
                dummy_shape=dummy_shape
            )

        # If early stopping says to stop, break now
        if save_due_to_early_stop:
            log.info("⏹ Early stopping triggered!")
            break

    # -------------------------------------------------------------------------
    # 4. After the loop: Optionally save the best model
    # -------------------------------------------------------------------------
    if save_model and best_model_state is not None:
        # Optionally reload the best state into the model
        model.load_state_dict(best_model_state)

        # Save it under a 'best' prefix
        model_dir = Path(config['root_dir']) / config['model_dir'] / model_name
        timestr = time.strftime("%Y%m%d_%H%M%S")
        
        model_name_prefix = f"{model_name}_{config['encoder_name']}_best_{channel_info_str}_{timestr}"
        save_model_locally(
            model=model,
            model_dir=model_dir,
            model_name_prefix=model_name_prefix,
            dummy_shape=dummy_shape
        )
        log.info(f"✅ Best model saved with val_loss = {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Save plots and return
    # -------------------------------------------------------------------------
    timestr = time.strftime("%Y%m%d_%H%M%S")
    plt_save_path = Path(config['root_dir']) / 'outputs' / model_name / f'losses_{channel_info_str}_{timestr}.png'
    plot_training_validation_losses(train_losses, val_losses, plt_save_path)

    metrics_save_path = Path(config['root_dir']) / 'outputs' / model_name / f'metrics_{channel_info_str}_{timestr}.png'
    plot_training_validation_metrics(train_oAccus, val_oAccus, train_mIoUs, val_mIoUs, metrics_save_path)

    out_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_oAccus': train_oAccus,
        'val_oAccus': val_oAccus,
        'train_mIoUs': train_mIoUs,
        'val_mIoUs': val_mIoUs
    }
    train_log_file = Path(config['root_dir']) / 'outputs' / model_name / f'train_log_{channel_info_str}.yaml'
    dump_dict_to_yaml(out_dict, train_log_file)
    return out_dict


if __name__ == "__main__":
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)
    log.info(pformat(config))
    input_channels_ls = config['input_channels_ls']
    model_name_ls = config['model_name_ls']
    for model_name in model_name_ls:
        for input_channels in input_channels_ls:
            log.info(f"✈️ Training {model_name} with input channels: {input_channels}")
            train_model(config, 
                        input_channels=input_channels, 
                        model_name= model_name, 
                        pretrained_model_source=False, save_model=True)
            
            #Explicitly free GPU memory after each run
            torch.cuda.empty_cache()
            gc.collect()
    