import gc
import copy
import segmentation_models_pytorch as smp
import torch
from prepare_dataset import load_data
from pathlib import Path
import numpy as np
from tools.metrics_tools import calc_oAccu_mIoU
from tools.visualize_tools import plot_training_validation_losses, plot_training_validation_metrics
from tools.load_tools import save_model_locally, dump_dict_to_yaml, CONFIG
import time
from pprint import pformat
from tools.earlystopping import EarlyStopping
from tools.logger_setup import Logger
from tools.feature_fusion_helper import build_model_for_multi_channels, JointLoss

log = Logger()
IGNORE_VAL = 255


def train_model(config, train_subset_cnt, input_channels, model_setup_dict, 
                load_pretrain=False, save_model=False, save_onnx=False):
    train_loader, val_loader, _ = load_data(config, input_channels=input_channels, train_subset_cnt=train_subset_cnt)
    train_out_root_dir = Path(config['root_dir']) / f"run_subset_{train_subset_cnt:02d}"

    model_name = model_setup_dict['arch']
    encoder = model_setup_dict['encoder']
    out_file_str = model_setup_dict['name']
    model_dir = train_out_root_dir / config['model_dir'] / out_file_str
    model_dir.mkdir(parents=True, exist_ok=True)
    model = build_model_for_multi_channels(
        model_name=model_name,
        encoder_name=encoder,
        in_channels=len(input_channels),
        num_classes=config['num_classes']
    )
    pretrained_epoch = 0
    stop_early = config['stop_early']
    dummy_shape = list(next(iter(train_loader))[0].shape)
    epoch_num = config['max_epoch_num']
    channel_info_str = '_'.join([str(ch) for ch in input_channels])
    num_classes = config['num_classes']
    if load_pretrain:
        model_file = model_dir / config['pretrained_model_file']
        pretrained_epoch = int(model_file.stem.split('_')[-1])
        if model_file.exists():
            model.load_state_dict(torch.load(model_file, weights_only=True))
            log.info(f"======🌞Loaded model from disk: {model_file.stem}.======")
        else:
            raise FileNotFoundError(f"Pretrained model not found at {model_file}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your PyTorch installation.")
    model = model.to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = JointLoss(
        smp.losses.DiceLoss(mode='multiclass', ignore_index=IGNORE_VAL),
        torch.nn.CrossEntropyLoss(ignore_index=IGNORE_VAL),
        first_weight=0.5,
        second_weight=0.5
    )
    
    train_losses, val_losses = [], []
    train_oAccus, val_oAccus = [], []
    train_mIoUs, val_mIoUs = [], []

    early_stopper = EarlyStopping(patience=config['early_stop_patience'], mode='loss')
    best_val_loss = float('inf')
    best_model_state = None  # For storing state_dict of the best model

    # -------------------------------------------------------------------------
    # 3. Training loop
    # -------------------------------------------------------------------------
    for epoch in range(pretrained_epoch, epoch_num):
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for imgs, masks, buf_masks in train_loader:
            imgs = imgs.to('cuda')
            masks = masks.to('cuda')
            buf_masks = buf_masks.to('cuda')
            preds = model(imgs)
            valid_mask = buf_masks.squeeze(1).bool()
            core_masks = masks.clone()
            core_masks[~valid_mask] = IGNORE_VAL

            loss = loss_fn(preds, core_masks)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds_labels = torch.argmax(preds, dim=1)
            y_true_train.append(core_masks[valid_mask].cpu().numpy().ravel())
            y_pred_train.append(preds_labels[valid_mask].cpu().numpy().ravel())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Compute train metrics
        y_true_flat = np.concatenate(y_true_train)
        y_pred_flat = np.concatenate(y_pred_train)
        train_oAccu, train_mIoU = calc_oAccu_mIoU(y_true_flat, y_pred_flat, num_classes)
        train_oAccus.append(train_oAccu)
        train_mIoUs.append(train_mIoU)

        # ---------------------
        #   Validation phase
        # ---------------------
        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for imgs, masks, buf_masks in val_loader:
                imgs = imgs.to('cuda')
                masks = masks.to('cuda')
                buf_masks = buf_masks.to('cuda')

                preds = model(imgs)
                valid_mask = buf_masks.squeeze(1).bool()
                core_masks = masks.clone()
                core_masks[~valid_mask] = IGNORE_VAL
                loss = loss_fn(preds, core_masks)
                val_loss += loss.item()

                
                preds_labels = torch.argmax(preds, dim=1)
                y_true_val.append(core_masks[valid_mask].cpu().numpy().ravel())
                y_pred_val.append(preds_labels[valid_mask].cpu().numpy().ravel())

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Compute val metrics
        y_true_flat = np.concatenate(y_true_val)
        y_pred_flat = np.concatenate(y_pred_val)
        val_oAccu, val_mIoU = calc_oAccu_mIoU(y_true_flat, y_pred_flat, num_classes)
        val_oAccus.append(val_oAccu)
        val_mIoUs.append(val_mIoU)

        log.info(f" {epoch + 1}/{epoch_num} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train oAcc: {train_oAccu:.4f}, mIoU: {train_mIoU:.4f} | "
              f"Val oAcc: {val_oAccu:.4f}, mIoU: {val_mIoU:.4f}")

        # ---------------------------------------------------------------------
        #  Update best model if the current validation loss is better
        # ---------------------------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

        early_stopper(val_loss)
        save_by_interval = (epoch + 1) % config['model_save_interval'] == 0
        save_due_to_early_stop = early_stopper.early_stop and stop_early

        if save_model and (save_by_interval or save_due_to_early_stop):
            model_name_prefix = f"{out_file_str}_epoch_{epoch+1:03d}_{channel_info_str}"
            save_model_locally(
                model=model,
                model_dir=model_dir,
                model_name_prefix=model_name_prefix,
                dummy_shape=dummy_shape
            )

        if save_due_to_early_stop:
            log.info("⏹ Early stopping triggered!")
            break

    # -------------------------------------------------------------------------
    # 4. After the loop: Optionally save the best model
    # -------------------------------------------------------------------------
    if save_model and best_model_state is not None:
        model.load_state_dict(best_model_state)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        model_name_prefix = f"{out_file_str}_best_{channel_info_str}_{timestr}"
        save_model_locally(
            model=model,
            model_dir=model_dir,
            model_name_prefix=model_name_prefix,
            dummy_shape=dummy_shape,
            save_onnx=save_onnx
        )
        log.info(f"✅ Best model saved with val_loss = {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 5. Save plots and return
    # -------------------------------------------------------------------------
    timestr = time.strftime("%Y%m%d_%H%M%S")
    train_loss_save_dir = train_out_root_dir / 'outputs' / out_file_str
    train_loss_save_dir.mkdir(parents=True, exist_ok=True)
    plt_save_path = train_loss_save_dir / f'losses_{channel_info_str}_{timestr}.png'
    plot_training_validation_losses(train_losses, val_losses, plt_save_path)

    metrics_save_path = train_loss_save_dir / f'metrics_{channel_info_str}_{timestr}.png'
    plot_training_validation_metrics(train_oAccus, val_oAccus, train_mIoUs, val_mIoUs, metrics_save_path)

    out_dict = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_oAccus': train_oAccus,
        'val_oAccus': val_oAccus,
        'train_mIoUs': train_mIoUs,
        'val_mIoUs': val_mIoUs
    }
    train_log_file = train_loss_save_dir / f'train_log_{channel_info_str}.yaml'
    dump_dict_to_yaml(out_dict, train_log_file)
    return out_dict


if __name__ == "__main__":
    log.info(pformat(CONFIG))
    input_channels_ls = CONFIG['input_channels_ls']
    ensemble_config = CONFIG['ensemble_config']
    pretrained_model_file = CONFIG['pretrained_model_file']
    save_onnx = CONFIG['save_onnx']
    train_subset_cnts = CONFIG.get('train_subset_cnts', [30])
    load_pretrain = True if pretrained_model_file else False
    for train_subset_cnt in train_subset_cnts:
        log.info(f"Training with train_subset_cnt: {train_subset_cnt}")
        for input_channels in input_channels_ls:
            for model_setup_dict in ensemble_config:
                log.info(f"Training model with input channels: {input_channels} \nand setup: {model_setup_dict}")
                state_dict = train_model(CONFIG, 
                                         train_subset_cnt=train_subset_cnt,
                                        input_channels=input_channels, 
                                        model_setup_dict=model_setup_dict,
                                        load_pretrain=load_pretrain,
                                        save_model=True, 
                                        save_onnx=save_onnx)
                
                torch.cuda.empty_cache()
                gc.collect()
