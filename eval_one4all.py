import torch
import argparse
import matplotlib.pyplot as plt
from prepare_dataset import load_data, depad_tensor_vertical_only
from tools.load_tools import load_config
from tools.feature_fusion_helper import build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from tools.visualize_tools import visualize_eval_output, write_eval_metrics_to_file, compare_uncertainty_with_error_map
from tools.metrics_tools import calc_segmentation_statistics, average_uncertainty_metrics_across_images
import time
import numpy as np
from tools.logger_setup import Logger

log = Logger()

def load_ensemble_models(config: dict, input_channels:list, eval_out_root_dir: Path) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    ensemble_config = config['ensemble_config']
    models = []
    for model_setup_dict in ensemble_config:
        model_parent_dir = model_setup_dict['name']
        model_name = model_setup_dict['arch']
        encoder_name = model_setup_dict['encoder']
        model_dir = eval_out_root_dir / config['model_dir'] / model_parent_dir
        channels_str = '_'.join([str(ch) for ch in input_channels])
        pattern = f"*best_{channels_str}_????????_??????.pth"
        model_file = next(model_dir.glob(pattern), None)
        if model_file.exists():
            model = build_model_for_multi_channels(model_name=model_name,
                                            encoder_name=encoder_name,
                                                in_channels=len(input_channels),
                                                num_classes=config['num_classes'])
            model.load_state_dict(torch.load(model_file, weights_only=True))
            log.info(f"======Loaded model from disk: {model_file.name}======")
        else:
            raise FileNotFoundError(f"Model file {model_file} not found. Please train the model first.")
        
        model = model.to("cuda")
        models.append(model)
    log.info(f"🌘 Loaded {len(models)} ensemble models.")
    return models


def ensemble_predict_with_uncertainty(models, imgs, buf_masks=None):
    """
    Compute ensemble prediction and uncertainty using logits averaging.

    Args:
        models (list of nn.Module): Trained models.
        imgs (torch.Tensor): Input images [B, C, H, W].
        buf_masks (torch.Tensor or None): binary mask from buffer zone [B, H, W].

    Returns:
        dict with:
            - 'pred': final predicted class indices [B, H, W]
            - 'total_uncertainty': entropy of mean prediction [B, H, W]
            - 'mutual_info': epistemic uncertainty [B, H, W]
            - 'var_based_epistemic': variance across softmax outputs [B, H, W]
    """
    imgs = imgs.to("cuda")
    buf_masks = buf_masks.to("cuda")  # [B, 1, H, W]
    Z_all = []  # raw logits

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(imgs)  # [B, C, H, W]
            Z_all.append(logits)

    Z_stack = torch.stack(Z_all)                     # [M, B, C, H, W]
    Z_mean = Z_stack.mean(dim=0)                     # [B, C, H, W]
    P_mean = torch.softmax(Z_mean, dim=1)            # [B, C, H, W]

    # Compute softmax for each model (for uncertainty)
    P_all = torch.softmax(Z_stack, dim=2)            # [M, B, C, H, W]
    P_var = P_all.var(dim=0)                         # [B, C, H, W]

    # --- Uncertainty Measures ---
    entropy = -torch.sum(P_mean * torch.log(P_mean + 1e-8), dim=1)  # [B, H, W]
    expected_entropy = -torch.mean(torch.sum(P_all * torch.log(P_all + 1e-8), dim=2), dim=0)  # [B, H, W]
    mutual_info = entropy - expected_entropy                        # [B, H, W]
    var_based_epistemic = P_var.sum(dim=1)                          # [B, H, W]

    # --- Apply optional mask ---
    if buf_masks is not None:
        # Check if buf_masks needs dimension adjustment
        if buf_masks.dim() == 3:  # [B, H, W]
            buf_mask_for_pmean = buf_masks.unsqueeze(1)  # [B, 1, H, W]
            buf_mask_for_uncertainty = buf_masks  # [B, H, W]
        elif buf_masks.dim() == 4:  # [B, 1, H, W]
            buf_mask_for_pmean = buf_masks  # [B, 1, H, W]
            buf_mask_for_uncertainty = buf_masks.squeeze(1)  # [B, H, W]
        else:
            raise ValueError(f"Unexpected buf_masks shape: {buf_masks.shape}")
            
        P_mean *= buf_mask_for_pmean
        entropy *= buf_mask_for_uncertainty
        mutual_info *= buf_mask_for_uncertainty
        var_based_epistemic *= buf_mask_for_uncertainty

    return {
        "pred": P_mean.argmax(dim=1),                  # [B, H, W]
        "total_uncertainty": entropy,                  # [B, H, W]
        "var_based_epistemic": var_based_epistemic,    # [B, H, W]
        "mutual_info": mutual_info                     # [B, H, W]
    }


def post_process_pred_batch(pred_batch: torch.Tensor, input_patch_h, buf_mask_ls) -> np.ndarray:
    """ Post-process the predicted batch of images.
    Args:
        pred_batch (torch.Tensor): Batch of predicted images of shape (N, C, H, W).
        input_patch_size (tuple): Size of the input patches (height, width).
    Returns:
        np.ndarray: Combined image after depadding and concatenation.
    """
    depad_img_tiles = [depad_tensor_vertical_only(img, original_height=input_patch_h) for img in pred_batch]
    stitched_img = stitch_buffered_tiles(torch.stack(depad_img_tiles), buf_mask_ls)

    return stitched_img

def stitch_buffered_tiles(tiles: torch.Tensor, buf_mask_ls: list[torch.Tensor]) -> np.ndarray:
    """
    Stitch tiles along W axis using buffer masks.
    
    Args:
        tiles: Tensor of shape [B, C, H, W] or [B, H, W]
        buf_masks: Tensor of shape [B, 1, H, W] or [B, H, W]
        
    Returns:
        full: stitched tensor of shape [C, H, W_total] or [H, W_total]
    """
    # Make sure dimensions are aligned
    if tiles.dim() == 4:  # [B, C, H, W]
        is_image = True
        _, C, H, W = tiles.shape
    elif tiles.dim() == 3:  # [B, H, W]
        is_image = False
        C, H, W = 1, *tiles.shape[1:]
        tiles = tiles.unsqueeze(1)  # [B, 1, H, W]
    else:
        raise ValueError("Expected 3D or 4D tiles tensor")

    # Extract valid region from each tile
    stitched_pieces = []
    for tile, mask in zip(tiles, buf_mask_ls):  # tile: [C, H, W], mask: [1, H, W]
        valid = mask.bool().expand_as(tile)   # [C, H, W]
        stitched_tile = tile[valid].reshape(C, H, -1)  # remove invalid cols, keep row structure
        stitched_pieces.append(stitched_tile)

    # Concatenate along width (dim=-1)
    full = torch.cat(stitched_pieces, dim=-1)  # [C, H, W_total]

    if not is_image:
        full = full.squeeze(0)  # [H, W_total] for label/pred map

    
    return full.cpu().numpy()



def evaluate_single_model(model, img_tiles, true_masks, buf_masks, config, 
                          input_channels: list, gt_available: bool, out_dir: Path,
                          show_now=False):
    """
    Evaluate a single model on image tiles (no uncertainty).
    
    Args:
        model: Single trained model
        img_tiles: Input images [B, C, H, W]
        true_masks: Ground truth masks [B, H, W]
        buf_masks: Buffer masks [B, H, W] or [B, 1, H, W]
        config: Configuration dictionary
        input_channels: List of input channels
        gt_available: Whether ground truth is available
        out_dir: Output directory
        show_now: Whether to show plots
    
    Returns:
        dict with pred_mask, true_mask, and evaluation results
    """
    num_classes = config['num_classes']
    input_h = config['input_size'][0]
    
    model.eval()
    img_tiles_cuda = img_tiles.to("cuda")
    buf_masks_cuda = buf_masks.to("cuda")
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    with torch.no_grad():
        logits = model(img_tiles_cuda)  # [B, C, H, W]
        pred = logits.argmax(dim=1)  # [B, H, W]
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    forward_time = time.time() - start_time
    
    img_tiles = img_tiles.cpu()
    pred = pred.cpu()
    
    if img_tiles.shape[1] > 3:
        img_tiles = img_tiles[:, :3, :, :]
    
    # Depad before stitching
    depad_img_tiles = [depad_tensor_vertical_only(img, original_height=input_h) for img in img_tiles]
    depat_buf_masks = [depad_tensor_vertical_only(mask, original_height=input_h) for mask in buf_masks]
    stitched_img = stitch_buffered_tiles(torch.stack(depad_img_tiles), depat_buf_masks)
    stitched_img = stitched_img.transpose(1, 2, 0)  # [H, W, C]
    
    stitched_pred_mask = post_process_pred_batch(pred, input_h, buf_mask_ls=depat_buf_masks)
    stitched_true_mask = post_process_pred_batch(true_masks, input_h, buf_mask_ls=depat_buf_masks) if \
        gt_available else np.zeros_like(stitched_pred_mask)
    
    eval_results = {
        "true_mask": stitched_true_mask,
        "pred_mask": stitched_pred_mask,
        "uncertainty_dict": {}  # Empty dict for individual model (no uncertainty)
    }
    
    visualize_eval_output(stitched_img,
                          eval_results,
                          num_classes=num_classes,
                          input_channels=input_channels,
                          config=config,
                          out_dir=out_dir,
                          gt_available=gt_available)
    
    if show_now:
        plt.show()
    
    return eval_results, forward_time


def evaluate_single_img(img_tiles, 
                        true_masks, 
                        buf_masks,
                        ensamble_models,
                        config, 
                        input_channels: list,
                        gt_available: bool,
                        out_dir: Path,
                        save_uncertainty_figs=True,
                        show_now=False):
    """
    Evaluate the model on the test set.

    Args:
    imgs (torch.Tensor): Input images of shape (N, C, H, W).
    true_masks (torch.Tensor): Ground truth masks of shape (N, H, W).
    config (dict): Configuration dictionary.
    input_channels (list): List of input channels.
    model_name (str): Name of the model.
    gt_available (bool): Whether ground truth masks are available.
    output_path (Path): Path to save the output image.
    show_now (bool): Whether to show the output image now.
    """
    num_classes = config['num_classes']
    input_h = config['input_size'][0]

    log.info(f"🤨 Evaluating the models and saving outputs in {out_dir}")
    # pred_dict = ensemble_predict(ensamble_models, img_tiles, buf_masks)
    pred_dict = ensemble_predict_with_uncertainty(ensamble_models, img_tiles, buf_masks)
    img_tiles = img_tiles.cpu()
    if img_tiles.shape[1] >3:
        img_tiles = img_tiles[:, :3, :, :]

    # depad before stitching
    depad_img_tiles = [depad_tensor_vertical_only(img, original_height=input_h) for img in img_tiles]
    depat_buf_masks = [depad_tensor_vertical_only(mask, original_height=input_h) for mask in buf_masks]
    stitched_img = stitch_buffered_tiles(torch.stack(depad_img_tiles), depat_buf_masks)
    stitched_img = stitched_img.transpose(1, 2, 0)  # [H, W, C] for visualization
    
    stitched_pred_mask = post_process_pred_batch(pred_dict['pred'], input_h, buf_mask_ls=depat_buf_masks)
    stitched_total_uncertainty = post_process_pred_batch(pred_dict['total_uncertainty'], input_h, buf_mask_ls=depat_buf_masks)
    stitched_var_based_epistemic = post_process_pred_batch(pred_dict['var_based_epistemic'], input_h, buf_mask_ls=depat_buf_masks)
    stitched_mutual_info = post_process_pred_batch(pred_dict['mutual_info'], input_h, buf_mask_ls=depat_buf_masks)
    stitched_true_mask = post_process_pred_batch(true_masks, input_h, buf_mask_ls=depat_buf_masks) if \
        gt_available else np.zeros_like(stitched_pred_mask)

    eval_results = {
        "true_mask": stitched_true_mask,
        "pred_mask": stitched_pred_mask,
        "total_uncertainty": stitched_total_uncertainty,
        "var_based_epistemic": stitched_var_based_epistemic,
        "mutual_info": stitched_mutual_info
    }
    _, uncertainty_dict = compare_uncertainty_with_error_map(eval_results, 
                                                                 output_dir=out_dir, 
                                                                 savefigs=save_uncertainty_figs)
    eval_results["uncertainty_dict"] = uncertainty_dict
    visualize_eval_output(stitched_img,
                          eval_results,
                          num_classes = num_classes,
                          input_channels = input_channels,
                          config = config,
                          out_dir = out_dir,
                          gt_available = gt_available) 
    
    if show_now:
        plt.show()

    return eval_results
    


def evaluate_imgs(config: dict, input_channels: list, train_subset_cnt: int, save_uncertainty_figs: bool):
    show_now = config['eval_imshow']
    dataset_name = config['dataset_name']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    eval_out_root_dir = Path(config['root_dir']) / f"run_subset_{train_subset_cnt:02d}"
    out_dir = eval_out_root_dir / 'outputs' / f'eval_{channels_str}'
    out_dir.mkdir(parents=True, exist_ok=True)
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), \
        "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    true_mask_ls = []
    pred_mask_ls = []
    ensemble_models = load_ensemble_models(config, input_channels, eval_out_root_dir)
    uncertainty_ls = []
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        log.info(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks, buf_masks = list(test_loader)[test_img_idx]
        if "mangrove" in dataset_name.lower():
            key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem.split('_')[-3][-4:] # the four numbers represent the test image dataset.
        elif "semantic3d" in dataset_name.lower() or "forestsemantic" in dataset_name.lower():
            key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}; Double check the key_str extraction logic in evaluate_imgs()")
        img_eval_out_dir = out_dir / key_str
        img_eval_out_dir.mkdir(parents=False, exist_ok=True)
        eval_results = evaluate_single_img(imgs, 
                                             true_masks, 
                                             buf_masks,
                                             ensemble_models,
                                             config, 
                                             input_channels,
                                            eval_gt_available, 
                                            img_eval_out_dir, 
                                            save_uncertainty_figs=save_uncertainty_figs,
                                            show_now=show_now)

        true_mask_ls.append(eval_results['true_mask'].flatten())
        pred_mask_ls.append(eval_results['pred_mask'].flatten())
        uncertainty_ls.append(eval_results['uncertainty_dict'])

    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)
    eval_metrics_dict = calc_segmentation_statistics(true_mask, pred_mask, num_classes)
    avg_uncertainty_dict = average_uncertainty_metrics_across_images(uncertainty_ls)
    eval_metrics_dict.update(avg_uncertainty_dict)
    write_eval_metrics_to_file(eval_metrics_dict, out_dir, key_str=channels_str)



def main():
    parser = argparse.ArgumentParser(description='Evaluate one-for-all model')
    parser.add_argument('--config', type=str, default='params/paths_rc_forestsemantic.json',
                        help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    CONFIG = load_config(args.config)
    print(f"Using config file: {args.config}")
    
    input_channels_ls = CONFIG['input_channels_ls']
    train_subset_cnts = CONFIG['train_subset_cnts']
    save_uncertainty_figs = CONFIG['save_uncertainty_figs']
    for train_subset_cnt in train_subset_cnts:
        for input_channels in input_channels_ls:
            log.info(f"Input channels: {input_channels}")
            evaluate_imgs(CONFIG, input_channels, train_subset_cnt, save_uncertainty_figs=save_uncertainty_figs)


if __name__ == '__main__':
    main()