import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, depad_tensor_vertical_only
# from training_one4all import build_model_for_multi_channels
from tools.feature_fusion_helper import build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from tools.visualize_tools import visualize_eval_output, write_eval_metrics_to_file, compare_uncertainty_with_error_map
from tools.metrics_tools import calculate_segmentation_statistics
import time
import numpy as np
from tools.load_tools import CONFIG
from tools.logger_setup import Logger

log = Logger()

def load_ensamble_models(config: dict, input_channels:list) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    ensemble_config = CONFIG['ensemble_config']
    models = []
    for model_setup_dict in ensemble_config:
        model_parent_dir = model_setup_dict['name']
        model_name = model_setup_dict['arch']
        encoder_name = model_setup_dict['encoder']
        model_dir = Path(config['root_dir']) / config['model_dir'] / model_parent_dir
        channels_str = '_'.join([str(ch) for ch in input_channels])
        model_file = next(model_dir.glob(f"*best_{channels_str}_*.pth"), None)
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


def ensemble_predict(models, imgs, buf_masks):
    imgs = imgs.to("cuda")
    buf_masks = buf_masks.to("cuda")  # [B, 1, H, W]
    mask = buf_masks.squeeze(1)       # [B, H, W]

    P_all = []  # [M, B, C, H, W]

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(imgs)                   # [B, C, H, W]
            probs = torch.softmax(logits, dim=1)   # [B, C, H, W]
            P_all.append(probs)

    P_all_stack = torch.stack(P_all)               # [M, B, C, H, W]
    P_mean = P_all_stack.mean(dim=0)               # [B, C, H, W]
    P_var = P_all_stack.var(dim=0)                 # [B, C, H, W]

    # Uncertainty maps
    var_based_epistemic = P_var.sum(dim=1)         # [B, H, W]
    entropy = -torch.sum(P_mean * torch.log(P_mean + 1e-8), dim=1)  # [B, H, W]
    expected_entropy = -torch.stack([
        torch.sum(P_i * torch.log(P_i + 1e-8), dim=1) for P_i in P_all
    ]).mean(dim=0)                                 # [B, H, W]
    mutual_info = entropy - expected_entropy       # [B, H, W]

    # Apply buffer mask to outputs
    P_mean = P_mean * mask.unsqueeze(1)                 # [B, C, H, W]
    var_based_epistemic = var_based_epistemic * mask    # [B, H, W]
    entropy = entropy * mask                            # [B, H, W]
    mutual_info = mutual_info * mask                    # [B, H, W]

    assert (P_mean.shape[0], P_mean.shape[2], P_mean.shape[3]) == entropy.shape == \
        var_based_epistemic.shape == mutual_info.shape, \
        f"Shapes mismatch: P_mean: {P_mean.shape}, entropy: {entropy.shape}, " \
        f"var_based_epistemic: {var_based_epistemic.shape}, mutual_info: {mutual_info.shape}"

    pred_dict = {
        "pred": P_mean.argmax(dim=1),                  # [B, H, W]
        "total_uncertainty": entropy,                  # [B, H, W]
        "var_based_epistemic": var_based_epistemic,    # [B, H, W]
        "mutual_info": mutual_info                     # [B, H, W]
    }

    return pred_dict


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



def evaluate_single_img(img_tiles, 
                        true_masks, 
                        buf_masks,
                        ensamble_models,
                        config, 
                        input_channels: list,
                        gt_available: bool,
                        out_dir: Path,
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
    pred_dict = ensemble_predict(ensamble_models, img_tiles, buf_masks)
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
    visualize_eval_output(stitched_img,
                          eval_results,
                          num_classes = num_classes,
                          input_channels = input_channels,
                          out_dir = out_dir,
                          gt_available = gt_available) 
    
    if show_now:
        plt.show()

    return eval_results
    


def evaluate_imgs(config: dict, input_channels: list):
    show_now = config['eval_imshow']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    out_dir = Path(config['root_dir']) / 'outputs' / f'eval_{channels_str}'
    out_dir.mkdir(parents=True, exist_ok=True)
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), \
        "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    true_mask_ls = []
    pred_mask_ls = []
    ensamble_models = load_ensamble_models(config, input_channels)
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        log.info(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks, buf_masks = list(test_loader)[test_img_idx]
        key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem.split('_')[-3][-4:] # the four numbers represent the test image dataset.
        img_eval_out_dir = out_dir / key_str
        img_eval_out_dir.mkdir(parents=False, exist_ok=True)
        eval_results = evaluate_single_img(imgs, 
                                             true_masks, 
                                             buf_masks,
                                             ensamble_models,
                                             config, 
                                             input_channels,
                                            eval_gt_available, 
                                            img_eval_out_dir, 
                                            show_now=show_now)
        compare_uncertainty_with_error_map(eval_results, output_dir=img_eval_out_dir)
        true_mask_ls.append(eval_results['true_mask'].flatten())
        pred_mask_ls.append(eval_results['pred_mask'].flatten())

    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)
    eval_metrics_dict = calculate_segmentation_statistics(true_mask, pred_mask, num_classes)
    
    write_eval_metrics_to_file(eval_metrics_dict, out_dir, key_str=channels_str)



def main():
    input_channels_ls = CONFIG['input_channels_ls']
    for input_channels in input_channels_ls:
        assert input_channels in CONFIG['input_channels_ls'], \
            f"Input channel {input_channels} not found in the list of input channels."
        log.info(f"Input channels: {input_channels}")
        evaluate_imgs(CONFIG, input_channels)


if __name__ == '__main__':
    main()