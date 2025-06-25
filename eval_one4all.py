import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, depad_img_or_mask
from training_one4all import build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from tools.visualize_tools import visualize_eval_output, write_eval_metrics_to_file
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


def ensemble_predict(models, imgs):
    imgs = imgs.to("cuda")     
    P_all = []  # shape: [M models × T samples, B, C, H, W]
    for m in models:
        m.eval()
        with torch.no_grad():
            logits = m(imgs)              # [B, C, H, W]
            P_all.append(torch.softmax(logits, dim=1))
    P_all_stack = torch.stack(P_all)
    P_mean = P_all_stack.mean(dim=0)  # [B, C, H, W]
    P_var = P_all_stack.var(dim=0)    # [B, C, H, W]
    var_based_epistemic = P_var.sum(dim=1)  # [B, H, W], variance based epistemic uncertainty
    # Entropy of expected prediction 
    entropy = -torch.sum(P_mean * torch.log(P_mean + 1e-8), dim=1)  # [B, H, W]; total uncertainty

    # Expected entropy (average of individual model entropies)
    expected_entropy = -torch.stack([torch.sum(P_i * torch.log(P_i + 1e-8), dim=1) for P_i in P_all]).mean(dim=0)

    # Mutual information: Epistemic
    mutual_info = entropy - expected_entropy
    assert (P_mean.shape[0], P_mean.shape[2], P_mean.shape[3]) == entropy.shape == \
        var_based_epistemic.shape == mutual_info.shape, \
        f"Shapes mismatch: P_mean: {P_mean.shape}, entropy: {entropy.shape}, " \
        f"var_based_epistemic: {var_based_epistemic.shape}, mutual_info: {mutual_info.shape}"
    pred_dict = {"pred": P_mean.argmax(dim=1),  # shape: [B, H, W]
                "total_uncertainty": entropy,  # shape: [B, H, W]
                "var_based_epistemic": var_based_epistemic,  # shape: [B, H, W]
                "mutual_info": mutual_info}  # shape: [B, H, W]
    return pred_dict

def post_process_pred_batch(pred_batch: torch.Tensor, input_patch_size) -> np.ndarray:
    """ Post-process the predicted batch of images.
    Args:
        pred_batch (torch.Tensor): Batch of predicted images of shape (N, C, H, W).
        input_patch_size (tuple): Size of the input patches (height, width).
    Returns:
        np.ndarray: Combined image after depadding and concatenation.
    """

    pred_batch = pred_batch.cpu()
    depadded_tiles = [depad_img_or_mask(tile, input_patch_size) for tile in pred_batch]
    combined_tile = torch.cat([tile for tile in depadded_tiles], dim=1)
    combined_tile = combined_tile.numpy()

    return combined_tile

def evaluate_single_img(imgs, 
                        true_masks, 
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
    input_size = config['input_size']
    patches_per_image = config['patches_per_image']

    log.info(f"🤨 Evaluating the models and saving outputs in {out_dir}")
    pred_dict = ensemble_predict(ensamble_models, imgs)
    input_patch_size = (input_size[0], input_size[1] // patches_per_image)
    imgs = imgs.cpu()
    if imgs.shape[1] >3:
        imgs = imgs[:, :3, :, :]
    imgs_reshaped = [depad_img_or_mask(img.permute(1, 2, 0).numpy(), input_patch_size) for img in imgs]
    combined_img = np.concatenate([img for img in imgs_reshaped], axis=1)  # shape: (H, W * N, 3)
    
    combined_pred_mask = post_process_pred_batch(pred_dict['pred'], input_patch_size)
    combined_total_uncertainty = post_process_pred_batch(pred_dict['total_uncertainty'], input_patch_size)
    combined_var_based_epistemic = post_process_pred_batch(pred_dict['var_based_epistemic'], input_patch_size)
    combined_mutual_info = post_process_pred_batch(pred_dict['mutual_info'], input_patch_size)
    combined_true_mask = post_process_pred_batch(true_masks, input_patch_size) if \
        gt_available else np.zeros_like(combined_pred_mask)

    eval_results = {
        "true_mask": combined_true_mask,
        "pred_mask": combined_pred_mask,
        "total_uncertainty": combined_total_uncertainty,
        "var_based_epistemic": combined_var_based_epistemic,
        "mutual_info": combined_mutual_info
    }
    visualize_eval_output(combined_img,
                          eval_results,
                          num_classes = num_classes,
                          input_channels = input_channels,
                          out_dir = out_dir,
                          gt_available = gt_available) 
    
    if show_now:
        plt.show()

    return combined_true_mask, combined_pred_mask
    


def evaluate_imgs(config: dict, input_channels: list):
    show_now = config['eval_imshow']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    out_dir = Path(config['root_dir']) / 'outputs' / 'eval'
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), \
        "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    true_mask_ls = []
    pred_mask_ls = []
    ensamble_models = load_ensamble_models(config, input_channels)
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        log.info(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks = list(test_loader)[test_img_idx]

        key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem.split('_')[-3][-4:] # the four numbers represent the test image dataset.
        img_eval_out_dir = out_dir / key_str
        img_eval_out_dir.mkdir(parents=True, exist_ok=True)
        combined_true_mask, combined_pred_mask = evaluate_single_img(imgs, 
                                                                     true_masks, 
                                                                     ensamble_models,
                                                                     config, 
                                                                    input_channels,
                                                                    eval_gt_available, 
                                                                    img_eval_out_dir, 
                                                                    show_now=show_now)  
        
        true_mask_ls.append(combined_true_mask.flatten())
        pred_mask_ls.append(combined_pred_mask.flatten())

    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)
    eval_metrics_dict = calculate_segmentation_statistics(true_mask, pred_mask, num_classes)
    channels_str = '_'.join([str(ch) for ch in input_channels])
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