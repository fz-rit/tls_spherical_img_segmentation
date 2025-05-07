import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, resize_image_or_mask, NUM_CLASSES
from training_one4all import train_model, build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from monte_carlo_dropout import MonteCarloDropoutUncertainty
import numpy as np
from tools.visualize_tools import visualize_eval_output, compare_uncertainty_with_error_map
import time

INPUT_RESOLUTION = (540, 1440)  # (H, W)

def load_model(config: dict, device: str) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    
    model_dir = Path(config['root_dir']) / config['model_dir'] / config['model_name']
    model_file = model_dir / config['model_file']
    # Load the model if there is a saved model, otherwise train a new model
    if model_file.exists():
        model = build_model_for_multi_channels(model_name=config['model_name'],
                                           encoder_name=config['encoder_name'])
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"======Loaded model from disk: {model_file.stem}.======")
    else:
        out_dict = train_model(config)
        model = out_dict['model']
        print("####Trained a new model.####")
    
    model = model.to(device)

    return model




def evaluate(imgs, true_masks, config, gt_available, 
             output_paths: list[Path],
             show_now=False):
    """
    Evaluate the model on the test set.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to run the evaluation on.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(config, device)
    
    imgs = imgs.to(device)                  # (N, C, H, W)
    true_masks = true_masks.to(device)      # (N, H, W)

    # ---------Evaluate model in Monte Carlo Dropout mode and estimate uncertainty.------
    print("🔮Evaluating the model in Bayesian mode...")
    mcdu = MonteCarloDropoutUncertainty(model, imgs)
    mcdu.execute(mc_iterations=40, 
                mutual_information=True, 
                output_path=output_paths[1])
    uncertainty_map = mcdu.uncertainty_map

    # -----Evaluate model in normal mode.--------------
    print("🙂Evaluating the model in normal mode...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        preds = model(imgs)                 # (N, C, H, W)
        end_time = time.time()
        print(f"⏲️Time taken for inference: {(end_time - start_time)*1e3:.2f} ms")
        pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

    imgs = imgs.cpu()
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()
    

    # Images: (N, 3, H, W) -> concatenate along width
    combined_img = torch.cat([img for img in imgs], dim=2)  # shape: (3, H, W * N)
    # Masks: (N, H, W) -> concatenate along width
    combined_true_mask = torch.cat([mask for mask in true_masks], dim=1)  # shape: (H, W * N)
    combined_pred_mask = torch.cat([mask for mask in pred_masks], dim=1)  # shape: (H, W * N)
    combined_true_mask = torch.zeros_like(combined_pred_mask) if not gt_available else combined_true_mask
    combined_img = combined_img.permute(1, 2, 0).numpy()  #[C,H,W] to [H,W,C]
    combined_true_mask = combined_true_mask.numpy()
    combined_pred_mask = combined_pred_mask.numpy()

    img = resize_image_or_mask(combined_img, INPUT_RESOLUTION) 
    true_mask = resize_image_or_mask(combined_true_mask, INPUT_RESOLUTION)
    pred_mask = resize_image_or_mask(combined_pred_mask, INPUT_RESOLUTION)
    visualize_eval_output(img, 
                          true_mask, 
                          pred_mask,
                          output_path = output_paths[0],
                          gt_available = gt_available) 
    
    if gt_available:
        print("🔍Comparing uncertainty map with error map...")
        error_map = np.zeros_like(true_mask)
        error_map[true_mask != pred_mask] = 1
        metrics_dict = compare_uncertainty_with_error_map(uncertainty_map, error_map, output_path=output_paths[2])
        # print(f"Metrics comparing uncertainty map with error map: {metrics_dict}")
    
    if show_now:
        plt.show()

    

def main():
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    _, _, test_loader = load_data(config)
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        print(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks = list(test_loader)[test_img_idx]
        assert imgs.shape[0] == true_masks.shape[0], f"{imgs.shape} vs {true_masks.shape}❗The number of images and masks should be the same."

        # Prepare output paths.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        key_str = str(test_loader.dataset.image_file_paths[test_img_idx].stem).split('_')[1][-4:] # the four numbers represent the test image dataset.
        out_dir = Path(config['root_dir']) / 'outputs' / config['model_name'] / key_str
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path_1 = out_dir / f'combined_output_{key_str}_{timestamp}.png'
        output_path_2 = out_dir / f'uncertainty_map_{key_str}_{timestamp}.png'
        output_path_3 = out_dir / f'uncertainty_vs_error_{key_str}_{timestamp}.png'
        output_paths = [output_path_1, output_path_2, output_path_3]
        evaluate(imgs, true_masks, config, eval_gt_available, output_paths, show_now=False)  

if __name__ == '__main__':
    main()