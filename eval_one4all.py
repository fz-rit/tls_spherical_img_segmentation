import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, depad_img_or_mask
from training_one4all import build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from tools.visualize_tools import visualize_eval_output
from tools.metrics_tools import calculate_segmentation_statistics
import time
import numpy as np
from tools.load_tools import dump_dict_to_yaml, CONFIG
from tools.logger_setup import Logger
import re

log = Logger()

def load_model(config: dict, input_channels:list, model_name: str, device: str) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    
    model_dir = Path(config['root_dir']) / config['model_dir'] / model_name
    channels_str = '_'.join([str(ch) for ch in input_channels])
    # pattern = re.compile(rf"^.*_avg_model_{channels_str}_\d{{8}}_\d{{6}}\.pth$")

    # model_file_ls = [f for f in model_dir.glob("*.pth") if pattern.search(f.name)]
    # model_file = model_file_ls[0] if len(model_file_ls) > 0 else None
    # model_file = next(model_dir.glob(f"*avg_model_{channels_str}_*.pth"), None)
    # model_file = model_dir / "unet_resnext50_32x4d_best_0_1_2_fold0_20250623_202135.pth"
    model_file = model_dir / "unet_resnext50_32x4d_best_0_1_2_fold3_20250623_204119.pth"
    # Load the model if there is a saved model, otherwise train a new model
    if model_file.exists():
        model = build_model_for_multi_channels(model_name=model_name,
                                           encoder_name=config['encoder_name'],
                                            in_channels=len(input_channels),
                                            num_classes=config['num_classes'])
        model.load_state_dict(torch.load(model_file, weights_only=True))
        log.info(f"======Loaded model from disk: {model_file.stem}.pth.======")
    else:
        raise FileNotFoundError(f"Model file {model_file} not found. Please train the model first.")
    
    model = model.to(device)

    return model




def evaluate_single_img(imgs, 
                        true_masks, 
                        config, 
                        input_channels: list,
                        model_name: str,
                        gt_available: bool,
                        output_path: Path,
                        show_now=False):
    """
    Evaluate the model on the test set.

    Args:
        config (dict): Configuration dictionary.
        device (str): Device to run the evaluation on.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = config['num_classes']
    input_size = config['input_size']
    patches_per_image = config['patches_per_image']
    model = load_model(config, input_channels, model_name, device)
    
    imgs = imgs.to(device)                  # (N, C, H, W)
    true_masks = true_masks.to(device)      # (N, H, W)


    # -----Evaluate model in normal mode.--------------
    log.info("🙂 Evaluating the model ...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        preds = model(imgs)                 # (N, Class, H, W)
        end_time = time.time()
        log.info(f"⏲️ Time taken for inference: {(end_time - start_time)*1e3:.2f} ms")
        pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

    imgs = imgs.cpu()
    if imgs.shape[1] >3:
        imgs = imgs[:, :3, :, :]
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()
    
    # -----Depad the images and masks.--------------
    # Depad the images and masks to the original size.
    input_patch_size = (input_size[0], input_size[1] // patches_per_image)
    imgs_reshaped = [depad_img_or_mask(img.permute(1, 2, 0).numpy(), input_patch_size) for img in imgs]
    true_masks = [depad_img_or_mask(mask, input_patch_size) for mask in true_masks]
    pred_masks = [depad_img_or_mask(mask, input_patch_size) for mask in pred_masks]


    # ----- Display the predicted masks. --------------
    combined_img = np.concatenate([img for img in imgs_reshaped], axis=1)  # shape: (H, W * N, 3)
    combined_true_mask = torch.cat([mask for mask in true_masks], dim=1)  # shape: (H, W * N)
    combined_pred_mask = torch.cat([mask for mask in pred_masks], dim=1)  # shape: (H, W * N)
    combined_true_mask = torch.zeros_like(combined_pred_mask) if not gt_available else combined_true_mask
    combined_true_mask = combined_true_mask.numpy()
    combined_pred_mask = combined_pred_mask.numpy()

    visualize_eval_output(combined_img, 
                          combined_true_mask, 
                          combined_pred_mask,
                          num_classes = num_classes,
                          input_channels = input_channels,
                          output_path = output_path,
                          gt_available = gt_available) 
    
    
    if show_now:
        plt.show()

    return combined_true_mask, combined_pred_mask
    


def evaluate_model(config: dict, input_channels: list, model_name: str = 'unet'):
    show_now = config['eval_imshow']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    true_mask_ls = []
    pred_mask_ls = []
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        log.info(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks = list(test_loader)[test_img_idx]

        # Prepare output paths.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem.split('_')[1][-4:] # the four numbers represent the test image dataset.
        out_dir = Path(config['root_dir']) / 'outputs' / model_name / key_str
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_img_output_path = out_dir / f'combined_output_{key_str}_{channels_str}_{timestamp}.png'
        combined_true_mask, combined_pred_mask = evaluate_single_img(imgs, 
                                                                     true_masks, 
                                                                     config, 
                                                                    input_channels,
                                                                    model_name,
                                                                eval_gt_available, 
                                                                eval_img_output_path, 
                                                                show_now=show_now)  
        
        true_mask_ls.append(combined_true_mask.flatten())
        pred_mask_ls.append(combined_pred_mask.flatten())

    # Calculate the overall accuracy and mean IoU for all images and write to a .yaml file.
    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)

    metric_dict = calculate_segmentation_statistics(true_mask, pred_mask, num_classes)
    output_file = out_dir.parent / f"eval_metrics_{channels_str}_{timestamp}.yaml"
    dump_dict_to_yaml(metric_dict, output_file)



def main():
    input_channels_ls = CONFIG['input_channels_ls']
    model_name_ls = CONFIG['model_name_ls']
    for model_name in model_name_ls:
        for input_channels in input_channels_ls:
            assert input_channels in CONFIG['input_channels_ls'], f"Input channel {input_channels} not found in the list of input channels."
            log.info(f"Input channels: {input_channels}")
            evaluate_model(CONFIG, input_channels, model_name)




if __name__ == '__main__':
    main()