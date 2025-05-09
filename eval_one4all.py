import torch
import matplotlib.pyplot as plt
from prepare_dataset import load_data, NUM_CLASSES, depad_img_or_mask, PATCH_PER_IMAGE
from training_one4all import train_model, build_model_for_multi_channels
import json
from pathlib import Path
import segmentation_models_pytorch as smp
import datetime
from tools.visualize_tools import visualize_eval_output
from tools.metrics_tools import calculate_segmentation_statistics
import time
import numpy as np
from tools.load_tools import dump_dict_to_yaml

INPUT_RESOLUTION = (540, 1440)  # (H, W)
N_CLASSES = 6

def load_model(config: dict, input_channels:list, device: str) -> smp.Unet:
    """
    Load the trained model.

    Args:
    config (dict): Configuration dictionary.

    Returns:
    model (smp.Unet): Trained model.
    """
    
    model_dir = Path(config['root_dir']) / config['model_dir'] / config['model_name']
    channels_str = '_'.join([str(ch) for ch in input_channels])
    model_file_ls = [model_dir / model_file_name for model_file_name in config['model_file_ls'] if channels_str in model_file_name]
    model_file = model_file_ls[0] if len(model_file_ls) > 0 else None
    # Load the model if there is a saved model, otherwise train a new model
    if model_file.exists():
        model = build_model_for_multi_channels(model_name=config['model_name'],
                                           encoder_name=config['encoder_name'],
                                            in_channels=len(input_channels))
        model.load_state_dict(torch.load(model_file, weights_only=True))
        print(f"======Loaded model from disk: {model_file.stem}.======")
    else:
        # out_dict = train_model(config)
        # model = out_dict['model']
        # print("####Trained a new model.####")
        raise FileNotFoundError(f"Model file {model_file} not found. Please train the model first.")
    
    model = model.to(device)

    return model




def evaluate_single_img(imgs, true_masks, config, 
             input_channels: list,
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
    model = load_model(config, input_channels, device)
    
    imgs = imgs.to(device)                  # (N, C, H, W)
    true_masks = true_masks.to(device)      # (N, H, W)


    # -----Evaluate model in normal mode.--------------
    print("🙂 Evaluating the model ...")
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        preds = model(imgs)                 # (N, Class, H, W)
        end_time = time.time()
        print(f"⏲️ Time taken for inference: {(end_time - start_time)*1e3:.2f} ms")
        pred_masks = torch.argmax(preds, dim=1)  # (N, H, W)

    imgs = imgs.cpu()
    if imgs.shape[1] >3:
        imgs = imgs[:, :3, :, :]
    true_masks = true_masks.cpu()
    pred_masks = pred_masks.cpu()
    
    # -----Depad the images and masks.--------------
    # Depad the images and masks to the original size.
    input_patch_size = (INPUT_RESOLUTION[0], INPUT_RESOLUTION[1] // PATCH_PER_IMAGE)
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
                          output_path = output_path,
                          gt_available = gt_available) 
    
    
    if show_now:
        plt.show()

    return combined_true_mask, combined_pred_mask
    


def evaluate_model(config: dict, input_channels: list):
    show_now = config['eval_imshow']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    true_mask_ls = []
    pred_mask_ls = []
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        print(f"🔍Evaluating image {test_img_idx}...")
        imgs, true_masks = list(test_loader)[test_img_idx]

        # Prepare output paths.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        key_str = Path(test_loader.dataset.image_file_paths[test_img_idx]).stem.split('_')[1][-4:] # the four numbers represent the test image dataset.
        out_dir = Path(config['root_dir']) / 'outputs' / config['model_name'] / key_str
        out_dir.mkdir(parents=True, exist_ok=True)
        eval_img_output_path = out_dir / f'combined_output_{key_str}_{channels_str}_{timestamp}.png'
        combined_true_mask, combined_pred_mask = evaluate_single_img(imgs, true_masks, config, 
                                                            input_channels,
                                                          eval_gt_available, 
                                                          eval_img_output_path, 
                                                          show_now=show_now)  
        
        true_mask_ls.append(combined_true_mask.flatten())
        pred_mask_ls.append(combined_pred_mask.flatten())

    # Calculate the overall accuracy and mean IoU for all images and write to a .yaml file.
    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)

    metric_dict = calculate_segmentation_statistics(true_mask, pred_mask, N_CLASSES)
    output_file = out_dir.parent / f"eval_metrics_{timestamp}.yaml"
    dump_dict_to_yaml(metric_dict, output_file)



def main():
    config_file = 'params/paths_zmachine.json'
    with open(config_file, 'r') as f:
        config = json.load(f)

    input_channels_ls = config['input_channels_ls']
    for input_channels in input_channels_ls:
        assert input_channels in config['input_channels_ls'], f"Input channel {input_channels} not found in the list of input channels."
        print(f"Input channels: {input_channels}")
        evaluate_model(config, input_channels)




if __name__ == '__main__':
    main()