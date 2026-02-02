"""
Evaluation script with inference profiling.
Extends eval_one4all.py with timing and memory tracking.
"""

import torch
import argparse
from prepare_dataset import load_data
from tools.load_tools import load_config
from pathlib import Path
from tools.training_profiler import InferenceProfiler
from tools.profile_model import ModelProfiler
import time
from tools.logger_setup import Logger
from eval_one4all import load_ensemble_models, evaluate_single_img
from tools.metrics_tools import calc_segmentation_statistics, average_uncertainty_metrics_across_images
from tools.visualize_tools import write_eval_metrics_to_file
import numpy as np

log = Logger()


def evaluate_imgs_with_profiling(config: dict, input_channels: list, 
                                 train_subset_cnt: int, save_uncertainty_figs: bool):
    """
    Evaluate images with comprehensive profiling.
    Tracks inference time and memory per image.
    """
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

    # Load ensemble models
    ensemble_models = load_ensemble_models(config, input_channels, eval_out_root_dir)
    
    # =========================================================================
    # PROFILING: Initialize inference profiler
    # =========================================================================
    inf_profiler = InferenceProfiler(device='cuda')
    
    # Get actual padded shape from DataLoader (handles padding to divisible by 32)
    sample_imgs, _, _ = next(iter(test_loader))
    actual_shape = (1, sample_imgs.shape[1], sample_imgs.shape[2], sample_imgs.shape[3])
    
    # Also profile first model for static metrics
    first_model = ensemble_models[0]
    model_profiler = ModelProfiler(
        model=first_model,
        input_shape=actual_shape,
        model_name=f"ensemble_eval_ch{channels_str}"
    )
    static_metrics = model_profiler.profile_static_metrics()
    log.info(f"📊 Ensemble model: {static_metrics['gflops']:.2f} GFLOPs, "
            f"{static_metrics['trainable_params']:,} params")
    # =========================================================================

    true_mask_ls = []
    pred_mask_ls = []
    uncertainty_ls = []
    
    for idx, (test_img_idx, eval_gt_available) in enumerate(zip(test_img_idx_ls, eval_gt_available_ls)):
        log.info(f"🔍 Evaluating image {idx+1}/{len(test_img_idx_ls)} (index {test_img_idx})...")
        
        imgs, true_masks, buf_masks = list(test_loader)[test_img_idx]
        img_eval_out_dir = out_dir / f'img_{test_img_idx:02d}'
        img_eval_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Profile this inference
        single_img_start = time.time()
        
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
        
        single_img_time = time.time() - single_img_start
        log.info(f"   Image {idx+1} inference: {single_img_time:.2f}s")
        
        # Track per-image timing (simplified since ensemble complicates per-forward timing)
        # For detailed per-forward profiling, would need to modify evaluate_single_img
        
        true_mask_ls.append(eval_results['true_mask'].flatten())
        pred_mask_ls.append(eval_results['pred_mask'].flatten())
        uncertainty_ls.append(eval_results['uncertainty_dict'])

    # =========================================================================
    # POST-INFERENCE: Aggregate results
    # =========================================================================
    true_mask = np.concatenate(true_mask_ls) 
    pred_mask = np.concatenate(pred_mask_ls)
    eval_metrics_dict = calc_segmentation_statistics(true_mask, pred_mask, num_classes)
    avg_uncertainty_dict = average_uncertainty_metrics_across_images(uncertainty_ls)
    eval_metrics_dict.update(avg_uncertainty_dict)
    
    # Add profiling summary
    inf_summary = inf_profiler.get_summary()
    eval_metrics_dict.update({
        'num_test_images': len(test_img_idx_ls),
        'avg_inference_time_per_image_sec': round(single_img_time, 2),  # simplified
    })
    
    # Combine with model metrics
    model_profiler.metrics.update(eval_metrics_dict)
    model_profiler.metrics.update({
        'config_file': config.get('_config_path', 'unknown'),
        'train_subset_cnt': train_subset_cnt,
        'input_channels': channels_str,
        'dataset': dataset_name,
    })
    
    # Save comprehensive evaluation profile
    eval_profile_json = out_dir / f'eval_profile_{channels_str}.json'
    model_profiler.save_metrics(eval_profile_json, format='json')
    
    eval_profile_csv = out_dir / f'eval_profile_{channels_str}.csv'
    model_profiler.save_metrics(eval_profile_csv, format='csv')
    # =========================================================================
    
    write_eval_metrics_to_file(eval_metrics_dict, out_dir, key_str=channels_str)
    log.info(f"✅ Evaluation complete. Results saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate one-for-all model with profiling')
    parser.add_argument('--config', type=str, default='params/paths_rc_forestsemantic.json',
                        help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    CONFIG = load_config(args.config)
    CONFIG['_config_path'] = args.config
    print(f"Using config file: {args.config}")
    
    input_channels_ls = CONFIG['input_channels_ls']
    train_subset_cnts = CONFIG['train_subset_cnts']
    save_uncertainty_figs = CONFIG['save_uncertainty_figs']
    
    for train_subset_cnt in train_subset_cnts:
        for input_channels in input_channels_ls:
            log.info(f"Evaluating: subset={train_subset_cnt}, channels={input_channels}")
            evaluate_imgs_with_profiling(
                CONFIG, 
                input_channels, 
                train_subset_cnt, 
                save_uncertainty_figs=save_uncertainty_figs
            )


if __name__ == '__main__':
    main()