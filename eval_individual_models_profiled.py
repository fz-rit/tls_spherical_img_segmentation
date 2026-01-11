"""
Evaluation script for individual ensemble models.
Evaluates each model separately (without ensemble/uncertainty).
Profiles inference time and memory, then compares against ensemble.
"""

import torch
import argparse
from prepare_dataset import load_data, depad_tensor_vertical_only
from tools.load_tools import load_config
from tools.feature_fusion_helper import build_model_for_multi_channels
from pathlib import Path
from tools.training_profiler import InferenceProfiler
from tools.profile_model import ModelProfiler
import time
from tools.logger_setup import Logger
from eval_one4all import load_ensemble_models, evaluate_single_model
from tools.metrics_tools import calc_segmentation_statistics
from tools.visualize_tools import write_eval_metrics_to_file
import numpy as np
import json

log = Logger()


def evaluate_individual_models_with_profiling(config: dict, input_channels: list, 
                                             train_subset_cnt: int):
    """
    Evaluate individual models from ensemble separately.
    No uncertainty measures, just predictions and metrics.
    Profiles each model's inference time and memory.
    """
    show_now = config['eval_imshow']
    dataset_name = config['dataset_name']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    eval_out_root_dir = Path(config['root_dir']) / f"run_subset_{train_subset_cnt:02d}"
    base_out_dir = eval_out_root_dir / 'outputs' / f'eval_{channels_str}_individual'
    base_out_dir.mkdir(parents=True, exist_ok=True)
    
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    assert len(test_img_idx_ls) == len(eval_gt_available_ls), \
        "The lengths of test_img_idx_ls and eval_gt_available_ls must be the same."

    # Load individual models
    ensemble_models = load_ensemble_models(config, input_channels, eval_out_root_dir)
    ensemble_config = config['ensemble_config']
    
    log.info(f"🎯 Evaluating {len(ensemble_models)} individual models")
    
    # Get actual padded shape from DataLoader (handles padding to divisible by 32)
    sample_imgs, _, _ = next(iter(test_loader))
    actual_shape = (1, sample_imgs.shape[1], sample_imgs.shape[2], sample_imgs.shape[3])
    
    # Dictionary to store results for each model
    individual_results = {}
    
    for model_idx, (model, model_config) in enumerate(zip(ensemble_models, ensemble_config)):
        model_name = model_config['name']
        arch_name = model_config['arch']
        log.info(f"\n{'='*60}")
        log.info(f"📊 Evaluating Model {model_idx + 1}/{len(ensemble_models)}: {model_name} ({arch_name})")
        log.info(f"{'='*60}")
        
        # Create output directory for this model
        model_out_dir = base_out_dir / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize profiler for this model
        inf_profiler = InferenceProfiler(device='cuda')
        model_profiler = ModelProfiler(
            model=model,
            input_shape=actual_shape,
            model_name=f"{model_name}_{arch_name}"
        )
        static_metrics = model_profiler.profile_static_metrics()
        log.info(f"   GFLOPs: {static_metrics['gflops']:.2f}, Params: {static_metrics['trainable_params']:,}")
        
        true_mask_ls = []
        pred_mask_ls = []
        inference_times = []
        
        for img_idx, (test_img_idx, eval_gt_available) in enumerate(zip(test_img_idx_ls, eval_gt_available_ls)):
            log.info(f"   Image {img_idx + 1}/{len(test_img_idx_ls)} (index {test_img_idx})")
            
            imgs, true_masks, buf_masks = list(test_loader)[test_img_idx]
            img_eval_out_dir = model_out_dir / f'img_{test_img_idx:02d}'
            img_eval_out_dir.mkdir(parents=True, exist_ok=True)
            
            # Profile inference
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            eval_results, forward_time = evaluate_single_model(
                model, imgs, true_masks, buf_masks,
                config, input_channels,
                eval_gt_available, img_eval_out_dir,
                show_now=show_now
            )

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            inference_times.append(forward_time)
            log.info(f"      Forward time (no postproc): {forward_time:.3f}s | total step: {elapsed:.3f}s")
            
            true_mask_ls.append(eval_results['true_mask'].flatten())
            pred_mask_ls.append(eval_results['pred_mask'].flatten())
        
        # Aggregate metrics for this model
        true_mask = np.concatenate(true_mask_ls)
        pred_mask = np.concatenate(pred_mask_ls)
        metrics_dict = calc_segmentation_statistics(true_mask, pred_mask, num_classes)
        
        # Add profiling info
        avg_inference_time = np.mean(inference_times)
        metrics_dict.update({
            'num_test_images': len(test_img_idx_ls),
            'avg_inference_time_sec': round(avg_inference_time, 3),
            'total_inference_time_sec': round(sum(inference_times), 2),
            'min_inference_time_sec': round(min(inference_times), 3),
            'max_inference_time_sec': round(max(inference_times), 3),
        })
        
        # Store for comparison
        individual_results[model_name] = {
            'arch': arch_name,
            'metrics': metrics_dict,
            'gflops': static_metrics['gflops'],
            'params': static_metrics['trainable_params']
        }
        
        # Save metrics for this model
        write_eval_metrics_to_file(metrics_dict, model_out_dir, key_str=model_name)
        log.info(f"✅ Model {model_name} evaluation complete")
        log.info(f"   Mean IOU: {metrics_dict.get('mIoU', 0):.4f}")
        log.info(f"   Overall Accuracy: {metrics_dict.get('oAcc', 0):.4f}")
    
    return individual_results, base_out_dir


def evaluate_ensemble_and_compare(config: dict, input_channels: list, 
                                 train_subset_cnt: int, save_uncertainty_figs: bool,
                                 individual_results: dict, individual_out_dir: Path):
    """
    Evaluate ensemble model and compare with individual models.
    """
    from eval_one4all import evaluate_imgs
    
    log.info(f"\n{'='*60}")
    log.info(f"📊 Evaluating Ensemble Model")
    log.info(f"{'='*60}")
    
    # Run ensemble evaluation (existing function)
    dataset_name = config['dataset_name']
    num_classes = config['num_classes']
    _, _, test_loader = load_data(config, input_channels)
    channels_str = '_'.join([str(ch) for ch in input_channels])
    eval_out_root_dir = Path(config['root_dir']) / f"run_subset_{train_subset_cnt:02d}"
    out_dir = eval_out_root_dir / 'outputs' / f'eval_{channels_str}'
    
    # Evaluate ensemble
    show_now = config['eval_imshow']
    _, _, test_loader = load_data(config, input_channels)
    
    test_img_idx_ls = config['test_img_idx_ls'] 
    eval_gt_available_ls = config['eval_gt_available_ls']
    
    ensemble_models = load_ensemble_models(config, input_channels, eval_out_root_dir)
    from eval_one4all import ensemble_predict_with_uncertainty, post_process_pred_batch, stitch_buffered_tiles
    
    input_h = config['input_size'][0]
    true_mask_ls = []
    pred_mask_ls = []
    inference_times = []
    
    for test_img_idx, eval_gt_available in zip(test_img_idx_ls, eval_gt_available_ls):
        imgs, true_masks, buf_masks = list(test_loader)[test_img_idx]
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        pred_dict = ensemble_predict_with_uncertainty(ensemble_models, imgs, buf_masks)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        inference_times.append(elapsed)
        
        imgs = imgs.cpu()
        pred = pred_dict['pred'].cpu()
        
        depad_img_tiles = [depad_tensor_vertical_only(img, original_height=input_h) for img in imgs]
        depat_buf_masks = [depad_tensor_vertical_only(mask, original_height=input_h) for mask in buf_masks]
        
        stitched_pred_mask = post_process_pred_batch(pred, input_h, buf_mask_ls=depat_buf_masks)
        stitched_true_mask = post_process_pred_batch(true_masks, input_h, buf_mask_ls=depat_buf_masks) if \
            eval_gt_available else np.zeros_like(stitched_pred_mask)
        
        true_mask_ls.append(stitched_true_mask.flatten())
        pred_mask_ls.append(stitched_pred_mask.flatten())
    
    true_mask = np.concatenate(true_mask_ls)
    pred_mask = np.concatenate(pred_mask_ls)
    ensemble_metrics = calc_segmentation_statistics(true_mask, pred_mask, num_classes)
    
    avg_inference_time = np.mean(inference_times)
    ensemble_metrics.update({
        'num_test_images': len(test_img_idx_ls),
        'avg_inference_time_sec': round(avg_inference_time, 3),
        'total_inference_time_sec': round(sum(inference_times), 2),
        'min_inference_time_sec': round(min(inference_times), 3),
        'max_inference_time_sec': round(max(inference_times), 3),
    })
    
    log.info(f"✅ Ensemble evaluation complete")
    log.info(f"   Mean IOU: {ensemble_metrics.get('mIoU', 0):.4f}")
    log.info(f"   Overall Accuracy: {ensemble_metrics.get('oAcc', 0):.4f}")
    log.info(f"   Avg Inference Time: {avg_inference_time:.3f}s")
    
    # Generate comparison report
    generate_comparison_report(individual_results, ensemble_metrics, 
                             individual_out_dir, channels_str)


def generate_comparison_report(individual_results: dict, ensemble_metrics: dict, 
                              output_dir: Path, channels_str: str):
    """
    Generate a detailed comparison report between individual models and ensemble.
    """
    log.info(f"\n{'='*60}")
    log.info(f"📈 Comparison Report: Individual Models vs Ensemble")
    log.info(f"{'='*60}")
    
    comparison_data = {
        'ensemble': {
            'metrics': ensemble_metrics,
            'type': 'ensemble'
        }
    }
    comparison_data.update(individual_results)
    
    # Create comparison table
    comparison_file = output_dir / f'model_comparison_{channels_str}.json'
    with open(comparison_file, 'w') as f:
        # Convert non-serializable items
        serializable_data = {}
        for model_name, data in comparison_data.items():
            serializable_data[model_name] = {
                'arch': data.get('arch', 'ensemble'),
                'metrics': {k: v for k, v in data['metrics'].items() if isinstance(v, (int, float, str))},
                'gflops': data.get('gflops', 0),
                'params': data.get('params', 0)
            }
        json.dump(serializable_data, f, indent=2)
    
    log.info(f"✅ Comparison report saved to {comparison_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Model Name':<20} {'mIoU':<12} {'Acc':<12} {'Time(s)':<12} {'GFLOPs':<12}")
    print("-"*80)
    
    for model_name in ['ensemble'] + list(individual_results.keys()):
        if model_name == 'ensemble':
            data = {'metrics': ensemble_metrics, 'gflops': 0}
        else:
            data = individual_results[model_name]
        
        metrics = data['metrics']
        miou = metrics.get('mIoU', 0)
        acc = metrics.get('oAcc', 0)
        time_s = metrics.get('avg_inference_time_sec', 0)
        gflops = data.get('gflops', 0)
        
        print(f"{model_name:<20} {miou:<12.4f} {acc:<12.4f} {time_s:<12.3f} {gflops:<12.2f}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate individual ensemble models and compare against ensemble'
    )
    parser.add_argument('--config', type=str, default='params/paths_rc_forestsemantic.json',
                        help='Path to the configuration JSON file')
    args = parser.parse_args()
    
    CONFIG = load_config(args.config)
    CONFIG['_config_path'] = args.config
    print(f"Using config file: {args.config}")
    
    input_channels_ls = CONFIG['input_channels_ls']
    train_subset_cnts = CONFIG['train_subset_cnts']
    save_uncertainty_figs = CONFIG.get('save_uncertainty_figs', False)
    
    for train_subset_cnt in train_subset_cnts:
        for input_channels in input_channels_ls:
            log.info(f"\n{'#'*60}")
            log.info(f"Evaluating: subset={train_subset_cnt}, channels={input_channels}")
            log.info(f"{'#'*60}")
            
            # Evaluate individual models
            individual_results, individual_out_dir = evaluate_individual_models_with_profiling(
                CONFIG, input_channels, train_subset_cnt
            )
            
            # Evaluate ensemble and compare
            evaluate_ensemble_and_compare(
                CONFIG, input_channels, train_subset_cnt,
                save_uncertainty_figs, individual_results, individual_out_dir
            )
            
            log.info(f"✅ All evaluations complete for subset={train_subset_cnt}, channels={input_channels}")


if __name__ == '__main__':
    main()
