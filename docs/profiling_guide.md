# Experiment Profiling Guide

This guide explains how to use the profiling utilities for reproducible benchmarking.

## Quick Start

### 1. Training with Profiling

```bash
python training_multichannel_one4all.py --config params/rc_mangrov3d.json
```

This automatically:
- Measures model params and GFLOPs
- Tracks time and memory per epoch
- Logs to TensorBoard
- Saves results to `profile_*.json` and `profile_*.csv`

### 2. Evaluation with Profiling

```bash
python eval_one4all_profiled.py --config params/rc_mangrov3d.json
```

This measures:
- Inference time per image
- GPU memory during inference
- Overall evaluation metrics

### 3. Aggregate Results

```bash
python tools/aggregate_profiles.py --root /path/to/experiments --output summary.csv --latex
```

Creates:
- `summary.csv`: All experiments in one table
- `summary.tex`: LaTeX table ready for papers

## Output Files

### Training

```
run_subset_04/
├── outputs/
│   └── unetpp_r34/
│       ├── profile_0_0_0_20250101_120000.json  # Complete metrics
│       ├── profile_0_0_0.csv                    # For aggregation
│       ├── tensorboard/                         # TensorBoard logs
│       ├── losses_0_0_0_*.png                   # Loss curves
│       └── metrics_0_0_0_*.png                  # Accuracy curves
```

### Evaluation

```
run_subset_04/
└── outputs/
    └── eval_0_0_0/
        ├── eval_profile_0_0_0.json  # Inference metrics
        └── eval_profile_0_0_0.csv
```

## Metrics Explained

### Static Metrics
- **total_params**: Total parameters (including non-trainable)
- **trainable_params**: Only trainable parameters
- **gflops**: Floating point operations (billions) per forward pass

### Training Metrics
- **total_training_time_min**: Full training duration
- **mean_epoch_time_sec**: Average time per epoch
- **peak_memory_mb**: Maximum GPU memory used

### Inference Metrics
- **mean_inference_time_ms**: Average per-image inference
- **mean_peak_memory_mb**: Average GPU memory per inference

## TensorBoard

View real-time training progress:

```bash
tensorboard --logdir run_subset_04/outputs/unetpp_r34/tensorboard
```

Charts include:
- Loss curves (train/val)
- Metrics (oAcc, mIoU)
- Time per epoch
- GPU memory per epoch

## Best Practices

1. **Warmup**: Models are warmed up before timing (GPU kernel compilation)
2. **Synchronization**: CUDA operations are synchronized for accurate timing
3. **Memory tracking**: Peak memory is reset at epoch start
4. **Reproducibility**: Fixed input shapes, documented hardware

## For Papers

The profiling outputs are designed for ISPRS/IEEE/RSE papers:

1. Run experiments with different configurations
2. Aggregate with `aggregate_profiles.py`
3. Use generated LaTeX table directly
4. Reference TensorBoard plots for supplementary material

## Example Table

| Model | Channels | Params (M) | GFLOPs | Train Time (min) | mIoU | OA |
|-------|----------|------------|--------|------------------|------|-----|
| UNet++ | RGB | 24.8 | 156.2 | 45.3 | 0.78 | 0.85 |
| DeepLabV3+ | RGB+NIR | 39.6 | 248.7 | 62.1 | 0.81 | 0.87 |

