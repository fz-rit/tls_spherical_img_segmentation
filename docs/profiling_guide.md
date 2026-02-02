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

### 2. Ensemble Evaluation with Profiling

```bash
python eval_one4all_profiled.py --config params/rc_mangrov3d.json
```

This measures:
- Inference time per image
- GPU memory during inference
- Overall evaluation metrics
- Uncertainty metrics (entropy, mutual information, epistemic uncertainty)

### 3. Individual Model Evaluation with Profiling

```bash
python eval_individual_models_profiled.py --config params/rc_mangrov3d.json
```

This evaluates **each model in the ensemble separately** and measures:
- Per-model inference time
- Per-model accuracy metrics (mIoU, overall accuracy, per-class IoU)
- Model complexity (GFLOPs, parameters)
- Saves individual predictions separately for comparison
- Automatically compares against ensemble performance

### 4. Generate Comparison Analysis

```bash
python -m tools.compare_models --eval-dir run_subset_01/outputs --channels 0_1_2
```

This analyzes individual model results and generates:
- Side-by-side metrics comparison table
- Performance comparison plots
- Accuracy vs speed trade-off visualization
- Detailed summary report with recommendations

### 3. Aggregate Results

```bash
python tools/aggregate_profiles.py --root /path/to/experiments --output summary.csv --latex
```

Creates:
- `summary.csv`: All experiments in one table
- `summary.tex`: LaTeX table ready for papers

## Individual Model Evaluation

When using `eval_individual_models_profiled.py`, you can compare each ensemble member against the full ensemble:

```bash
python eval_individual_models_profiled.py --config params/rc_mangrov3d.json
```

**Output structure:**

```
run_subset_XX/outputs/
├── eval_[channels]/                    # Ensemble predictions & uncertainty
├── eval_[channels]_individual/         # Individual model predictions
│   ├── model_1_name/
│   │   ├── img_00/, img_01/, ...      # Per-image visualizations
│   │   └── model_1_name_metrics.json   # Individual metrics
│   ├── model_2_name/
│   ├── ...
│   └── model_comparison_[channels].json # Raw comparison data
└── comparison_reports_[channels]/      # Analysis & plots
    ├── metrics_comparison.md            # Detailed metrics table
    ├── summary_report.md                # Executive summary
    ├── miou_comparison.png
    ├── accuracy_comparison.png
    ├── inference_time_comparison.png
    └── accuracy_speed_tradeoff.png
```

**Individual metrics include:**
- **mIoU**: Mean Intersection over Union
- **overall_accuracy**: Pixel-wise accuracy
- **class_iou_[i]**: Per-class IoU scores
- **avg_inference_time_sec**: Average time per image
- **total_inference_time_sec**: Total evaluation time
- **min/max_inference_time_sec**: Inference time bounds

## Model Comparison Analysis

After evaluating individual models, generate comprehensive comparison reports:

```bash
python -m tools.compare_models --eval-dir run_subset_01/outputs --channels 0_1_2
```

**Generated outputs:**

1. **metrics_comparison.md**: Markdown table with:
   - Summary metrics (mIoU, accuracy, inference time)
   - Per-class IoU for each model

2. **summary_report.md**: Comprehensive report including:
   - Best model for accuracy
   - Fastest model
   - Per-model detailed metrics
   - Recommendations for different use cases

3. **Visualization plots:**
   - `miou_comparison.png`: Bar chart of mIoU scores
   - `accuracy_comparison.png`: Overall accuracy comparison
   - `inference_time_comparison.png`: Speed comparison
   - `accuracy_speed_tradeoff.png`: Accuracy vs speed scatter plot

### Interpreting Results

**For Accuracy Priority:**
- Look at `miou_comparison.png` to identify the highest-performing model
- Check `summary_report.md` for "Best for Accuracy" recommendation

**For Speed Priority:**
- Look at `inference_time_comparison.png`
- Use `accuracy_speed_tradeoff.png` to find best speed-accuracy balance
- Reference `summary_report.md` for "Best for Speed" recommendation

**For Ensemble Benefits:**
- Compare individual model mIoU with ensemble mIoU
- Ensemble typically provides robustness and uncertainty estimates
- Cost: ensemble inference time ≈ sum of individual model times

## Aggregate Results

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
    ├── eval_0_0_0/
    │   ├── eval_profile_0_0_0.json              # Ensemble inference metrics
    │   ├── eval_profile_0_0_0.csv
    │   ├── [img_00, img_01, ...]/               # Per-image ensemble results
    │   └── eval_metrics_0_0_0.txt
    └── eval_0_0_0_individual/
        ├── model_1_name/
        │   ├── img_00/, img_01/, ...            # Per-image predictions
        │   └── model_1_name_metrics.json
        ├── model_2_name/
        ├── ...
        └── model_comparison_0_0_0.json
```

### Comparison Reports

```
run_subset_04/
└── outputs/
    └── comparison_reports_0_0_0/
        ├── metrics_comparison.md                # Detailed comparison table
        ├── summary_report.md                    # Executive summary
        ├── miou_comparison.png                  # Performance plots
        ├── accuracy_comparison.png
        ├── inference_time_comparison.png
        └── accuracy_speed_tradeoff.png
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

### Ensemble Inference Metrics
- **mean_inference_time_ms**: Average per-image inference
- **mean_peak_memory_mb**: Average GPU memory per inference
- **total_uncertainty**: Entropy of mean prediction
- **mutual_info**: Epistemic uncertainty from ensemble variance
- **var_based_epistemic**: Variance-based uncertainty measure

### Individual Model Metrics
- **mIoU**: Mean Intersection over Union across all classes
- **overall_accuracy**: Pixel-wise accuracy across all images
- **class_iou_[i]**: Per-class IoU score for class i
- **avg_inference_time_sec**: Average inference time per image
- **total_inference_time_sec**: Total time for all test images
- **min/max_inference_time_sec**: Inference time bounds

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
5. **Individual vs Ensemble**: Always evaluate individual models separately for fair comparison
   - Individual models: Direct inference, no aggregation
   - Ensemble: Aggregated logits, uncertainty estimates (slower but more robust)
6. **Comparison Analysis**: Use `compare_models.py` after individual evaluation to generate insights

## Workflow Summary

### Complete Profiling Workflow

```bash
# 1. Train model (with automatic profiling)
python training_multichannel_one4all.py --config params/rc_mangrov3d.json

# 2. Evaluate ensemble (with uncertainty)
python eval_one4all_profiled.py --config params/rc_mangrov3d.json

# 3. Evaluate individual models (separate predictions for each)
python eval_individual_models_profiled.py --config params/rc_mangrov3d.json

# 4. Generate comparison analysis
python -m tools.compare_models --eval-dir run_subset_01/outputs --channels 0_1_2

# 5. (Optional) Aggregate all results
python tools/aggregate_profiles.py --root /path/to/experiments --output summary.csv --latex
```

### Quick Performance Check

If you only need to compare models quickly without full ensemble uncertainty:

```bash
# Skip full eval_one4all_profiled.py, just use individual evaluation
python eval_individual_models_profiled.py --config params/rc_mangrov3d.json

# Then generate comparison
python -m tools.compare_models --eval-dir run_subset_01/outputs --channels 0_1_2
```

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

