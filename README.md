# Sphrical Image Segmentation for TLS

Semantic segmentation on multi-channel spherical images generated from Terrestrial Laser Scanning (TLS) point clouds. This project implements ensemble deep learning models for forest semantic segmentation using multi-channel image data.

## Features

- **Multi-channel image segmentation** using RGB and additional spectral channels
- **Ensemble model architecture** supporting UNet++, DeepLabV3+, and Segformer2
- **Uncertainty quantification** through model ensemble predictions
- **Comprehensive evaluation metrics** including IoU, accuracy, and uncertainty analysis
- **Flexible configuration system** for different datasets and model configurations
- **Support for multiple datasets**: ForestSemantic, Mangrove3D, InLUT3D

## Project Structure

```
├── README.md                           # This file
├── prepare_dataset.py                  # Dataset preparation and loading
├── training_multichannel_one4all.py    # Main training script
├── eval_one4all.py                     # Model evaluation script
├── params/                             # Configuration files
│   ├── config_*.json                  # Model and training configurations
│   └── paths_*.json                   # Dataset path configurations
├── tools/                              # Utility modules
│   ├── load_tools.py                  # Model loading utilities
│   ├── metrics_tools.py               # Evaluation metrics
│   ├── visualize_tools.py             # Visualization functions
│   └── feature_fusion_helper.py       # Multi-channel model builders
├── outputs/                            # Generated outputs and results
└── log/                               # Training and evaluation logs
```

## Installation

### Environment Setup
```bash
# Create conda environment
conda create -n img_seg_env python=3.9 -y
conda activate img_seg_env

# Install PyTorch and core dependencies
conda install -c conda-forge pytorch torchvision opencv tensorboard seaborn -y

# Install additional packages
pip install albumentations segmentation-models-pytorch
```

## Usage

### 1. Dataset Preparation

First, prepare your dataset and verify the data loaders:

```bash
python prepare_dataset.py --config params/paths_rc_forestsemantic.json
```

This script will:
- Load and validate image-mask pairs
- Create train/validation/test splits
- Generate data loaders with proper preprocessing
- Display dataset statistics

### 2. Model Training

Train the ensemble models using:

```bash
python training_multichannel_one4all.py --config params/paths_rc_forestsemantic.json
```

The training script will:
- Train multiple model architectures (UNet++, DeepLabV3+, Segformer)
- Use different encoder backbones (ResNet34, EfficientNet-B3, MiT-B1)
- Save best models based on validation performance
- Generate training logs and visualization plots
- Support early stopping to prevent overfitting

### 3. Model Evaluation

Evaluate trained models and generate predictions:

```bash
python eval_one4all.py --config params/paths_rc_forestsemantic.json
```

The evaluation script will:
- Load ensemble models and make predictions
- Calculate segmentation metrics (IoU, accuracy, F1-score)
- Generate uncertainty maps from ensemble predictions
- Save evaluation results and visualizations
- Export detailed metrics to files

## Configuration
Use JSON files in the `params/` directory to configure dataset paths, model parameters, training settings, and evaluation options.

## Supported Models

- **UNet++**: Enhanced U-Net with nested skip connections
- **DeepLabV3+**: Atrous convolution with encoder-decoder structure
- **Segformer**: Vision Transformer-based segmentation model

## Supported Encoders

- ResNet (resnet34, resnet50, resnet101)
- EfficientNet (efficientnet-b0 to efficientnet-b7)
- MiT (mit_b0 to mit_b5) for Segformer

## Output Files

The project generates several types of outputs:

- **Models**: Trained model weights (`.pth` files)
- **Logs**: Training progress and metrics
- **Visualizations**: Training curves, prediction maps, uncertainty maps
- **Metrics**: Detailed evaluation statistics (`.yaml` files)
- **ONNX Models**: For deployment (optional)

## Key Features

### Multi-Channel Input Support
The framework supports multi-channel inputs beyond RGB, allowing incorporation of additional spectral or geometric features from TLS data.

### Uncertainty Quantification
Ensemble predictions provide uncertainty estimates, helping identify regions where the model is less confident.

### Flexible Training
- Configurable training subset sizes
- Early stopping with patience
- Model checkpointing
- Comprehensive logging

## Citation
If you use this code for your research, please cite:

```bibtex
@article{zhang2025through,
  title={Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation},
  author={Zhang, Fei and Chancia, Rob and Clapp, Josie and Hassanzadeh, Amirhossein and Dera, Dimah and MacKenzie, Richard and van Aardt, Jan},
  journal={arXiv preprint arXiv:2510.06582},
  year={2025}
}
```

