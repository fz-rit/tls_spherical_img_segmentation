"""
Model profiling utilities for academic research.
Measures static (params, FLOPs) and dynamic (time, memory) metrics.

Design principles:
- Single-GPU focused
- Reproducible measurements
- Easy aggregation for papers
"""

import torch
import time
import json
import csv
from pathlib import Path
from typing import Dict, Tuple, Optional
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
import platform


class ModelProfiler:
    """
    Profiler for segmentation models.
    
    Measures:
    - Static: parameters, GFLOPs
    - Dynamic: training time, inference time, GPU memory
    """
    
    def __init__(self, model: torch.nn.Module, input_shape: Tuple[int, ...], 
                 device: str = 'cuda', model_name: str = 'model'):
        """
        Args:
            model: PyTorch model to profile
            input_shape: (batch, channels, height, width) for FLOPs calculation
            device: 'cuda' or 'cpu'
            model_name: descriptive name for logging
        """
        self.model = model
        self.input_shape = input_shape
        self.device = device
        self.model_name = model_name
        
        # Storage for metrics
        self.metrics = {
            'model_name': model_name,
            'input_shape': str(input_shape),
            'device': device
        }
        
    def profile_static_metrics(self) -> Dict:
        """
        Compute static model properties (params, FLOPs).
        These are deterministic and input-shape dependent.
        """
        # 1. Parameter count using torchinfo
        # This gives us trainable vs total params
        model_stats = summary(
            self.model, 
            input_size=self.input_shape,
            device=self.device,
            verbose=0  # suppress output
        )
        
        total_params = model_stats.total_params
        trainable_params = model_stats.trainable_params
        
        # 2. FLOPs using fvcore
        # Create dummy input for FLOPs analysis
        dummy_input = torch.randn(self.input_shape).to(self.device)
        flops = FlopCountAnalysis(self.model, dummy_input)
        total_flops = flops.total()
        
        # Convert to GFLOPs (standard for papers)
        gflops = total_flops / 1e9
        
        self.metrics.update({
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'total_flops': int(total_flops),
            'gflops': round(gflops, 3)
        })
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'gflops': gflops
        }
    
    def get_hardware_info(self) -> Dict:
        """Collect hardware configuration for reproducibility."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
        }
        
        if torch.cuda.is_available():
            info.update({
                'cuda_available': True,
                'cuda_version': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
            })
        else:
            info['cuda_available'] = False
            
        self.metrics.update(info)
        return info
    
    def _convert_to_serializable(self, obj):
        """Convert numpy/torch types to JSON-serializable Python types."""
        import numpy as np
        
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # torch tensors
            return obj.item()
        else:
            return obj
    
    def save_metrics(self, output_path: Path, format: str = 'json'):
        """
        Save collected metrics to file.
        
        Args:
            output_path: where to save
            format: 'json' or 'csv'
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any numpy/torch types to native Python types
        serializable_metrics = self._convert_to_serializable(self.metrics)
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        elif format == 'csv':
            # For easy aggregation across experiments
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=serializable_metrics.keys())
                writer.writeheader()
                writer.writerow(serializable_metrics)
        
        print(f"📊 Metrics saved to {output_path}")


class RuntimeProfiler:
    """
    Context manager for profiling runtime metrics.
    Handles timing and GPU memory tracking.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        """Start profiling."""
        if self.device == 'cuda':
            torch.cuda.synchronize()  # ensure all ops complete
            torch.cuda.reset_peak_memory_stats()  # reset memory counter
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        """End profiling and collect metrics."""
        if self.device == 'cuda':
            torch.cuda.synchronize()  # wait for GPU ops
        self.end_time = time.time()
        
    @property
    def elapsed_time(self) -> float:
        """Time elapsed in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def peak_memory_mb(self) -> float:
        """Peak GPU memory in MB."""
        if self.device == 'cuda':
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0


def profile_inference_speed(model: torch.nn.Module, 
                           input_shape: Tuple[int, ...],
                           device: str = 'cuda',
                           warmup_runs: int = 5,
                           num_runs: int = 50) -> Dict:
    """
    Measure inference time with proper warmup.
    
    Why warmup? GPU kernels are compiled on first run (cuDNN, etc.)
    Following best practices from PyTorch benchmarking guide.
    
    Args:
        model: model to profile
        input_shape: input dimensions
        device: 'cuda' or 'cpu'
        warmup_runs: number of warmup iterations
        num_runs: number of timed iterations
        
    Returns:
        dict with mean, std, min, max inference times
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup phase
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
    
    # Actual measurement
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device == 'cuda':
                torch.cuda.synchronize()
            start = time.time()
            
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed)
    
    import numpy as np
    return {
        'mean_time_ms': round(np.mean(times) * 1000, 2),
        'std_time_ms': round(np.std(times) * 1000, 2),
        'min_time_ms': round(np.min(times) * 1000, 2),
        'max_time_ms': round(np.max(times) * 1000, 2),
    }


if __name__ == "__main__":
    # Example usage
    from tools.feature_fusion_helper import build_model_for_multi_channels
    
    model = build_model_for_multi_channels(
        model_name='Unet',
        encoder_name='resnet34',
        in_channels=3,
        num_classes=6
    )
    model = model.to('cuda')
    
    # Profile static metrics
    profiler = ModelProfiler(
        model=model,
        input_shape=(1, 3, 512, 512),
        model_name='UNet_resnet34'
    )
    
    print("Static metrics:")
    static = profiler.profile_static_metrics()
    print(f"  Parameters: {static['total_params']:,}")
    print(f"  GFLOPs: {static['gflops']:.2f}")
    
    print("\nHardware info:")
    hw = profiler.get_hardware_info()
    print(f"  GPU: {hw.get('gpu_name', 'N/A')}")
    
    # Profile inference
    print("\nInference speed:")
    inf_stats = profile_inference_speed(model, (1, 3, 512, 512))
    print(f"  Mean: {inf_stats['mean_time_ms']:.2f} ms")
    
    # Save all metrics
    profiler.metrics.update(inf_stats)
    profiler.save_metrics(Path('profile_test.json'))