"""
Training-specific profiling hooks.
Tracks per-epoch metrics: time, memory, loss.
"""

import torch
import time
from typing import Dict, Optional
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tools.logger_setup import Logger

log = Logger()


class TrainingProfiler:
    """
    Lightweight profiler for training loop.
    
    Tracks:
    - Time per epoch
    - Peak GPU memory per epoch
    - Cumulative training time
    - Integrates with TensorBoard
    """
    
    def __init__(self, log_dir: Path, device: str = 'cuda', 
                 use_tensorboard: bool = True):
        """
        Args:
            log_dir: directory for TensorBoard logs
            device: 'cuda' or 'cpu'
            use_tensorboard: whether to log to TensorBoard
        """
        self.device = device
        self.use_tensorboard = use_tensorboard
        
        # Metrics storage
        self.epoch_times = []
        self.epoch_memory = []
        self.total_time = 0.0
        
        # Epoch-level trackers
        self.epoch_start_time = None
        
        # TensorBoard
        if use_tensorboard:
            self.writer = SummaryWriter(log_dir / 'tensorboard')
        else:
            self.writer = None
            
    def start_epoch(self):
        """Call at the beginning of each epoch."""
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.epoch_start_time = time.time()
        
    def end_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: Optional[Dict] = None,
                  val_metrics: Optional[Dict] = None):
        """
        Call at the end of each epoch.
        
        Args:
            epoch: current epoch number
            train_loss: training loss
            val_loss: validation loss
            train_metrics: dict with oAcc, mIoU, etc.
            val_metrics: dict with oAcc, mIoU, etc.
        """
        # Time measurement
        if self.device == 'cuda':
            torch.cuda.synchronize()
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_times.append(epoch_time)
        self.total_time += epoch_time
        
        # Memory measurement
        peak_mem_mb = 0.0
        if self.device == 'cuda':
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        self.epoch_memory.append(peak_mem_mb)
        
        # Log to console
        log.info(f"⏱️  Epoch {epoch}: {epoch_time:.2f}s | "
                f"Peak GPU: {peak_mem_mb:.1f} MB")
        
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar('Time/epoch_time', epoch_time, epoch)
            self.writer.add_scalar('Memory/peak_gpu_mb', peak_mem_mb, epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            
            if train_metrics:
                for key, val in train_metrics.items():
                    self.writer.add_scalar(f'Train/{key}', val, epoch)
            if val_metrics:
                for key, val in val_metrics.items():
                    self.writer.add_scalar(f'Val/{key}', val, epoch)
    
    def get_summary(self) -> Dict:
        """Get summary statistics for all epochs."""
        import numpy as np
        return {
            'total_training_time_sec': round(self.total_time, 2),
            'total_training_time_min': round(self.total_time / 60, 2),
            'mean_epoch_time_sec': round(np.mean(self.epoch_times), 2) if self.epoch_times else 0,
            'std_epoch_time_sec': round(np.std(self.epoch_times), 2) if self.epoch_times else 0,
            'peak_memory_mb': round(max(self.epoch_memory), 2) if self.epoch_memory else 0,
            'mean_memory_mb': round(np.mean(self.epoch_memory), 2) if self.epoch_memory else 0,
            'num_epochs': len(self.epoch_times)
        }
    
    def close(self):
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


class InferenceProfiler:
    """
    Profiler for evaluation/inference phase.
    Tracks per-image inference time and memory.
    """
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.inference_times = []
        self.peak_memories = []
        
    def profile_single_inference(self, model, imgs: torch.Tensor) -> Dict:
        """
        Profile a single inference pass.
        
        Returns:
            dict with time_ms and peak_memory_mb
        """
        if self.device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.time()
        with torch.no_grad():
            _ = model(imgs)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.time() - start
        
        peak_mem = 0.0
        if self.device == 'cuda':
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        self.inference_times.append(elapsed)
        self.peak_memories.append(peak_mem)
        
        return {
            'time_ms': round(elapsed * 1000, 2),
            'peak_memory_mb': round(peak_mem, 2)
        }
    
    def get_summary(self) -> Dict:
        """Get inference statistics across all images."""
        import numpy as np
        times_ms = [t * 1000 for t in self.inference_times]
        
        return {
            'num_inferences': len(self.inference_times),
            'mean_inference_time_ms': round(np.mean(times_ms), 2) if times_ms else 0,
            'std_inference_time_ms': round(np.std(times_ms), 2) if times_ms else 0,
            'min_inference_time_ms': round(np.min(times_ms), 2) if times_ms else 0,
            'max_inference_time_ms': round(np.max(times_ms), 2) if times_ms else 0,
            'mean_peak_memory_mb': round(np.mean(self.peak_memories), 2) if self.peak_memories else 0,
            'max_peak_memory_mb': round(np.max(self.peak_memories), 2) if self.peak_memories else 0,
        }