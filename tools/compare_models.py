"""
Comparison analysis tool for individual models vs ensemble.
Generates detailed reports, plots, and metrics comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tools.logger_setup import Logger
import argparse

log = Logger()


class ModelComparisonAnalyzer:
    """Analyzes and compares model evaluation results."""
    
    def __init__(self, base_eval_dir: Path):
        """
        Initialize analyzer.
        
        Args:
            base_eval_dir: Base evaluation directory containing individual and ensemble results
        """
        self.base_eval_dir = base_eval_dir
        self.individual_dir = base_eval_dir / 'eval_individual'
        self.ensemble_dir = base_eval_dir
        self.individual_results = {}
        self.ensemble_results = {}
    
    def load_results(self):
        """Load all evaluation results."""
        log.info("Loading evaluation results...")
        
        # Load individual model results
        if self.individual_dir.exists():
            for model_dir in self.individual_dir.iterdir():
                if model_dir.is_dir():
                    metrics_file = model_dir / f'{model_dir.name}_metrics.json'
                    if metrics_file.exists():
                        with open(metrics_file) as f:
                            self.individual_results[model_dir.name] = json.load(f)
                        log.info(f"  ✓ Loaded {model_dir.name}")
        
        log.info(f"Loaded {len(self.individual_results)} individual models")
    
    def create_metrics_comparison_table(self, output_file: Path, include_iou_per_class: bool = True):
        """
        Create a detailed metrics comparison table.
        
        Args:
            output_file: Path to save the comparison table
            include_iou_per_class: Whether to include per-class IOU metrics
        """
        if not self.individual_results:
            log.warning("No individual results to compare")
            return
        
        log.info(f"Creating metrics comparison table...")
        
        # Collect all models
        all_models = list(self.individual_results.keys())
        
        # Key metrics to compare
        key_metrics = [
            'mIoU',
            'oAcc',
            'avg_inference_time_sec',
            'total_inference_time_sec',
        ]
        
        # Create comparison table
        with open(output_file, 'w') as f:
            f.write("# Model Performance Comparison\n\n")
            
            # Summary table
            f.write("## Summary Metrics\n\n")
            f.write("| Model | mIoU | Overall Acc | Avg Time (s) | Total Time (s) |\n")
            f.write("|-------|------|------------|--------------|----------------|\n")
            
            for model_name in sorted(all_models):
                metrics = self.individual_results[model_name]
                miou = metrics.get('mIoU', 0)
                acc = metrics.get('oAcc', 0)
                avg_time = metrics.get('avg_inference_time_sec', 0)
                total_time = metrics.get('total_inference_time_sec', 0)
                
                f.write(f"| {model_name} | {miou:.4f} | {acc:.4f} | {avg_time:.3f} | {total_time:.2f} |\n")
            
            # Per-class IOU
            if include_iou_per_class:
                f.write("\n## Per-Class IoU Metrics\n\n")
                
                for model_name in sorted(all_models):
                    metrics = self.individual_results[model_name]
                    f.write(f"\n### {model_name}\n\n")
                    
                    # Find all class IOU keys
                    iou_keys = [k for k in metrics.keys() if 'class_iou' in k or 'IoU_class' in k]
                    if iou_keys:
                        f.write("| Class | IoU |\n")
                        f.write("|-------|-----|\n")
                        for iou_key in sorted(iou_keys):
                            class_num = ''.join(filter(str.isdigit, iou_key))
                            iou_val = metrics[iou_key]
                            f.write(f"| Class {class_num} | {iou_val:.4f} |\n")
        
        log.info(f"✅ Comparison table saved to {output_file}")
    
    def create_performance_plots(self, output_dir: Path):
        """
        Create comparison plots.
        
        Args:
            output_dir: Directory to save plots
        """
        if not self.individual_results:
            log.warning("No individual results to plot")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Creating performance plots...")
        
        models = sorted(self.individual_results.keys())
        
        # Plot 1: mIoU comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        miou_values = [self.individual_results[m].get('mIoU', 0) for m in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, miou_values, color=colors)
        ax.set_ylabel('Mean IoU', fontsize=12)
        ax.set_title('Model Comparison: Mean IoU', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, val in zip(bars, miou_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'miou_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved mIoU comparison plot")
        
        # Plot 2: Overall accuracy comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        acc_values = [self.individual_results[m].get('oAcc', 0) for m in models]
        bars = ax.bar(models, acc_values, color=colors)
        ax.set_ylabel('Overall Accuracy', fontsize=12)
        ax.set_title('Model Comparison: Overall Accuracy', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        for bar, val in zip(bars, acc_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved accuracy comparison plot")
        
        # Plot 3: Inference time comparison
        fig, ax = plt.subplots(figsize=(12, 6))
        time_values = [self.individual_results[m].get('avg_inference_time_sec', 0) for m in models]
        bars = ax.bar(models, time_values, color=colors)
        ax.set_ylabel('Average Inference Time (seconds)', fontsize=12)
        ax.set_title('Model Comparison: Inference Speed', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, time_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}s', ha='center', va='bottom', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_dir / 'inference_time_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved inference time comparison plot")
        
        # Plot 4: mIoU vs Inference Time (scatter)
        fig, ax = plt.subplots(figsize=(10, 8))
        for i, model in enumerate(models):
            miou = self.individual_results[model].get('mIoU', 0)
            time_val = self.individual_results[model].get('avg_inference_time_sec', 0)
            ax.scatter(time_val, miou, s=200, alpha=0.6, label=model, color=colors[i])
            ax.annotate(model, (time_val, miou), fontsize=9, 
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Avg Inference Time (seconds)', fontsize=12)
        ax.set_ylabel('Mean IoU', fontsize=12)
        ax.set_title('Accuracy vs Speed Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'accuracy_speed_tradeoff.png', dpi=150, bbox_inches='tight')
        plt.close()
        log.info(f"  ✓ Saved trade-off plot")
        
        log.info(f"✅ All plots saved to {output_dir}")
    
    def generate_summary_report(self, output_file: Path):
        """
        Generate a comprehensive summary report.
        
        Args:
            output_file: Path to save the report
        """
        if not self.individual_results:
            log.warning("No individual results for report")
            return
        
        log.info(f"Generating summary report...")
        
        models = sorted(self.individual_results.keys())
        
        with open(output_file, 'w') as f:
            f.write("# Individual Models vs Ensemble - Comprehensive Report\n\n")
            
            f.write("## Executive Summary\n\n")
            
            # Find best model
            best_miou_model = max(models, 
                                 key=lambda m: self.individual_results[m].get('mIoU', 0))
            best_miou = self.individual_results[best_miou_model].get('mIoU', 0)
            
            fastest_model = min(models,
                               key=lambda m: self.individual_results[m].get('avg_inference_time_sec', float('inf')))
            fastest_time = self.individual_results[fastest_model].get('avg_inference_time_sec', 0)
            
            f.write(f"- **Best mIoU Model**: {best_miou_model} ({best_miou:.4f})\n")
            f.write(f"- **Fastest Model**: {fastest_model} ({fastest_time:.3f}s per image)\n")
            f.write(f"- **Total Models Evaluated**: {len(models)}\n\n")
            
            f.write("## Detailed Metrics\n\n")
            
            for model_name in models:
                metrics = self.individual_results[model_name]
                f.write(f"\n### {model_name}\n\n")
                f.write("**Performance Metrics:**\n\n")
                f.write(f"- Mean IoU (mIoU): {metrics.get('mIoU', 0):.4f}\n")
                f.write(f"- Overall Accuracy: {metrics.get('oAcc', 0):.4f}\n")
                f.write(f"- Average Inference Time: {metrics.get('avg_inference_time_sec', 0):.3f}s\n")
                f.write(f"- Total Inference Time: {metrics.get('total_inference_time_sec', 0):.2f}s\n")
                f.write(f"- Min Inference Time: {metrics.get('min_inference_time_sec', 0):.3f}s\n")
                f.write(f"- Max Inference Time: {metrics.get('max_inference_time_sec', 0):.3f}s\n")
                f.write(f"- Num Test Images: {metrics.get('num_test_images', 0)}\n\n")
                
                # Per-class metrics
                iou_keys = [k for k in metrics.keys() if 'class_iou' in k or 'IoU_class' in k]
                if iou_keys:
                    f.write("**Per-Class IoU:**\n\n")
                    for iou_key in sorted(iou_keys):
                        class_num = ''.join(filter(str.isdigit, iou_key))
                        iou_val = metrics[iou_key]
                        f.write(f"- Class {class_num}: {iou_val:.4f}\n")
                    f.write("\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("### Best for Accuracy\n")
            f.write(f"Use **{best_miou_model}** if maximizing segmentation accuracy is priority.\n\n")
            
            f.write("### Best for Speed\n")
            f.write(f"Use **{fastest_model}** if minimizing inference time is priority.\n\n")
            
            f.write("### Ensemble Benefits\n")
            f.write("The ensemble model combines predictions from all models, potentially offering:\n")
            f.write("- Better robustness through model diversity\n")
            f.write("- Uncertainty estimates via ensemble variance\n")
            f.write("- Improved generalization\n")
            f.write("\nNote: Ensemble inference time is sum of individual inference times.\n")
        
        log.info(f"✅ Summary report saved to {output_file}")


def analyze_eval_directory(eval_dir: Path, channels_str: str):
    """
    Analyze evaluation directory structure and generate comparison reports.
    
    Args:
        eval_dir: Base evaluation directory (e.g., run_subset_XX/outputs)
        channels_str: Channel string identifier (e.g., '0_1_2')
    """
    log.info(f"\nAnalyzing evaluation results from {eval_dir}")
    log.info(f"Channel configuration: {channels_str}")
    
    # Look for both individual and ensemble directories
    individual_eval_dir = eval_dir / f'eval_{channels_str}_individual'
    ensemble_eval_dir = eval_dir / f'eval_{channels_str}'
    
    if not individual_eval_dir.exists():
        log.warning(f"Individual evaluation directory not found: {individual_eval_dir}")
        return
    
    # Initialize analyzer
    analyzer = ModelComparisonAnalyzer(ensemble_eval_dir)
    analyzer.individual_dir = individual_eval_dir
    analyzer.load_results()
    
    # Create reports directory
    reports_dir = eval_dir / f'comparison_reports_{channels_str}'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reports
    analyzer.create_metrics_comparison_table(reports_dir / 'metrics_comparison.md')
    analyzer.create_performance_plots(reports_dir)
    analyzer.generate_summary_report(reports_dir / 'summary_report.md')
    
    log.info(f"\n✅ Comparison analysis complete. Reports saved to {reports_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate model comparison analysis')
    parser.add_argument('--eval-dir', type=str, default=None,
                        help='Evaluation directory (e.g., run_subset_01/outputs)')
    parser.add_argument('--channels', type=str, default='0_1_2',
                        help='Channel string identifier')
    args = parser.parse_args()
    
    if args.eval_dir is None:
        log.error("Please specify --eval-dir argument")
        return
    
    eval_dir = Path(args.eval_dir)
    if not eval_dir.exists():
        log.error(f"Evaluation directory not found: {eval_dir}")
        return
    
    analyze_eval_directory(eval_dir, args.channels)


if __name__ == '__main__':
    main()
