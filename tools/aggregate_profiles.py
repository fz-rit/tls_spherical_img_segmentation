"""
Aggregate profiling results from multiple experiments into a single table.
Useful for creating tables for papers.
"""

import pandas as pd
import json
from pathlib import Path
from typing import List
import argparse


def collect_profile_jsons(root_dir: Path, pattern: str = "**/profile_*.json") -> List[Path]:
    """Find all profile JSON files recursively."""
    return sorted(root_dir.glob(pattern))


def load_and_flatten(json_path: Path) -> dict:
    """Load JSON and flatten nested dicts if needed."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def create_summary_table(profile_files: List[Path], output_csv: Path):
    """
    Create a summary table from profile JSONs.
    
    Columns:
    - Model name
    - Input channels
    - Params (M)
    - GFLOPs
    - Training time (min)
    - Inference time (ms)
    - Peak GPU (MB)
    - Best val mIoU
    - Best val oAcc
    """
    rows = []
    
    for pf in profile_files:
        data = load_and_flatten(pf)
        
        row = {
            'experiment': pf.parent.name,
            'model_name': data.get('model_name', 'N/A'),
            'input_channels': data.get('input_channels', 'N/A'),
            'train_subset': data.get('train_subset_cnt', 'N/A'),
            'params_M': round(data.get('trainable_params', 0) / 1e6, 2),
            'gflops': data.get('gflops', 0),
            'train_time_min': data.get('total_training_time_min', 0),
            'mean_epoch_time_sec': data.get('mean_epoch_time_sec', 0),
            'peak_gpu_mb': data.get('peak_memory_mb', 0),
            'best_epoch': data.get('best_epoch', 'N/A'),
            'best_val_loss': data.get('best_val_loss', 'N/A'),
            'final_val_oAcc': data.get('final_val_oAcc', 'N/A'),
            'final_val_mIoU': data.get('final_val_mIoU', 'N/A'),
            'gpu': data.get('gpu_name', 'N/A'),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by model and channels for readability
    if len(df) > 0:
        df = df.sort_values(['model_name', 'input_channels'])
    
    df.to_csv(output_csv, index=False)
    print(f"📊 Summary table saved to {output_csv}")
    print(f"   Total experiments: {len(df)}")
    print("\nPreview:")
    print(df.head(10))
    
    return df


def create_latex_table(df: pd.DataFrame, output_tex: Path):
    """
    Generate LaTeX table from DataFrame.
    Useful for direct inclusion in ISPRS/IEEE papers.
    """
    # Select key columns for paper
    paper_cols = [
        'model_name', 
        'input_channels',
        'params_M', 
        'gflops', 
        'train_time_min',
        'final_val_mIoU',
        'final_val_oAcc'
    ]
    
    df_paper = df[paper_cols].copy()
    
    # Rename for publication
    df_paper.columns = [
        'Model', 
        'Channels', 
        'Params (M)', 
        'GFLOPs', 
        'Train Time (min)',
        'mIoU',
        'OA'
    ]
    
    latex_str = df_paper.to_latex(
        index=False,
        float_format="%.2f",
        caption="Model Performance and Computational Metrics",
        label="tab:model_comparison"
    )
    
    output_tex.parent.mkdir(parents=True, exist_ok=True)
    with open(output_tex, 'w') as f:
        f.write(latex_str)
    
    print(f"📄 LaTeX table saved to {output_tex}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate profiling results')
    parser.add_argument('--root', type=str, required=True,
                        help='Root directory to search for profile JSONs')
    parser.add_argument('--output', type=str, default='profile_summary.csv',
                        help='Output CSV path')
    parser.add_argument('--latex', action='store_true',
                        help='Also generate LaTeX table')
    args = parser.parse_args()
    
    root = Path(args.root)
    output_csv = Path(args.output)
    
    print(f"🔍 Searching for profile JSONs in {root}...")
    profile_files = collect_profile_jsons(root)
    print(f"   Found {len(profile_files)} profile files")
    
    if len(profile_files) == 0:
        print("⚠️  No profile files found. Check your path.")
    else:
        df = create_summary_table(profile_files, output_csv)
        
        if args.latex:
            tex_path = output_csv.with_suffix('.tex')
            create_latex_table(df, tex_path)