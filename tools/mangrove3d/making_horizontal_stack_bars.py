import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np

# Set publication-quality style
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 3,
    'xtick.major.width': 0.5,
    'ytick.major.size': 3,
    'ytick.major.width': 0.5,
    'legend.frameon': False,
    'legend.fontsize': 7,
    'figure.dpi': 300
})

# ------Image Cube Class Counts------
classes = ['Ground & Water', 'Root', 'Canopy', 'Stem', 'Object', 'Void']
train_val_counts = [9756642, 6756962, 3733188, 1355628, 297663, 1427917]
test_counts  = [2427764, 2799529,  954145,  300423,  73090, 443449]
output_key_str = "Pixels"

# # ------PCD Class Counts------
# classes = ['Ground & Water', 'Root', 'Canopy', 'Stem', 'Object']
# train_val_counts = [9894609, 7378971, 4931957, 1466620, 299341]
# test_counts  = [2453250, 3044917, 1213787, 323756, 73675]
# output_key_str = "Points"


output_dir = 'outputs'
# Create DataFrame with calculated percentages
df = pd.DataFrame({
    'Class': classes,
    'train_val': train_val_counts,
    'test': test_counts
}).set_index('Class')

# Calculate total and percentages for annotations
df['Total'] = df['train_val'] + df['test']
df['train_val_pct'] = (df['train_val'] / df['Total'] * 100).round(1)
df['test_pct'] = (df['test'] / df['Total'] * 100).round(1)

# Plot setup - Nature/Science standard figure size (single column ~3.5 inches)
fig, ax = plt.subplots(figsize=(3.5, 2.8))

# Bar positions
y_pos = np.arange(len(df))

# Nature/Science style color palette - more conservative and colorblind-friendly
train_val_color = "#2166ac"  # Blue
test_color  = "#762a83"  # Purple

# Bar height for professional appearance
bar_height = 0.6

# Plot stacked horizontal bars
bars1 = ax.barh(y_pos, df['train_val'], height=bar_height, label='Train & Validation', 
                color=train_val_color, alpha=0.8, edgecolor='white', linewidth=0.5)
bars2 = ax.barh(y_pos, df['test'], left=df['train_val'], height=bar_height, 
                label='Test', color=test_color, alpha=0.8, edgecolor='white', linewidth=0.5)

# Y-axis class labels with proper spacing
ax.set_yticks(y_pos)
ax.set_yticklabels(df.index, fontsize=8)
ax.invert_yaxis()  # Invert to match typical scientific figures

# X-axis formatting
ax.set_xlabel(f'Number of {output_key_str}', fontsize=8, fontweight='normal')

# Human-readable tick formatter optimized for scientific publications
def scientific_format(x, pos):
    """Format numbers in scientific notation appropriate for publications"""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}'
    else:
        return f'{x:.0f}'

# Set up x-axis
ax.xaxis.set_major_formatter(ticker.FuncFormatter(scientific_format))
ax.tick_params(axis='x', labelsize=7, pad=2)
ax.tick_params(axis='y', labelsize=7, pad=2)

# Set x-axis limits with some padding
max_val = df['Total'].max()
ax.set_xlim(0, max_val * 1.1)

# Legend positioned appropriately for scientific figures
legend = ax.legend(loc='lower right', frameon=False, fontsize=7, 
                  handlelength=1.5, handletextpad=0.5, columnspacing=1)

# Remove spines for clean appearance typical of Nature/Science
for spine in ax.spines.values():
    spine.set_visible(False)
    
# Add subtle grid for better readability
ax.grid(True, axis='x', alpha=0.3, linewidth=0.5, linestyle='-')
ax.set_axisbelow(True)

# Optimize layout for publication
plt.tight_layout(pad=0.5)

# Optional: Add sample size annotations for scientific rigor
for i, (idx, row) in enumerate(df.iterrows()):
    # Add total count annotation at the end of each bar
    total = row['Total']
    ax.text(total + max_val * 0.01, i, f'n={total/1000:.0f}k', 
            va='center', ha='left', fontsize=6, color='gray')

# Save with publication-quality settings
plt.savefig(f'{output_dir}/{output_key_str}_distribution.pdf', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.savefig(f'{output_dir}/{output_key_str}_distribution.png', dpi=300, bbox_inches='tight', 
            facecolor='white', edgecolor='none')

plt.show()
