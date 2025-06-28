import matplotlib.pyplot as plt
from tools.load_tools import get_color_map, get_pil_palette, get_label_map
from tools.metrics_tools import calculate_segmentation_statistics, uncertainty_vs_error, calc_precision_recall_curve
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.cm import get_cmap
import matplotlib.colors as colors
from tools.logger_setup import Logger

log = Logger()

plt.rcParams.update({
    'font.size': 12,         # base font size
    'axes.titlesize': 12,    # title size
    'axes.labelsize': 10,    # x/y label size
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 12
})

def save_mask_as_image(mask: np.ndarray, mono_path: Path=None, color_path: Path=None):
    """
    Save the mask as a monochrome and color image.

    Args:
    mask (np.ndarray): Mask array.
    mono_path (Path): Path to save the monochrome image.
    color_path (Path): Path to save the color image.
    """
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask)
    if mono_path:
        mask.save(mono_path)
        log.info(f"☯️ Monochrome mask saved to {mono_path.name}")

    if color_path:
        mask.putpalette(get_pil_palette())
        mask.save(color_path)
        log.info(f"🌈 Color mask saved to {color_path.name}")

    assert mono_path or color_path, "At least one path must be provided to save the mask."


def write_eval_metrics_to_file(eval_metrics: dict, out_dir: Path, key_str:str=''):
    """
    Write evaluation metrics to a text file.
    
    Args:
        eval_metrics (dict): Dictionary containing evaluation metrics.
        out_dir (Path): Directory to save the metrics file.
    """
    metrics_path = out_dir / f"eval_metrics_{key_str}.txt"
    with open(metrics_path, 'w') as f:
        for key, value in eval_metrics.items():
            f.write(f"{key}: {value}\n")
    log.info(f"🗒️ Evaluation metrics saved to {metrics_path.parent.name}/{metrics_path.name}")

def visualize_eval_output(img, 
                          eval_results: dict, 
                          input_channels, 
                          num_classes,
                          gt_available=True, 
                          out_dir: Path = None):
    """"
    Visualize the image and masks.
    """
    input_channels_str = '_'.join([str(ch) for ch in input_channels])
    num_subplots = len(eval_results) + 1
    fig, axs = plt.subplots(num_subplots, 1, figsize=(8, 4* num_subplots))
    color_map, _ = get_color_map()
    subplot_titles = ['Input Image']  # Title for the input image subplot
    subplot_titles += [key for key in eval_results.keys()]  # Titles for predicted masks
    for i in range(len(subplot_titles)):
        if subplot_titles[i] == 'Input Image':
            axs[i].imshow(img)
        elif 'mask' in subplot_titles[i]:
            mask = eval_results[subplot_titles[i]]
            axs[i].imshow(mask, cmap=color_map, vmin=0, vmax=num_classes - 1, interpolation='nearest')
        else:
            axs[i].imshow(eval_results[subplot_titles[i]], cmap='hot', interpolation='nearest')
            cbar = plt.colorbar(axs[i].images[0], ax=axs[i], orientation='horizontal', fraction=0.08, pad=0.04, aspect=20)
        axs[i].set_title(subplot_titles[i])
        axs[i].axis('off')        
    plt.tight_layout()
    output_path = out_dir / "combined_eval.png"
    fig.savefig(output_path)
    log.info(f"😌 Evaluation maps saved to {output_path.name}")

    pred_mask_color_path = out_dir / f"pred_mask_color.png"
    save_mask_as_image(eval_results['pred_mask'], color_path = pred_mask_color_path)

    if gt_available:
        eval_metrics_dict = calculate_segmentation_statistics(true_flat =  eval_results['true_mask'].flatten(),
                                                        pred_flat = eval_results['pred_mask'].flatten(),
                                                        num_classes = num_classes)
        write_eval_metrics_to_file(eval_metrics_dict, out_dir, key_str=input_channels_str)
    

def plot_training_validation_losses(train_losses, val_losses, plt_save_path, clip_val_loss=True):
    """
    Plot and save training and validation losses.

    Parameters:
    - train_losses (list or np.array): Training loss values over epochs.
    - val_losses (list or np.array): Validation loss values over epochs.
    - plt_save_path (str): Path to save the plot image.
    - clip_val_loss (bool): Whether to clip validation loss for clearer plots.
    """

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)

    # Prevent empty input or division by zero
    if len(train_losses) == 0 or len(val_losses) == 0:
        log.info("Error: Empty loss lists.")
        return

    if clip_val_loss:
        max_train = max(train_losses)
        val_losses = val_losses.clip(max=3*max_train)
        log.info(f"Clipping validation loss to max of 3 times the max training loss: {3*max_train}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', marker='o')
    plt.plot(val_losses, label='Val Loss' + (' (clipped)' if clip_val_loss else ''), marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)

    # Ensure save directory exists
    plt_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plt_save_path)
    plt.close()

    log.info(f"---- Loss plot saved at: {plt_save_path} ----")


def plot_training_validation_metrics(train_oAccus, val_oAccus, train_mIoUs, val_mIoUs, plt_save_path):
    """
    Plot and save training and validation metrics.

    Parameters:
    - train_oAccus (list or np.array): Training overall accuracy values over epochs.
    - val_oAccus (list or np.array): Validation overall accuracy values over epochs.
    - train_mIoUs (list or np.array): Training mean IoU values over epochs.
    - val_mIoUs (list or np.array): Validation mean IoU values over epochs.
    - plt_save_path (str): Path to save the plot image.
    """
    train_oAccus = np.array(train_oAccus)
    val_oAccus = np.array(val_oAccus)
    train_mIoUs = np.array(train_mIoUs)
    val_mIoUs = np.array(val_mIoUs)

    # Prevent empty input or division by zero
    if len(train_oAccus) == 0 or len(val_oAccus) == 0:
        log.info("Error: Empty metric lists.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(train_oAccus, label='Train Overall Accuracy', marker='o')
    plt.plot(val_oAccus, label='Val Overall Accuracy', marker='o')
    plt.plot(train_mIoUs, label='Train Mean IoU', marker='x')
    plt.plot(val_mIoUs, label='Val Mean IoU', marker='x')

    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.grid(True)

    # Ensure save directory exists
    plt_save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plt_save_path)
    plt.close()

    log.info(f"---- Metrics plot saved at: {plt_save_path} ----")



def compare_uncertainty_with_error_map(eval_results,
                                       output_dir: Path = None, 
                                       save_maps: bool = False):
    """
    Compare uncertainty map with error map and visualize metrics as scatter plots.
    
    Args:
    uncertainty_map (np.ndarray): Uncertainty map.
    error_map (np.ndarray): Error map.
    
    Returns:
    None
    """
    uncertainty_map, error_map, metrics_by_threshold = uncertainty_vs_error(eval_results, verbose=False)

    if save_maps:
        uncertainty_map_path = output_dir / "uncertainty_map.png"
        error_map_path = output_dir / "error_map.png"
        plt.imsave(uncertainty_map_path, uncertainty_map, cmap='hot', vmin=uncertainty_map.min(), vmax=uncertainty_map.max())
        plt.imsave(error_map_path, error_map, cmap='gray', vmin=0, vmax=1)
        log.info(f"🗺️ Uncertainty map saved to {uncertainty_map_path.name}")
        log.info(f"🗺️ Error map saved to {error_map_path.name}")

    for title_str in ['total_uncertainty', 'var_based_epistemic', 'mutual_info']:
        if title_str in eval_results:
            log.info(f"Plotting precision-recall curve for {title_str}")
            plot_precision_recall_curve(y_true=error_map, 
                                        y_scores=eval_results[title_str], 
                                        output_dir=output_dir, title_str=title_str)
        else:
            raise ValueError(f"Title string '{title_str}' not found in eval_results.")

    fig = plt.figure(figsize=(16, 10))
    outer = gridspec.GridSpec(2, 2, height_ratios=[2, 3], hspace=0.3, wspace=0.3)

    # === First row: Uncertainty & Error maps ===
    ax1 = plt.subplot(outer[0, 0])
    unct_map_im = ax1.imshow(uncertainty_map, cmap='hot', interpolation='nearest')
    ax1.set_title('Uncertainty Map')
    ax1.axis('off')

    ax2 = plt.subplot(outer[0, 1])
    ax2.imshow(error_map, cmap='gray', vmax=1, interpolation='nearest')
    ax2.set_title('Error Map (0-miss, 1-hit)')
    ax2.axis('off')

    # === Second row: 8 sub-subplots ===
    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=outer[1, :], wspace=0.4, hspace=0.4)

    metric_names = ['F1', 'Precision', 'Recall', 'IoU', 'SSIM', 'Hausdorff_Distance']
    observe_minmax = [1, 1, 1, 1, 1, 0] # 1: observe max, 0: observe min
    thresholds = metrics_by_threshold['threshold']

    for i, metric in enumerate(metric_names):
        ax = plt.subplot(inner_grid[i])
        values = metrics_by_threshold[metric]
        if observe_minmax[i]:
            obs_val = max(values)
            obs_str = 'max'
            th = thresholds[np.argmax(values)]
        else:
            obs_val = min(values)
            obs_str = 'min'
            th = thresholds[np.argmin(values)]
        ax.set_title(f"{metric}({obs_str}={obs_val:.3f}; th={th:.2f})", fontsize=8)
        ax.scatter(thresholds, values, s=10)
        ax.scatter(th, obs_val, s=20, c='red', marker='x')
        ax.set_xlabel('Threshold', fontsize=8)
        ax.set_ylabel(metric, fontsize=8)
        ax.tick_params(labelsize=6)

    plt.colorbar(unct_map_im, ax=ax1, orientation='vertical', fraction=0.02, pad=0.04)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"uncertainty_error_{timestamp}.png"
    fig.savefig(output_path)
    log.info(f"😉Uncertainty and error map saved to {output_path}")
    # plt.show()

    return metrics_by_threshold


def plot_precision_recall_curve(y_true, y_scores, output_dir: Path = None, 
                                title_str: str = ""):
    """
    Plot and save the precision-recall curve.

    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth binary labels.

    y_scores : np.ndarray
        Predicted scores or probabilities.

    output_dir : Path
        Directory to save the plot.

    title : str
        Title of the plot.

    output_stem : str
        Stem for the output filename.
    """

    pr_dict = calc_precision_recall_curve(y_true, y_scores)  # Ensure the function is called to compute precision and recall
    fig = plt.figure(figsize=(4, 3))
    plt.plot(pr_dict['recall'], pr_dict['precision'], marker='o', label=f"AUPRC={pr_dict['auprc']:.3f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title_str)
    plt.grid(True)
    plt.legend()

    output_path = output_dir / f"precision_recall_curve_{title_str}.png"
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    log.info(f"📈 Precision-Recall curve saved to {output_path}")
    
    plt.close()



def plot_correlation_matrix(corr_matrix, 
                            band_names=None, 
                            title="Correlation Matrix of Feature Maps",
                            output_stem=None):
    """
    Plots the correlation matrix as a heatmap.

    Parameters:
    -----------
    corr_matrix : np.ndarray
        A (C, C) correlation matrix.

    band_names : list of str, optional
        A list of names for the spectral bands (length C). If None, band indices will be used.

    title : str
        Title of the heatmap.
    """
    C = corr_matrix.shape[0]
    if band_names is None:
        band_names = [f'Band {i}' for i in range(C)]

    assert len(band_names) == C, f"Length of band names {len(band_names)} must match the number of bands {C}."
    corr_fig = plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                xticklabels=band_names, yticklabels=band_names,
                square=True, cbar_kws={"shrink": 0.75})
    plt.title(title)
    plt.tight_layout()

    output_path = Path(f"outputs/correlation_matrix_{output_stem}.png")
    corr_fig.savefig(output_path)
    log.info(f"1️Correlation matrix saved to {output_path}")


def plot_pca_components(pcs, output_stem = None):
    """
    Plots the PCA/ICA/MNF components as single channel images.

    Parameters:
    -----------
    pcs : np.ndarray
        Principal components of shape (n_components, H, W).
    """
    n_components = pcs.shape[0]
    fig, axes = plt.subplots(n_components, 1, figsize=(6, 3 * n_components))
    if n_components == 1:
        axes = [axes]

    for i in range(n_components):
        axes[i].imshow(pcs[i], cmap='plasma') # cmaps: 'gray', 'hot', 'cool', 'viridis', 'plasma', 'inferno'
        axes[i].set_title(f'Component {i+1}')
        axes[i].axis('off')

    if 'PCA' in output_stem:
        plt.suptitle("PCA Components")
    elif 'MNF' in output_stem:
        plt.suptitle("MNF Components")
    elif 'ICA' in output_stem:
        plt.suptitle("ICA Components")
    else:
        plt.suptitle("Unknown Components")
    plt.tight_layout()

    output_path = Path(f"outputs/pca_components_{output_stem}.png")
    fig.savefig(output_path)
    log.info(f"2️PCA/MNF/ICA components saved to {output_path}")
    

def plot_rgb_permutations(components, output_stem=None):
    """
    Plots RGB images from all permutations of the first 3 components.

    Parameters:
    -----------
    components : np.ndarray
        Component images of shape (3, H, W).
    title : str
        Title prefix for each subplot.
    """
    from itertools import permutations

    permuts = list(permutations([0, 1, 2]))
    H, W = components.shape[1:]
    fig, axes = plt.subplots(len(permuts), 1, figsize=(6, 3 * len(permuts)))

    for ax, perm in zip(axes, permuts):
        rgb = np.stack([components[i] for i in perm], axis=-1)
        # Normalize each channel
        for i in range(3):
            ch = rgb[:, :, i]
            rgb[:, :, i] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

        # save rgb image
        rgb = (rgb * 255).astype(np.uint8)
        rgb_img = Image.fromarray(rgb)
        rgb_img.save(f"outputs/{output_stem}_rgb_{perm[0]}_{perm[1]}_{perm[2]}.png")
        ax.imshow(rgb)
        ax.set_title(f"{output_stem} rgb permutations\nR:Comp{perm[0]+1} G:Comp{perm[1]+1} B:Comp{perm[2]+1}")
        ax.axis('off')

    plt.tight_layout()
    
    if output_stem:
        output_path = f"outputs/{output_stem}_PCs_permutations.png"
        fig.savefig(output_path)
        log.info(f"3️Saved RGB permutations plot to {output_path}")


def plot_channel_histograms(image_cube, channel_names=None, bins=256, colormap='tab10'):
    """
    Plots histogram for each of 8 channels in a 2x4 grid with shared x-axis.

    Parameters:
    - image_cube: np.ndarray of shape (8, H, W)
    - channel_names: optional list of channel names for titles
    - bins: number of histogram bins (default 256)
    """
    assert image_cube.shape[0] == 8, "This function assumes 8 channels"
    cmap = get_cmap(colormap)
    colors = [cmap(i) for i in range(8)]  # Get 8 distinct colors from the colormap

    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True)
    axes = axes.flatten()

    for i in range(8):
        ax = axes[i]
        ax.hist(image_cube[i].ravel(), bins=bins, color=colors[i], alpha=0.75)
        title = f'Channel {i}' if channel_names is None else channel_names[i]
        ax.set_title(title)
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        # ax.set_yscale('log')
        ax.grid(True)

    plt.tight_layout()