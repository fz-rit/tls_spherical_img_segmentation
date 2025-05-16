import matplotlib.pyplot as plt
from tools.load_tools import custom_cmap, get_pil_palette
from tools.metrics_tools import calculate_segmentation_statistics, compare_binary_maps
import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.cm import get_cmap


def save_mask_as_image(mask: np.ndarray, mono_path: Path, color_path: Path):
    """
    Save the mask as a monochrome and color image.

    Args:
    mask (np.ndarray): Mask array.
    mono_path (Path): Path to save the monochrome image.
    color_path (Path): Path to save the color image.
    """
    mask = mask.astype(np.uint8)
    mask_mono = Image.fromarray(mask)
    mask_mono.save(mono_path)
    print(f"📸Monochrome mask saved to {mono_path}")

    mask_color = Image.fromarray(mask)
    mask_color.putpalette(get_pil_palette())
    mask_color.save(color_path)
    print(f"🎨Color mask saved to {color_path}")


def visualize_eval_output(img, 
                          true_mask, 
                          pred_mask, 
                          input_channels, 
                          num_classes,
                          label_map=None,
                          gt_available=True, 
                          output_path: Path = None):
    """"
    Visualize the image and masks.
    """
    input_channels_str = '_'.join([str(ch) for ch in input_channels])

    # Compute metrics between true_mask and pred_mask
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    num_subplots = 4 if gt_available else 3

    # fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 6))
    fig, axs = plt.subplots(
            num_subplots, 1, 
            figsize=(8, 10), 
            gridspec_kw={'height_ratios': [1, 1, 1, 2] if gt_available else None
                         }  # last plot (conf matrix) gets more height
        )
    axs_img, axs_true, axs_pred = axs[0], axs[1], axs[2]

    if gt_available:
        metric_dict = calculate_segmentation_statistics(true_flat, pred_flat, num_classes)
        oAcc, mAcc, mIoU, FWIoU, dice_coefficient = metric_dict['oAcc'], metric_dict['mAcc'], metric_dict['mIoU'], metric_dict['FWIoU'], metric_dict['dice_coefficient']
        confusion_matrix = metric_dict['confusion_matrix']
        pred_title = ' '.join(['Predicted Mask\n',
                    f'oAcc: {oAcc:.4f};',
                    f'mAcc: {mAcc:.4f};',
                    f'mIoU: {mIoU:.4f};',
                    f'FWIoU: {FWIoU:.4f};',
                    f'dice_coeff: {dice_coefficient:.4f}'])
        true_title = 'Ground Truth Mask' 
    else:
        pred_title = 'Predicted Mask (No GT)'
        true_title = 'Ground Truth Mask (Not Available)'

    axs_img.imshow(img)
    axs_img.set_title(f'Input Image {input_channels_str}')
    axs_img.axis('off')

    # For masks, use a discrete colormap to distinguish classes
    axs_true.imshow(true_mask, cmap=custom_cmap(), vmin=0, vmax=num_classes - 1, interpolation='nearest')
    axs_true.set_title(true_title)
    axs_true.axis('off')

    axs_pred.imshow(pred_mask, cmap=custom_cmap(), vmin=0, vmax=num_classes - 1, interpolation='nearest')
    axs_pred.set_title(pred_title)
    axs_pred.axis('off')

    # Plot confusion matrix in axs_confmtx
    if gt_available:
        axs_confmtx = axs[3]
        # label_map = {
        #             0: 'Void',
        #             1: 'Ground & Water',
        #             2: 'Stem',
        #             3: 'Canopy',
        #             4: 'Roots',
        #             5: 'Objects'
        #         }
        # # sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axs_confmtx)
        # # axs_confmtx.set_title('Confusion Matrix')
        # # axs_confmtx.set_xlabel('Predicted Class')
        # # axs_confmtx.set_ylabel('True Class')
        # # axs_confmtx.set_xticklabels([f'{label_map[i]}' for i in range(num_classes)], rotation=45)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=axs_confmtx, cbar=True)
        axs_confmtx.set_title('Confusion Matrix')
        axs_confmtx.set_xlabel('Predicted Class')
        axs_confmtx.set_ylabel('True Class')
        # axs_confmtx.set_xticklabels([label_map[i] for i in range(num_classes)], rotation=30, ha='right')
        # axs_confmtx.set_yticklabels([label_map[i] for i in range(num_classes)], rotation=0)


    plt.tight_layout()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"outputs/output_{timestamp}.png") if output_path is None else output_path
    fig.savefig(output_path)
    print(f"😌Segmentation map saved to {output_path}")

    # Save the pred_mask in rbg image.
    pred_mask_mono_path = output_path.parent / f"pred_mask_mono_{timestamp}.png"
    pred_mask_color_path = output_path.parent / f"pred_mask_color_{timestamp}.png"
    save_mask_as_image(pred_mask, pred_mask_mono_path, pred_mask_color_path)


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
        print("Error: Empty loss lists.")
        return

    if clip_val_loss:
        max_train = max(train_losses)
        val_losses = val_losses.clip(max=3*max_train)
        print(f"Clipping validation loss to max of 3 times the max training loss: {3*max_train}")

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

    print(f"---- Loss plot saved at: {plt_save_path} ----")


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
        print("Error: Empty metric lists.")
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

    print(f"---- Metrics plot saved at: {plt_save_path} ----")



def compare_uncertainty_with_error_map(uncertainty_map: np.ndarray, 
                                       error_map: np.ndarray,
                                       output_path: Path = None):
    """
    Compare uncertainty map with error map and visualize metrics as scatter plots.
    
    Args:
    uncertainty_map (np.ndarray): Uncertainty map.
    error_map (np.ndarray): Error map.
    
    Returns:
    None
    """
    # Simulate or fetch the metrics_by_threshold
    metrics_by_threshold = compare_binary_maps(uncertainty_map, error_map)
    
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
    output_path = Path(f"outputs/uncertainty_error_{timestamp}.png") if output_path is None else output_path
    fig.savefig(output_path)
    print(f"😉Uncertainty and error map saved to {output_path}")
    # plt.show()

    return metrics_by_threshold



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
    print(f"1️Correlation matrix saved to {output_path}")


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
    print(f"2️PCA/MNF/ICA components saved to {output_path}")
    

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
        print(f"3️Saved RGB permutations plot to {output_path}")


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