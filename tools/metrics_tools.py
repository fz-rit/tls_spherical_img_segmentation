import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.metrics import hausdorff_distance, structural_similarity
from skimage.morphology import binary_erosion
from sklearn.metrics import confusion_matrix, mutual_info_score
from typing import Tuple
from sklearn.decomposition import PCA, FastICA
from pathlib import Path
from tools.logger_setup import Logger

log = Logger()

def compare_binary_image_metrics(src_img: np.ndarray, target_image: np.ndarray) -> dict:
    assert src_img.shape == target_image.shape, f"Binary images must have the same shape: {src_img.shape} vs {target_image.shape}"
    assert set(np.unique(src_img)).issubset({0, 1}), "Image 1 must be binary"
    assert set(np.unique(target_image)).issubset({0, 1}), "Image 2 must be binary"

    src_img = src_img.astype(bool)
    target_image = target_image.astype(bool)

    TP = np.sum((src_img == 1) & (target_image == 1))
    TN = np.sum((src_img == 0) & (target_image == 0))
    FP = np.sum((src_img == 0) & (target_image == 1))
    FN = np.sum((src_img == 1) & (target_image == 0))

    iou = TP / (TP + FP + FN) if (TP + FP + FN) != 0 else 0
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    # accuracy = (TP + TN) / (TP + TN + FP + FN)

    # Hausdorff Distance using skimage
    haus_dist = hausdorff_distance(src_img, target_image)

    # Structural Similarity Index (SSIM)
    ssim = structural_similarity(src_img.astype(float), target_image.astype(float), data_range=1.0, multichannel=False)


    return {
        "IoU": float(iou),
        "F1": float(f1),
        "Hausdorff_Distance": float(haus_dist),
        "SSIM": float(ssim),
        "Precision": float(precision),
        "Recall": float(recall)
    }


def compare_binary_maps(uncertainty_map: np.ndarray, error_map: np.ndarray) -> dict:
    """
    Compare uncertainty map with error map.
    Args:
        uncertainty_map (np.ndarray): Uncertainty map.
        error_map (np.ndarray): Error map.
    Returns:
        dict: Dictionary containing comparison metrics.
    """
    assert uncertainty_map.shape == error_map.shape, f"Uncertainty map and error map must have the same shape: {uncertainty_map.shape} vs {error_map.shape}"
    assert set(np.unique(error_map)).issubset({0, 1}), "Error map must be binary."
    thresholds = np.linspace(0.01, 0.50, 50)
    metrics_by_threshold = {'IoU': [], 
                            'F1': [], 
                            'Hausdorff_Distance': [], 
                            'SSIM': [], 
                            'Precision': [],
                            'Recall': []}
    
    for threshold in thresholds:
        uncertainty_map_binary = np.zeros_like(uncertainty_map)
        uncertainty_map_binary[uncertainty_map > threshold] = 1
        metrics_dict = compare_binary_image_metrics(error_map, uncertainty_map_binary)
        for key in metrics_dict:
            metrics_by_threshold[key].append(metrics_dict[key])

    assert len(metrics_by_threshold['IoU']) == len(thresholds), "Metrics length mismatch with thresholds"
    assert set(metrics_dict.keys()) == set(metrics_by_threshold.keys()), "Keys mismatch between metrics_dict and metrics_by_threshold"
    metrics_by_threshold['threshold'] = thresholds
    return metrics_by_threshold


def calculate_segmentation_statistics(true_flat: np.ndarray, 
                 pred_flat: np.ndarray, 
                 num_classes: int) -> dict:
    """
    Calculate evaluation metrics for semantic segmentation.
    """
    original_conf_mtx = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

    conf_mtx = original_conf_mtx.copy()
    # Keep only the classes that appear in either true or pred
    mask = (conf_mtx.sum(axis=0) + conf_mtx.sum(axis=1)) > 0
    conf_mtx = conf_mtx[mask][:, mask]

    intersection = np.diag(conf_mtx)
    total = conf_mtx.sum()
    oAccu = intersection.sum() / total if total > 0 else 0

    # Mean class Accuracy
    class_accuracy = intersection / np.maximum(conf_mtx.sum(axis=1), 1)
    mAcc = np.nanmean(class_accuracy)

    # IoU
    union = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0) - intersection
    IoU = np.round(intersection / np.maximum(union, 1),  4)
    mIoU = np.nanmean(IoU)

    # Frequency Weighted IoU
    freq = conf_mtx.sum(axis=1) / total if total > 0 else np.zeros_like(conf_mtx.sum(axis=1))
    FWIoU = np.sum(freq * IoU)

    # Dice Coefficient
    denom = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0)
    dice = 2 * intersection / np.maximum(denom, 1)
    dice_coefficient = np.nanmean(dice)

    return {
        'confusion_matrix': original_conf_mtx,
        'oAcc': round(oAccu, 4),
        'mAcc': round(mAcc, 4),
        'mIoU': round(mIoU, 4),
        'IoU_per_class': IoU,
        'FWIoU': round(FWIoU, 4),
        'dice_coefficient': round(dice_coefficient, 4)
    }



def calc_oAccu_mIoU(true_flat: np.ndarray,
                    pred_flat: np.ndarray,
                    num_classes: int) -> Tuple[float, float]:
    """
    Calculate overall accuracy and mean intersection over union (IoU).

    Args:
    true_flat (numpy.ndarray): Flattened ground truth mask (1D array).
    pred_flat (numpy.ndarray): Flattened predicted mask (1D array).
    num_classes (int): Number of classes in the dataset.

    Returns:
    oAccu (float): Overall accuracy.
    mIoU (float): Mean intersection over union.
    """

    metrics_dict = calculate_segmentation_statistics(true_flat, pred_flat, num_classes)
    oAcc = metrics_dict['oAcc']
    mIoU = metrics_dict['mIoU']
    return oAcc, mIoU


def compute_band_correlation(image):
    """
    Computes the correlation matrix between spectral bands of a multichannel image.

    Parameters:
    -----------
    image : np.ndarray
        A numpy array of shape (C, H, W) representing the multichannel image.
        C is the number of spectral bands.

    Returns:
    --------
    corr_matrix : np.ndarray
        A (C, C) correlation matrix between the spectral bands.
    """
    if image.ndim != 3:
        raise ValueError("Input image must have 3 dimensions (C, H, W)")

    C = image.shape[0]
    reshaped = image.reshape(C, -1)  # Flatten spatial dimensions
    corr_matrix = np.corrcoef(reshaped)

    return corr_matrix

def compute_pca_components(image, n_components=3):
    """
    Applies PCA to a (C, H, W) image cube.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W).
    n_components : int
        Number of principal components to compute.

    Returns:
    --------
    pcs : np.ndarray
        Principal components reshaped to (n_components, H, W).
    pca : sklearn.decomposition.PCA
        The fitted PCA object (contains explained variance, components, etc.).
    """
    if image.ndim != 3:
        raise ValueError("Input image must be 3D (C, H, W)")

    C, H, W = image.shape
    reshaped = image.reshape(C, -1).T  # Shape: (H*W, C)

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(reshaped)  # Shape: (H*W, n_components)
    pcs = pca_result.T.reshape(n_components, H, W)  # (n_components, H, W)

    return pcs, pca

def compute_mnf(image, noise_estimation=True, n_components=3):
    """
    Computes the Minimum Noise Fraction (MNF) transform.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W), where C is the number of bands.
    noise_estimation : bool
        If True, estimate noise as difference between adjacent pixels (simple method).
    n_components : int
        Number of MNF components to return.

    Returns:
    --------
    mnf_components : np.ndarray
        MNF components of shape (n_components, H, W).
    """
    C, H, W = image.shape
    X = image.reshape(C, -1).T  # Shape: (H*W, C)

    if noise_estimation:
        noise = X[1:] - X[:-1]
    else:
        noise = np.random.normal(0, 1, X.shape)

    noise_cov = np.cov(noise.T)
    signal_cov = np.cov(X.T)

    eigvals, eigvecs = np.linalg.eigh(np.linalg.inv(noise_cov) @ signal_cov)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]

    mnf_data = X @ eigvecs[:, :n_components]
    mnf_components = mnf_data.T.reshape(n_components, H, W)

    return mnf_components

def compute_ica(image, n_components=3):
    """
    Applies Independent Component Analysis (ICA) to a (C, H, W) image cube.

    Parameters:
    -----------
    image : np.ndarray
        Input image of shape (C, H, W).
    n_components : int
        Number of ICA components to compute.

    Returns:
    --------
    ica_components : np.ndarray
        ICA components of shape (n_components, H, W).
    """
    C, H, W = image.shape
    reshaped = image.reshape(C, -1).T  # Shape: (H*W, C)

    ica = FastICA(n_components=n_components, random_state=0)
    ica_result = ica.fit_transform(reshaped)
    ica_components = ica_result.T.reshape(n_components, H, W)

    return ica_components


def uncertainty_vs_error(eval_results, verbose=True):

    true_mask = eval_results['true_mask']
    pred_mask = eval_results['pred_mask']
    uncertainty_map = eval_results['mutual_info']
    error_map = np.zeros_like(true_mask)
    error_map[true_mask != pred_mask] = 1
    metrics_dict = compare_binary_maps(uncertainty_map, error_map)

    if verbose:
        log.info(f"Metrics comparing uncertainty map with error map: {metrics_dict}")

    return uncertainty_map, error_map, metrics_dict


if __name__ == "__main__":
    img1 = np.random.randint(0, 2, (256, 256))
    img2 = np.random.randint(0, 2, (256, 256))

    metrics = compare_binary_image_metrics(img1, img2)
    print(metrics)

