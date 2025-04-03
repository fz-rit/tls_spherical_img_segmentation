import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt
from skimage.metrics import hausdorff_distance, structural_similarity
from skimage.morphology import binary_erosion
from sklearn.metrics import confusion_matrix, mutual_info_score
from typing import Tuple


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

    # # Mutual Information
    # mutual_info = mutual_info_score(img1.ravel(), img2.ravel())

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
    conf_mtx = confusion_matrix(true_flat, pred_flat, labels=np.arange(num_classes))

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
    IoU = intersection / np.maximum(union, 1)
    mIoU = np.nanmean(IoU)

    # Frequency Weighted IoU
    freq = conf_mtx.sum(axis=1) / total if total > 0 else np.zeros_like(conf_mtx.sum(axis=1))
    FWIoU = np.sum(freq * IoU)

    # Dice Coefficient
    denom = conf_mtx.sum(axis=1) + conf_mtx.sum(axis=0)
    dice = 2 * intersection / np.maximum(denom, 1)
    dice_coefficient = np.nanmean(dice)

    return {
        'confusion_matrix': conf_mtx,
        'oAcc': oAccu,
        'mAcc': mAcc,
        'mIoU': mIoU,
        'FWIoU': FWIoU,
        'dice_coefficient': dice_coefficient
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

if __name__ == "__main__":
    img1 = np.random.randint(0, 2, (256, 256))
    img2 = np.random.randint(0, 2, (256, 256))

    metrics = compare_binary_image_metrics(img1, img2)
    print(metrics)

