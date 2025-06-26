import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from tools.load_tools import get_color_map, get_pil_palette


# ---------- Load Image ---------- #
image_path = '/home/fzhcis/mylab/data/point_cloud_segmentation/segmentation_on_unwrapped_image/palau_2024/pca_outputs/7489/UMBCBL009_1830507489_image_cube_PCA_rgb_2_1_0.png'
img = cv2.imread(image_path)
if img is None:
    raise ValueError("Image not found: " + image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---------- Feature Construction ---------- #
h, w, _ = img.shape
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
xs = xs.astype(np.float32) / w
ys = ys.astype(np.float32) / h
img_norm = img.astype(np.float32) / 255.0
features = np.concatenate([img_norm, xs[..., np.newaxis], ys[..., np.newaxis]], axis=2)
features = features.reshape((-1, 5))
features[:, 3:] *= 0.5  # spatial weight

# ---------- K-means Clustering (K=6) ---------- #
K = 6
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
_, labels, _ = cv2.kmeans(np.float32(features), K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
labels = labels.flatten()
segmentation_map = labels.reshape((h, w))

# ---------- Heuristic Semantic Mapping ---------- #
areas = {i: np.sum(segmentation_map == i) for i in range(K)}
object_cluster = min(areas, key=areas.get)

cluster_y_means = {}
for i in range(K):
    if i == object_cluster:
        continue
    mask = (segmentation_map == i)
    y_vals = np.where(mask)[0]
    cluster_y_means[i] = np.mean(y_vals) if len(y_vals) > 0 else np.inf

sorted_clusters = sorted(cluster_y_means.items(), key=lambda x: x[1])

semantic_mapping = {i: 0 for i in range(K)}  # Default to Void
if len(sorted_clusters) >= 4:
    semantic_mapping[sorted_clusters[0][0]] = 3  # Canopy
    semantic_mapping[sorted_clusters[1][0]] = 2  # Stem
    semantic_mapping[sorted_clusters[2][0]] = 4  # Roots
    semantic_mapping[sorted_clusters[3][0]] = 1  # Ground
semantic_mapping[object_cluster] = 5  # Objects

seg_mask = np.zeros_like(segmentation_map, dtype=np.uint8)
for cluster_id, semantic_id in semantic_mapping.items():
    seg_mask[segmentation_map == cluster_id] = semantic_id

# ---------- Visualization with Matplotlib ---------- #
plt.figure(figsize=(10, 10))
plt.imshow(seg_mask, cmap=get_color_map(), vmin=0, vmax=5)
plt.title("Semantic Segmentation: Void, Ground, Stem, Canopy, Roots, Objects")
plt.axis('off')
plt.show()

# ---------- Save as Paletted PNG ---------- #
output_pil = Image.fromarray(seg_mask, mode='P')
output_pil.putpalette(get_pil_palette())
output_pil.save('segmentation_mask_paletted.png')
print("Saved as segmentation_mask_paletted.png (with PIL palette)")
