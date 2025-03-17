# tls_unwrapped_image_segmentation
Semantic segmentation on the unwrapped image generated from TLS point clouds.



## Evaluation Metrics
### **1. Overall Accuracy (OA)**
**Definition**: OA is the ratio of correctly predicted points to the total number of points.

#### **Formula**:
\[
OA = \frac{\sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)}{N}
\]
Where:
- \(N\) is the total number of points in the dataset.
- \(\hat{y}_i\) is the predicted label for point \(i\).
- \(y_i\) is the ground truth label for point \(i\).
- \(\mathbb{1}(\hat{y}_i = y_i)\) is an indicator function that returns 1 if the prediction is correct and 0 otherwise.

**Interpretation**: Higher OA means that a larger proportion of points were correctly classified.


### **2. Mean Accuracy (mAcc) in 3D Point Cloud Semantic Segmentation**

**Definition**: Mean Accuracy (mAcc) is the average per-class accuracy, meaning it calculates the accuracy for each class separately and then averages them. It helps evaluate how well the model performs on each class independently, unlike Overall Accuracy (OA), which is biased towards dominant classes.

#### **Formula**:
For each class \( c \), accuracy is computed as:
\[
Acc_c = \frac{TP_c}{TP_c + FN_c}
\]
Where:
- \(TP_c\) = True Positives (correctly predicted points for class \(c\))
- \(FN_c\) = False Negatives (ground truth points of class \(c\) that were misclassified)

The **Mean Accuracy (mAcc)** is then computed as:
\[
mAcc = \frac{1}{C} \sum_{c=1}^{C} Acc_c
\]
Where \( C \) is the number of classes.

### **3. Mean Intersection over Union (mIoU)**
**Definition**: The IoU for each class is the ratio of correctly predicted points (true positives) to the union of ground truth and predicted points for that class. The mean IoU (mIoU) is the average IoU across all classes.

#### **Formula**:
For each class \(c\), the IoU is:
\[
IoU_c = \frac{TP_c}{TP_c + FP_c + FN_c}
\]
Where:
- \(TP_c\) = True Positives (points correctly predicted as class \(c\))
- \(FP_c\) = False Positives (points wrongly predicted as class \(c\))
- \(FN_c\) = False Negatives (points of class \(c\) wrongly classified as another class)

The **mean IoU (mIoU)** is:
\[
mIoU = \frac{1}{C} \sum_{c=1}^{C} IoU_c
\]
Where \(C\) is the number of classes.


---


### **Comparison with OA and mIoU**
| Metric | Formula | What it Measures | Strengths | Weaknesses |
|--------|---------|------------------|-----------|------------|
| **Overall Accuracy (OA)** | \( \frac{\sum_{i=1}^{N} \mathbb{1}(\hat{y}_i = y_i)}{N} \) | Fraction of correctly classified points over all points | Simple and intuitive | Biased toward dominant classes |
| **Mean Accuracy (mAcc)** | \( \frac{1}{C} \sum_{c=1}^{C} Acc_c \) | Average of per-class accuracies | More balanced evaluation across classes | Does not consider false positives |
| **Mean IoU (mIoU)** | \( \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c + FN_c} \) | Intersection-over-Union per class, then averaged | Best for class imbalance, penalizes both FP and FN | More complex computation |

### **When to Use Each Metric**
- **OA** is useful when class imbalance is not an issue.
- **mAcc** is better when evaluating per-class performance.
- **mIoU** is the most balanced and commonly used metric for segmentation tasks.




