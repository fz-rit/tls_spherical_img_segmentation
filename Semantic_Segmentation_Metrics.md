
# Common Evaluation Metrics in Semantic Segmentation

Semantic segmentation tasks are commonly evaluated using a range of metrics that quantify how well the predicted segmentation matches the ground truth. These metrics include:

## 1. Pixel Accuracy (PA)
- Measures the proportion of correctly classified pixels:
  $$
  PA = \frac{\text{Number of Correctly Predicted Pixels}}{\text{Total Number of Pixels}}
  $$
- Simple and intuitive but may not reflect performance well in imbalanced datasets (e.g., where one class dominates).

---

## 2. Mean Pixel Accuracy (mPA)
- Computes the average of the pixel accuracy for each class:
  $$
  mPA = \frac{1}{N} \sum_{i=1}^{N} \frac{\text{Correctly Predicted Pixels for Class } i}{\text{Total Pixels in Class } i}
  $$
- Gives equal importance to each class.

---

## 3. Intersection over Union (IoU) / Jaccard Index
- Measures the overlap between the predicted segmentation and the ground truth:
  $$
  IoU = \frac{\text{True Positives}}{\text{True Positives + False Positives + False Negatives}}
  $$
- Commonly computed per class and averaged to get **Mean IoU (mIoU)**:
  $$
  mIoU = \frac{1}{N} \sum_{i=1}^{N} IoU_i
  $$
- A widely used metric that balances false positives and false negatives.

---

## 4. Dice Coefficient (F1 Score for Segmentation)
- Similar to IoU, but emphasizes agreement between predicted and ground truth regions:
  $$
  Dice = \frac{2 \times \text{True Positives}}{2 \times \text{True Positives} + \text{False Positives} + \text{False Negatives}}
  $$
- Commonly averaged across classes as **Mean Dice Coefficient**.

---

## 5. Boundary IoU
- Focuses on the alignment of boundaries between the predicted segmentation and the ground truth.
- Useful in applications where precise boundary prediction is crucial (e.g., medical imaging).

---

## 6. Frequency Weighted Intersection over Union (FWIoU)
- A weighted version of IoU that accounts for class imbalance:
  $$
  FWIoU = \frac{1}{\sum_{i=1}^{N} T_i} \sum_{i=1}^{N} T_i \cdot IoU_i
  $$
  where \( T_i \) is the total number of pixels in class \( i \).

---

## 7. Precision, Recall, and F1 Score
- Calculated per class:
  - **Precision**: $\frac{\text{True Positives}}{\text{True Positives + False Positives}}$
  - **Recall**: $\frac{\text{True Positives}}{\text{True Positives + False Negatives}}$
  - **F1 Score**: Harmonic mean of precision and recall.
- Useful for assessing class-specific performance.

---

## 8. Confusion Matrix Metrics
- The confusion matrix provides a comprehensive breakdown of true positives, false positives, false negatives, and true negatives for each class, enabling derived metrics like Specificity and Balanced Accuracy.

---

## 9. Structural Similarity Index Measure (SSIM)
- Occasionally used to assess the perceptual similarity between the segmentation map and the ground truth, especially in medical and natural scene segmentation.

---

## 10. Hausdorff Distance
- Measures the maximum distance between the predicted and ground truth boundaries, providing insights into the accuracy of boundary predictions.

---

## Use in Practice
- **Pixel Accuracy** and **mIoU** are the most common metrics in general-purpose segmentation tasks.
- **Dice Coefficient** is often preferred in medical imaging.
- **FWIoU** and **mPA** are suitable for handling class imbalance.
