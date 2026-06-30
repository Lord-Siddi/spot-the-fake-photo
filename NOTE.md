# Spot the Fake Photo - ML Classifier Report

## Approach Summary
We built a lightweight binary classifier using 28 handcrafted computer vision features pooled across 5 spatial crops (Center, Top-Left, Top-Right, Bottom-Left, Bottom-Right) feeding into a Support Vector Machine (SVM RBF) classifier. The features capture screen-recapture indicators: moiré patterns (via 2D FFT peak-to-mean ratio in the mid-high frequency band), high-frequency subpixel color offsets (via Laplacian variance of RGB channel differences), JPEG blockiness (via 8x8 grid boundary step-changes), color cast (via Lab and RGB channel means and ratios), specular glare (via HSV thresholded blobs), edge sharpness uniformity (via 4x4 grid Laplacian variance coefficient of variation), and monitor bezel detection (via Canny and convex quadrilateral contour analysis). SVM with RBF kernel and RobustScaler on Multi-Crop pooled features outperformed Random Forest and Gradient Boosting. This approach is highly interpretable, fast, and lightweight (the model is ~250 KB), making it ideal for mobile integration.

## Honest Accuracy Numbers
- **Stratified 5-Fold Cross-Validation Accuracy**: **92.50%** (on the 120 training samples, SVM RBF: C=10.0, gamma=0.01)
- **Held-out Test Accuracy**: **93.33%** (28/30 correct on the untouched test slice; 14/15 real correct, 14/15 screen/recapture correct)
- *Note: Performance metrics are measured on our updated, perfectly balanced dataset of 150 images partitioned into 75 real, 25 laptop screen, 25 mobile screen, and 25 printed paper recaptures.*

## Latency
- **Mean Latency**: **347.51 ms** per image.
- **Profiling Device**: Intel64 Family 6 Model 154 Stepping 3 CPU (laptop run under Windows 11).
- **Latency Breakdown**:
  - Image Loading/Decoding: **~180 ms** (representing 51.8% of runtime).
  - Multi-Crop Feature Extraction (5 crops evaluation): **~167 ms** (representing 48.0% of runtime).
  - Model Inference: **< 1 ms**.

## Cost per Image
- **On-Device (Mobile App)**: **$0.00** (runs client-side on the user's phone, utilizing their CPU free of charge).
- **Cloud Server**: **$0.08 per 1,000 images** ($80.00 per million images).
  - *Assumptions*: A small cloud instance (e.g., AWS `t3.medium` at ~$0.0416/hr) can serve sequentially at 347.51 ms/image (~10,350 images/hr). Raw compute cost is $4.02 per million images. Factoring in a 20x overhead margin for load balancing, network latency, and server idling, the final estimate is $80.00 per million images.

---
## Future Improvements
1. **Adaptive Resolution Loading**: Load images at half-resolution (`cv2.IMREAD_REDUCED_COLOR_2`) to cut image decoding times in half (~100ms savings) while validating that subpixel features remain discriminative.
2. **Dynamic Cutoff Thresholding**: Instead of a default 0.5 decision threshold, tune the probability threshold using precision-recall curves based on the business costs of false positives (user friction) vs. false negatives (fraud bypass).
3. **Adaptive Retraining**: As screens and recapture methods evolve, run active learning to query low-confidence samples, label them, and retrain the model to capture new screen technologies (e.g., higher-density displays).
