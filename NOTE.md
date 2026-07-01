# Spot the Fake Photo - ML Classifier Report

**Live Deployed Application**: [https://spot-the-fake-photo.onrender.com/](https://spot-the-fake-photo.onrender.com/)

## Approach Summary
I built a lightweight binary classifier using 28 handcrafted computer vision features feeding into a Support Vector Machine (SVM RBF) classifier. The features capture screen-recapture indicators: moiré patterns (via 2D FFT peak-to-mean ratio in the mid-high frequency band), high-frequency subpixel color offsets (via Laplacian variance of RGB channel differences), JPEG blockiness (via 8x8 grid boundary step-changes), color cast (via Lab and RGB channel means and ratios), specular glare (via HSV thresholded blobs), edge sharpness uniformity (via 4x4 grid Laplacian variance coefficient of variation), and monitor bezel detection (via Canny and convex quadrilateral contour analysis). SVM with an RBF kernel and StandardScaler on single-crop features outperformed Random Forest, Logistic Regression, and Gradient Boosting. This approach is highly interpretable, fast, and lightweight (the model is ~250 KB), making it ideal for mobile integration.

## Honest Accuracy Numbers
- **Stratified 5-Fold Cross-Validation Accuracy**: **93.33%** (on the 120 training samples, SVM RBF: C=10.0, gamma=0.01)
- **Held-out Test Accuracy**: **90.00%** (27/30 correct on the untouched test slice; 13/15 real correct, 14/15 screen/recapture correct)
- *Note: Performance metrics are measured on our updated, perfectly balanced dataset of 150 images partitioned into 75 real, 25 laptop screen, 25 mobile screen, and 25 printed paper recaptures.*

## Latency
- **Mean Latency**: **130.99 ms** per image.
- **Profiling Device**: Intel64 Family 6 Model 154 Stepping 3 CPU (laptop run under Windows 11).
- **Latency Breakdown**:
  - Image Loading/Decoding: **~85 ms** (representing 64.9% of runtime).
  - Feature Extraction (FFT, color stats, blockiness, etc.): **~45 ms** (representing 34.4% of runtime).
  - Model Inference: **< 1 ms**.

## Cost per Image
- **On-Device (Mobile App)**: **$0.00** (runs client-side on the user's phone, utilizing their CPU free of charge).
- **Cloud Server**: **$0.03 per 1,000 images** ($30.00 per million images).
  - *Assumptions*: A small cloud instance (e.g., AWS `t3.medium` at ~$0.0416/hr) can serve sequentially at 130.99 ms/image (~27,480 images/hr). Raw compute cost is $1.51 per million images. Factoring in a 15x overhead margin for load balancing, network latency, and server idling, the final estimate is $30.00 per million images.

---
## Future Improvements
1. **Adaptive Resolution Loading**: Load images at half-resolution (`cv2.IMREAD_REDUCED_COLOR_2`) to cut image decoding times in half (~100ms savings) while validating that subpixel features remain discriminative.
2. **Dynamic Cutoff Thresholding**: Instead of a default 0.5 decision threshold, tune the probability threshold using precision-recall curves based on the business costs of false positives (user friction) vs. false negatives (fraud bypass).
3. **Adaptive Retraining**: As screens and recapture methods evolve, run active learning to query low-confidence samples, label them, and retrain the model to capture new screen technologies (e.g., higher-density displays).
