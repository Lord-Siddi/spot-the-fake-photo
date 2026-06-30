# Spot the Fake Photo - ML Classifier Report

## Approach Summary
We built a lightweight binary classifier using 28 handcrafted computer vision features feeding into a Logistic Regression (L2) classifier. The features capture screen-recapture indicators: moiré patterns (via 2D FFT peak-to-mean ratio in the mid-high frequency band), high-frequency subpixel color offsets (via Laplacian variance of RGB channel differences), JPEG blockiness (via 8x8 grid boundary step-changes), color cast (via Lab and RGB channel means and ratios), specular glare (via HSV thresholded blobs), edge sharpness uniformity (via 4x4 grid Laplacian variance coefficient of variation), and monitor bezel detection (via Canny and convex quadrilateral contour analysis). Logistic Regression (L2) outperformed Random Forest and Gradient Boosting in cross-validation. This approach is highly interpretable, fast, and lightweight (the model is ~250 KB), making it ideal for mobile integration.

## Honest Accuracy Numbers
- **Stratified 5-Fold Cross-Validation Accuracy**: **86.67%** (on the 120 training samples, Logistic Regression L2)
- **Held-out Test Accuracy**: **96.67%** (29/30 correct on the untouched test slice; 14/15 real correct, 15/15 screen correct)
- *Note: Performance metrics are measured on our updated, perfectly balanced dataset of 150 images (75 real, 75 screen recaptures including laptops, mobiles, and paper prints).*

## Latency
- **Mean Latency**: **159.64 ms** per image.
- **Profiling Device**: Intel64 Family 6 Model 154 Stepping 3 CPU (laptop run under Windows 11).
- **Latency Breakdown**:
  - Image Loading/Decoding: **~105 ms** (representing 65.8% of runtime).
  - Feature Extraction (FFT, color stats, blockiness, etc.): **~54 ms** (representing 33.8% of runtime).
  - Model Inference: **< 1 ms**.

## Cost per Image
- **On-Device (Mobile App)**: **$0.00** (runs client-side on the user's phone, utilizing their CPU free of charge).
- **Cloud Server**: **$0.03 per 1,000 images** ($30.00 per million images).
  - *Assumptions*: A small cloud instance (e.g., AWS `t3.medium` at ~$0.0416/hr) can serve sequentially at 159.64 ms/image (~22,550 images/hr). Raw compute cost is $1.84 per million images. Factoring in a 15x overhead margin for load balancing, network latency, and server idling, the final estimate is $30.00 per million images.

---
## Future Improvements
1. **Adaptive Resolution Loading**: Load images at half-resolution (`cv2.IMREAD_REDUCED_COLOR_2`) to cut image decoding times in half (~100ms savings) while validating that subpixel features remain discriminative.
2. **Dynamic Cutoff Thresholding**: Instead of a default 0.5 decision threshold, tune the probability threshold using precision-recall curves based on the business costs of false positives (user friction) vs. false negatives (fraud bypass).
3. **Adaptive Retraining**: As screens and recapture methods evolve, run active learning to query low-confidence samples, label them, and retrain the model to capture new screen technologies (e.g., higher-density displays).
