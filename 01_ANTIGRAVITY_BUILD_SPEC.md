# Build Spec: "Spot the Fake Photo" Take-Home (for Antigravity agent)

**Role:** You are an autonomous coding agent. Build a complete, working solution to the
SalesCode AI "Spot the Fake Photo" take-home challenge in this repo. Follow every
instruction below in order. Do not skip the report or the latency/cost numbers — they
are graded as heavily as the accuracy.

## 1. The task, restated precisely

- Input: one image.
- Output: a float in `[0, 1]` where `0 = real photo`, `1 = photo of a screen/printout` (a "recapture").
- Entry point must be runnable exactly as: `python predict.py some_image.jpg` → prints a single number.
- Target: **>95% accuracy** on photos we have never seen (held-out judge data, not our own training data).
- Must be small, fast, and eventually able to run on a phone (no huge models, no cloud GPU dependency required).
- Deliverables: code, a short note (half a page) with an honest accuracy number, latency in ms + device, and cost per image (on-device vs cloud at scale).

## 2. One thing you (the agent) cannot do yourself

You cannot take real-world photos with a phone camera. Data collection (Step 1 in the
assignment) requires a human with a phone. **Do not fabricate a dataset or silently
substitute internet stock photos and call it equivalent** — that would produce a
dishonest accuracy number, which the rubric explicitly penalizes.

What to do instead, in this order:
1. Check `/data/real/` and `/data/screen/` for user-supplied images first.
2. If they're empty or too small (<60 total images), **stop and ask the user** to follow
   `02_DATA_COLLECTION_GUIDE.md` before training. Don't proceed to `train.py` with no data.
3. While waiting, you may still build and unit-test the feature-extraction code using a
   handful of clearly-labeled placeholder/synthetic images (e.g. one photo of a monitor
   you generate procedurally with moiré-like stripes) purely to verify the code runs —
   but never report accuracy numbers from this synthetic data in the final note.

## 3. Recommended technical approach

Training a deep model is explicitly NOT required, and for a ~100-image dataset a deep
CNN will overfit and be slow/large anyway. The better play for this dataset size and
the "small, fast, cheap, honest" judging criteria is:

**Handcrafted features + a tiny classifier.** Extract ~15–25 numeric signals per image
that capture known recapture artifacts, then train a small, interpretable classifier
(logistic regression or gradient-boosted trees) on top of them. This is fast (no GPU),
tiny (kilobytes, not megabytes), explainable in the note, and realistic to port to a
phone.

### Signals to extract (implement each as its own function in `features.py`)

- **Moiré / periodicity (the strongest signal):** 2D FFT of the grayscale image; look
  for energy concentrated in a ring/peaks in the mid-high frequency band rather than
  smoothly decaying — this is the classic fingerprint of photographing a pixel grid.
  Score = ratio of peak energy in that band to total energy.
- **High-frequency energy ratio:** ratio of high-frequency to low-frequency energy from
  the FFT or via a high-pass filtered image's variance. Recaptures often show either an
  unnatural spike (moiré) or an unnatural drop (when downscaled/blurred screens are recaptured).
- **JPEG block / DCT artifact regularity:** 8x8 block DCT statistics; recaptures of a
  screen showing a compressed image often carry a second layer of compression artifacts.
- **Color cast / white balance stats:** mean and histogram skew per channel, and
  warm/cool bias. Screens emit light with characteristic spectra (often a blue or
  greenish cast, or oversaturated colors) vs. reflected ambient light on real objects.
- **Specular highlight / glare detection:** count and shape of small, very-bright,
  low-saturation blobs (thresholded on V channel in HSV) — screen glare looks different
  from natural specular highlights on real surfaces.
- **Edge sharpness profile (Laplacian variance):** real-world photos usually have a
  natural falloff of sharpness with depth; a recapture of a flat screen has a flatter,
  more uniform sharpness map, and a sharp rectangular bezel edge is sometimes visible.
- **Bezel / rectangle detection:** Canny edges + Hough line/contour detection for a
  large, nearly-perfect rectangle filling most of the frame (phone/monitor edge). If
  found, this is a strong standalone signal.
- **RGB subpixel stripe detection (close-up shots):** at high zoom, screens show regular
  RGB subpixel stripes; detect via a high-frequency color-channel periodicity check
  separate from the luminance moiré check above.
- **Color banding / quantization:** count of near-duplicate quantized color bins in
  smooth regions (screens + their source images often show banding that real scenes don't).

Compute each as a single float, normalize, and store as a flat feature vector.

### Classifier

- Use `scikit-learn`. Start with `LogisticRegression` (fast, interpretable coefficients —
  great for the "what'd you improve" section) and also try
  `GradientBoostingClassifier`/`RandomForestClassifier` as a comparison.
- Use stratified k-fold cross-validation (k=5) given the small dataset size, not a single
  train/test split — report the cross-validated accuracy honestly, and also hold out a
  final small test slice never touched during feature/model iteration.
- Save the trained model + feature normalization stats with `joblib` to `model.pkl`.

## 4. Project structure to create

```
.
├── data/
│   ├── real/                  # user-supplied photos
│   └── screen/                # user-supplied photos
├── features.py                # all feature-extraction functions
├── train.py                   # builds dataset, trains + cross-validates, saves model.pkl
├── predict.py                 # CLI: python predict.py image.jpg -> prints float 0..1
├── benchmark.py                # measures and prints latency per image
├── model.pkl                  # trained model + feature scaler (output of train.py)
├── demo/                       # optional live camera demo (see section 7)
│   └── index.html
├── NOTE.md                     # half-page honest report (see section 6)
└── README.md                   # how to set up and run everything
```

## 5. Step-by-step build order

1. Scaffold the folder structure above.
2. Implement `features.py` with one well-named, unit-testable function per signal in
   section 3, plus a top-level `extract_features(image_path) -> np.ndarray`.
3. Implement `train.py`:
   - Load all images from `data/real` (label 0) and `data/screen` (label 1).
   - Extract features for every image, build `X`, `y`.
   - Split off ~20% as a final untouched test set; cross-validate on the rest while
     tuning; report both CV accuracy and final test-set accuracy.
   - Print a confusion matrix and per-feature importance/coefficients.
   - Save model + scaler to `model.pkl`.
4. Implement `predict.py`:
   - `python predict.py path/to/image.jpg` loads `model.pkl`, extracts features from the
     one image, and prints **only** the probability score (0–1), nothing else, to stdout —
     match the assignment's exact expected output format.
   - Handle errors gracefully (missing file, unreadable image) without crashing silently
     into a wrong number.
5. Implement `benchmark.py`:
   - Run `predict.py`'s core function on ~20 images, discard the first (cold-start) run,
     average the rest, report mean ms per image and note the device/CPU used.
6. Write `NOTE.md` (section 6 below) using the **real, measured** numbers from steps 3–5.
7. (Optional, "impressive" bonus) Build the live camera demo (section 7).
8. Write `README.md` with exact setup/run commands (venv creation, `pip install`, how to
   run `train.py`, `predict.py`, `benchmark.py`, and the demo).

## 6. NOTE.md must contain (half a page, no more)

- **Approach in 3–4 sentences:** handcrafted frequency/color/geometry features (name them
  briefly) + a small classifier (name which one won), not a trained CNN — and why that's
  the right size/speed tradeoff for a phone.
- **Honest accuracy number:** the cross-validated AND held-out test accuracy from your own
  ~100-image dataset, clearly labeled as "on my own data, not SalesCode's held-out set."
  State the actual number, even if it's below 95% — do not round up or omit failure cases.
- **Latency:** the real measured number from `benchmark.py`, e.g. "~X ms on [CPU model]."
- **Cost per image:**
  - On-device: $0 (just stated, since the model is small enough to ship in-app).
  - Cloud server: a rough $/1,000 or $/1,000,000 images estimate. State the assumption
    explicitly (e.g. a small CPU instance at $X/hour serving Y images/sec).
- **What you'd improve with more time:** 2–3 concrete next steps (more data/devices,
  adding a learned CNN ensemble, calibrating the threshold with precision/recall curves, etc).
- **(If answering the "more experienced" bonus prompts)** briefly cover, in a few
  sentences each: how to keep accuracy up as cheaters adapt (collect new recapture
  techniques over time, retrain periodically, monitor score drift), how to shrink/speed
  it further for phones (quantize the classifier, run feature extraction in native code
  or TF Lite/Core ML, avoid full-resolution FFT by downsampling first), and how to pick
  the fraud cut-off (choose the threshold from a precision/recall or ROC curve based on
  the business cost of false positives vs false negatives, not just "0.5").

## 7. Optional live demo (only after everything above works)

A static HTML/JS page using `navigator.mediaDevices.getUserMedia` to grab camera frames,
draw to a `<canvas>`, and either:
- run the same handcrafted feature math directly in JS for a fully client-side demo, or
- POST the frame to a tiny local Flask/FastAPI endpoint that calls the Python `predict`
  function and returns the score.

Keep this a clearly separate, optional stretch — never let it block the core deliverable.

## 8. Acceptance checklist before declaring done

- [ ] `data/real/` and `data/screen/` contain real user-supplied photos (not placeholders)
- [ ] `python predict.py <image>` runs and prints exactly one float between 0 and 1
- [ ] `train.py` reports cross-validated AND held-out test accuracy, both printed and copied into `NOTE.md`
- [ ] `benchmark.py` reports a real measured latency number, copied into `NOTE.md`
- [ ] `NOTE.md` is ≤ half a page, has an honest accuracy number, latency, and cost-per-image with stated assumptions
- [ ] `README.md` lets a stranger set up and run everything from scratch
- [ ] No step silently fabricates data, accuracy, or latency numbers
