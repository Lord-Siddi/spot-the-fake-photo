# Spot the Fake Photo - Binary Classifier

**Live Deployed Application**: [https://spot-the-fake-photo.onrender.com/](https://spot-the-fake-photo.onrender.com/)

This repository contains a lightweight, high-performance binary classifier designed to distinguish between **genuine first-generation photos** and **screen recaptures / photo-of-a-screen photos**.

The solution uses **28 handcrafted computer vision features** (frequency moiré peaks, subpixel layouts, color cast, JPEG blockiness, specular glare, sharpness distributions, and screen bezels) fed into a lightweight **Support Vector Machine (SVM RBF) Classifier** instead of a heavy deep learning CNN.

## Project Structure

```
.
├── data/
│   ├── real/                  # User-provided genuine photos
│   └── screen/                # User-provided screen recapture photos
├── features.py                # All handcrafted feature extraction functions
├── train.py                   # Loads dataset, extracts features, trains & evaluates the model
├── predict.py                 # CLI interface to predict a single image (outputs a single float)
├── benchmark.py               # Latency profiling and system benchmarking tool
├── model.pkl                  # Serialized best model and feature scaling statistics
├── NOTE.md                    # Half-page technical report with honest metrics and cost analysis
└── README.md                  # This documentation file
```

---

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Environment Setup
Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required packages:

```bash
pip install numpy opencv-python scikit-learn scipy joblib
```

---

## How to Run

### 1. Train the Model
Ensure your images are placed under `data/real/` and screen recapture directories. Run `train.py` to extract features, run cross-validation, train the SVM RBF model, and save `model.pkl`:

```bash
python train.py
```

### 2. Run Single-Image Inference
To run a prediction on an image, run `predict.py` with the path to the target image:

```bash
python predict.py data/real/some_image.jpeg
```
**Output**: Prints exactly one float between 0.0 and 1.0 representing the probability that the photo is a screen recapture (e.g., `0.0700` for a real photo, `0.9417` for a screen photo).

### 3. Run Performance Benchmark
To measure prediction latency and check your CPU specifications, run `benchmark.py`:

```bash
python benchmark.py
```

### 4. Run Live Demo
To launch the interactive web dashboard with camera access, start the local server:

```bash
python demo/server.py
```
Once the server is running, navigate to `http://localhost:5000/` in your web browser. This interface allows you to select, drag-and-drop, or browse an image file from your device, display a preview, and POST the image payload to the backend model for a real-time classification verdict.

---

## Handcrafted Features Extracted

1. **Moiré / 2D FFT Periodicity**: Evaluates the 2D Fast Fourier Transform magnitude spectrum of the grayscale image to detect peak energy concentrations in the mid-high frequency band, indicative of screen pixel grid alignment.
2. **RGB Subpixel Stripe Detection**: Checks high-frequency color variations between the red, green, and blue channels which are spatially offset on display hardware.
3. **JPEG Block / DCT Artifacts**: Measures differences across 8x8 block boundaries compared to internal differences to identify grid-aligned compression remnants from the source image.
4. **Color Cast & White Balance**: Analyzes channel means, standard deviations, and ratios (RGB and Lab color spaces) to detect typical screen light emission spectra.
5. **Specular Glare**: Detects small, highly-bright, and desaturated white light blobs using HSV thresholds, representing screen glass reflections.
6. **Sharpness Uniformity**: Calculates Laplacian variance over a 4x4 grid. Screens have uniform focus across the flat display surface, whereas physical scenes have depth-of-field focus variations.
7. **Bezel Detection**: Detects large convex quadrilaterals aligning with screen monitors or phone borders.
