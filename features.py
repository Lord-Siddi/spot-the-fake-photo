import cv2
import numpy as np

def extract_fft_features(gray_crop):
    """
    Computes FFT-based features to detect moiré pattern peaks and energy distributions.
    """
    f = np.fft.fft2(gray_crop)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    
    # Clip to avoid zero division/log errors
    magnitude = np.clip(magnitude, 1e-6, None)
    
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    
    # Coordinate grid for distances from the center
    y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
    r = np.sqrt(x*x + y*y)
    
    max_r = min(cy, cx)
    
    # Define masks for frequency bands
    low_mask = r < 0.1 * max_r
    mid_mask = (r >= 0.1 * max_r) & (r < 0.5 * max_r)
    high_mask = (r >= 0.5 * max_r) & (r < 0.9 * max_r)
    mid_high_mask = (r >= 0.1 * max_r) & (r < 0.9 * max_r)
    
    # High frequency to low frequency energy ratio
    low_energy = np.sum(magnitude[low_mask])
    high_energy = np.sum(magnitude[high_mask])
    ratio_hf = high_energy / (low_energy + 1e-6)
    
    # Peak energy ratio in the mid-to-high frequency bands (typical of screen grid moiré)
    mid_high_vals = magnitude[mid_high_mask]
    if len(mid_high_vals) > 0:
        peak_val = np.max(mid_high_vals)
        mean_val = np.mean(mid_high_vals)
        peak_to_mean = peak_val / (mean_val + 1e-6)
    else:
        peak_to_mean = 0.0
        
    return [ratio_hf, peak_to_mean]

def extract_subpixel_features(rgb_crop):
    """
    Detects high-frequency variations in R-G and G-B differences caused by
    physical RGB pixel layouts on screens.
    """
    r = rgb_crop[:, :, 0].astype(float)
    g = rgb_crop[:, :, 1].astype(float)
    b = rgb_crop[:, :, 2].astype(float)
    
    diff_rg = r - g
    diff_gb = g - b
    
    lap_rg = cv2.Laplacian(diff_rg, cv2.CV_64F)
    lap_gb = cv2.Laplacian(diff_gb, cv2.CV_64F)
    
    var_rg = np.var(lap_rg)
    var_gb = np.var(lap_gb)
    
    return [var_rg, var_gb]

def extract_banding_features(gray_crop):
    """
    Measures quantization banding typical of recaptured displays through gradient stats.
    """
    gx = cv2.Sobel(gray_crop, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_crop, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    
    total_pixels = grad_mag.size
    # Ratio of flat or low-gradient pixels
    flat_pixels = np.sum(grad_mag < 1.0)
    flat_ratio = flat_pixels / float(total_pixels)
    
    # Standard deviation of gradient magnitude
    grad_std = np.std(grad_mag)
    
    return [flat_ratio, grad_std]

def extract_color_cast_features(rgb_resized):
    """
    Extracts white balance, color channel ratios, and Lab color statistics.
    """
    r = rgb_resized[:, :, 0].astype(float)
    g = rgb_resized[:, :, 1].astype(float)
    b = rgb_resized[:, :, 2].astype(float)
    
    mean_r = np.mean(r)
    mean_g = np.mean(g)
    mean_b = np.mean(b)
    
    std_r = np.std(r)
    std_g = np.std(g)
    std_b = np.std(b)
    
    ratio_rg = mean_r / (mean_g + 1e-6)
    ratio_bg = mean_b / (mean_g + 1e-6)
    ratio_rb = mean_r / (mean_b + 1e-6)
    
    # Convert to Lab
    lab = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0].astype(float)
    a_chan = lab[:, :, 1].astype(float)
    b_chan = lab[:, :, 2].astype(float)
    
    mean_a = np.mean(a_chan)
    mean_b_lab = np.mean(b_chan)
    std_a = np.std(a_chan)
    std_b_lab = np.std(b_chan)
    
    return [
        mean_r, mean_g, mean_b, 
        std_r, std_g, std_b, 
        ratio_rg, ratio_bg, ratio_rb, 
        mean_a, mean_b_lab, 
        std_a, std_b_lab
    ]

def extract_hsv_color_features(rgb_resized):
    """
    Extracts HSV saturation and value statistics to check for gamut compression (common in paper prints).
    """
    hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1].astype(float)
    v = hsv[:, :, 2].astype(float)
    
    mean_s = np.mean(s)
    std_s = np.std(s)
    mean_v = np.mean(v)
    std_v = np.std(v)
    
    return [mean_s, std_s, mean_v, std_v]

def extract_glare_features(rgb_resized):
    """
    Isolates and counts potential glare regions (high brightness, low saturation blobs).
    """
    hsv = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    
    # Glare condition: V is high (bright highlight), S is low (desaturated color)
    glare_mask = (v > 220) & (s < 50)
    glare_ratio = np.sum(glare_mask) / float(v.size)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(glare_mask.astype(np.uint8))
    num_blobs = max(0, num_labels - 1)
    
    if num_blobs > 0:
        avg_blob_size = np.mean(stats[1:, cv2.CC_STAT_AREA]) / float(v.size)
    else:
        avg_blob_size = 0.0
        
    return [glare_ratio, float(num_blobs), avg_blob_size]

def extract_sharpness_features(gray_crop):
    """
    Measures global sharpness (Laplacian variance) and focus uniformity across grid cells
    on the raw 512x512 crop to avoid any resizing/interpolation low-pass blur.
    Also computes Canny edge density to measure high-frequency detail content.
    """
    lap = cv2.Laplacian(gray_crop, cv2.CV_64F)
    global_var = np.var(lap)
    
    # 4x4 grid sharpness distribution (each cell is 128x128)
    sh, sw = 128, 128
    variances = []
    for i in range(4):
        for j in range(4):
            patch = gray_crop[i*sh:(i+1)*sh, j*sw:(j+1)*sw]
            patch_lap = cv2.Laplacian(patch, cv2.CV_64F)
            variances.append(np.var(patch_lap))
            
    variances = np.array(variances)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    coef_var = std_var / (mean_var + 1e-6)
    
    # Canny edge density
    edges = cv2.Canny(gray_crop, 30, 150)
    edge_density = np.sum(edges > 0) / float(edges.size)
    
    return [global_var, coef_var, edge_density]

def extract_blockiness_features(gray_crop):
    """
    Detects 8x8 grid boundary pixel step-changes (JPEG/compression artifacts) versus
    the internal differences within block grids.
    """
    # Horizontal differences
    diff_h = np.abs(gray_crop[:, :-1].astype(float) - gray_crop[:, 1:])
    cols = np.arange(diff_h.shape[1])
    boundary_cols = (cols + 1) % 8 == 0
    internal_cols = ~boundary_cols
    
    mean_boundary_h = np.mean(diff_h[:, boundary_cols])
    mean_internal_h = np.mean(diff_h[:, internal_cols])
    block_h = mean_boundary_h / (mean_internal_h + 1e-6)
    
    # Vertical differences
    diff_v = np.abs(gray_crop[:-1, :].astype(float) - gray_crop[1:, :])
    rows = np.arange(diff_v.shape[0])
    boundary_rows = (rows + 1) % 8 == 0
    internal_rows = ~boundary_rows
    
    mean_boundary_v = np.mean(diff_v[boundary_rows, :])
    mean_internal_v = np.mean(diff_v[internal_rows, :])
    block_v = mean_boundary_v / (mean_internal_v + 1e-6)
    
    return [block_h, block_v]

def extract_bezel_features(gray_resized):
    """
    Attempts to identify monitor / phone bezel shapes (large convex quadrilaterals).
    """
    blurred = cv2.bilateralFilter(gray_resized, 9, 75, 75)
    edges = cv2.Canny(blurred, 30, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_quad_area = 0.0
    bezel_found = 0.0
    img_area = float(gray_resized.shape[0] * gray_resized.shape[1])
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 0.05 * img_area:
                bezel_found = 1.0
                if area > best_quad_area:
                    best_quad_area = area
                    
    best_quad_fraction = best_quad_area / img_area
    return [bezel_found, best_quad_fraction]

def extract_features(image_path):
    """
    Extracts a 33-dimensional flat feature vector from an image.
    Uses full resolution loading with 512x512 crops to preserve subpixel details for peak accuracy.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Failed to load image: {image_path}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = img_gray.shape
    
    # 1. Extract 512x512 center crops (essential to preserve subpixel/moiré detail)
    cy, cx = h // 2, w // 2
    y_start, y_end = max(0, cy - 256), min(h, cy + 256)
    x_start, x_end = max(0, cx - 256), min(w, cx + 256)
    
    gray_crop = img_gray[y_start:y_end, x_start:x_end]
    rgb_crop = img_rgb[y_start:y_end, x_start:x_end]
    
    # Pad crops if the image is smaller than 512x512
    if gray_crop.shape != (512, 512):
        pad_y = 512 - gray_crop.shape[0]
        pad_x = 512 - gray_crop.shape[1]
        gray_crop = np.pad(gray_crop, ((0, pad_y), (0, pad_x)), mode='edge')
        rgb_crop = np.pad(rgb_crop, ((0, pad_y), (0, pad_x), (0, 0)), mode='edge')
        
    # 2. Resized images for global color/geometry details (512x384)
    img_rgb_resized = cv2.resize(img_rgb, (512, 384))
    img_gray_resized = cv2.resize(img_gray, (512, 384))
    
    # Extract feature components
    features = []
    
    features.extend(extract_fft_features(gray_crop))
    features.extend(extract_subpixel_features(rgb_crop))
    features.extend(extract_banding_features(gray_crop))
    features.extend(extract_color_cast_features(img_rgb_resized))
    features.extend(extract_hsv_color_features(img_rgb_resized))   # 4 features
    features.extend(extract_glare_features(img_rgb_resized))
    features.extend(extract_sharpness_features(gray_crop))          # 3 features (calculated on raw crop)
    features.extend(extract_blockiness_features(gray_crop))
    features.extend(extract_bezel_features(img_gray_resized))
    
    return np.array(features, dtype=float)

# List of feature names for tracking and analysis
FEATURE_NAMES = [
    "fft_high_low_ratio", "fft_moire_peak_to_mean",
    "subpixel_var_rg", "subpixel_var_gb",
    "banding_flat_ratio", "banding_grad_std",
    "color_mean_r", "color_mean_g", "color_mean_b",
    "color_std_r", "color_std_g", "color_std_b",
    "color_ratio_rg", "color_ratio_bg", "color_ratio_rb",
    "color_mean_a", "color_mean_b_lab", "color_std_a", "color_std_b_lab",
    "hsv_mean_s", "hsv_std_s", "hsv_mean_v", "hsv_std_v",
    "glare_ratio", "glare_num_blobs", "glare_avg_blob_size",
    "sharpness_global_var", "sharpness_coef_var", "sharpness_edge_density",
    "block_h", "block_v",
    "bezel_found", "bezel_quad_fraction"
]
