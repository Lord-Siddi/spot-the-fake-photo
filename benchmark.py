import os
import time
import glob
import numpy as np
import platform
import joblib
import cv2

# Import feature extraction functions
from features import extract_features

def main():
    # Gather sample images (10 real, 5 laptop screen, 5 mobile screen)
    real_paths = glob.glob(os.path.join("data", "real", "*.jpeg"))[:10]
    laptop_paths = glob.glob(os.path.join("data", "screen", "*.jpeg"))[:5]
    mobile_paths = glob.glob(os.path.join("data", "screen-mobile", "*.jpeg"))[:5]
    
    test_paths = real_paths + laptop_paths + mobile_paths
    if len(test_paths) == 0:
        print("Error: No test images found.")
        return
        
    print(f"Benchmarking on {len(test_paths)} images...")
    
    # Load model and scaler once
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        print("Error: model.pkl not found. Please train the model first.")
        return
        
    payload = joblib.load(model_path)
    scaler = payload["scaler"]
    model = payload["model"]
    
    latencies = []
    
    # Warm-up (cold-start discard)
    print("Running cold-start warm-up...")
    _ = extract_features(test_paths[0])
    
    # Core benchmark loop
    print("Measuring latencies...")
    for idx, path in enumerate(test_paths):
        start_time = time.perf_counter()
        
        # Pipeline execution
        features = extract_features(path)
        features_scaled = scaler.transform(features.reshape(1, -1))
        _ = model.predict_proba(features_scaled)[0][1]
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000.0
        latencies.append(elapsed_ms)
        print(f"  Image {idx+1:02d}: {elapsed_ms:.2f} ms | Path: {os.path.basename(path)}")
        
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    
    device_info = platform.processor() or platform.machine()
    print("\n--- Benchmark Results ---")
    print(f"Device: {device_info}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Number of runs: {len(test_paths)}")
    print(f"Mean Latency: {mean_latency:.2f} ms per image")
    print(f"Std Dev: {std_latency:.2f} ms")
    
    print(f"\nLATENCY_METRIC = {mean_latency:.2f} ms")

if __name__ == "__main__":
    main()
