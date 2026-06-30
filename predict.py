"""Fill this in. That's the whole interface.

Usage:
    python predict.py some_image.jpg
Prints ONE number from 0 to 1:
    0 = real photo,  1 = photo of a screen (recapture / fraud)
A hard 0 or 1 is fine if your method gives a yes/no answer.
"""

import sys
import os
import joblib
from PIL import Image

def predict(image_path: str) -> float:
    """
    Given one image path, returns how likely the image is a photo-of-a-screen.
    Returns a score in [0, 1].
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"File '{image_path}' does not exist.")
        
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please run train.py first.")
        
    # Load model and scaler
    payload = joblib.load(model_path)
    scaler = payload["scaler"]
    model = payload["model"]
    
    # Import feature extraction functions
    from features import extract_features
    
    # Extract features
    features = extract_features(image_path)
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict probability for class 1 (screen recapture)
    prob = model.predict_proba(features_scaled)[0][1]
    
    return float(prob)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Missing image path. Usage: python predict.py <image_path>", file=sys.stderr)
        sys.exit(1)
        
    try:
        score = predict(sys.argv[1])
        print(f"{score:.4f}")
    except Exception as e:
        print(f"Error processing image or running prediction: {e}", file=sys.stderr)
        sys.exit(1)
