import os
import glob
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Import feature extraction functions
from features import extract_features, FEATURE_NAMES

def load_dataset(real_dir, screen_dirs):
    print("Scanning directories for images...")
    
    # Supported formats
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    
    real_paths = []
    for ext in extensions:
        real_paths.extend(glob.glob(os.path.join(real_dir, ext)))
    real_paths = sorted(list(set(os.path.abspath(p) for p in real_paths)))
        
    screen_paths = []
    for s_dir in screen_dirs:
        for ext in extensions:
            screen_paths.extend(glob.glob(os.path.join(s_dir, ext)))
    screen_paths = sorted(list(set(os.path.abspath(p) for p in screen_paths)))
        
    print(f"Found {len(real_paths)} real images and {len(screen_paths)} screen/recapture images.")
    
    X = []
    y = []
    
    # Process real images (Label: 0)
    print("Extracting features for REAL photos...")
    for idx, path in enumerate(real_paths):
        if idx % 10 == 0 or idx == len(real_paths) - 1:
            print(f"  [{idx + 1}/{len(real_paths)}] {os.path.basename(path)}")
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(0)
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            
    # Process screen recaptures (Label: 1)
    print("Extracting features for SCREEN/RECAPTURE photos...")
    for idx, path in enumerate(screen_paths):
        if idx % 10 == 0 or idx == len(screen_paths) - 1:
            print(f"  [{idx + 1}/{len(screen_paths)}] {os.path.basename(path)}")
        try:
            feats = extract_features(path)
            X.append(feats)
            y.append(1)
        except Exception as e:
            print(f"  Error processing {path}: {e}")
            
    return np.array(X), np.array(y)

def main():
    real_dir = r"data\real"
    screen_dirs = []
    for d in ["screen", "screen-laptop", "screen-mobile", "screen-print"]:
        path = os.path.join("data", d)
        if os.path.exists(path):
            screen_dirs.append(path)
            
    X, y = load_dataset(real_dir, screen_dirs)
    
    if len(X) < 10:
        print("Error: Too few images found to perform training. Exiting.")
        return
        
    print(f"\nDataset size: {X.shape[0]} samples, {X.shape[1]} features.")
    
    # Step 1: Split into 80% train/val and 20% final held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    print(f"Training split: {X_train.shape[0]} samples")
    print(f"Held-out test split: {X_test.shape[0]} samples")
    
    # Step 2: Scale features using RobustScaler (resilient to outliers in CV metrics)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 3: Stratified 5-Fold Cross-Validation & Grid Search
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Define models and their parameter grids for systematic search
    models_to_search = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, solver='liblinear'),
            {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2']
            }
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight='balanced', random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'n_estimators': [50, 100]
            }
        )
    }
    
    best_cv_mean = 0.0
    best_clf = None
    best_clf_name = None
    
    print("\n--- Running Systematic Hyperparameter Grid Search ---")
    for name, (model, param_grid) in models_to_search.items():
        print(f"Grid searching {name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=skf,
            scoring='accuracy',
            n_jobs=-1
        )
        grid_search.fit(X_train_scaled, y_train)
        
        mean_cv = grid_search.best_score_
        best_params = grid_search.best_params_
        print(f"  Best params: {best_params}")
        print(f"  Best CV Accuracy: {mean_cv:.4f}")
        
        if mean_cv > best_cv_mean:
            best_cv_mean = mean_cv
            best_clf = grid_search.best_estimator_
            best_clf_name = f"{name} ({best_params})"
            
    print(f"\nBest classifier based on CV: {best_clf_name} (Accuracy: {best_cv_mean:.4f})")
    
    # Evaluate on the untouched 20% held-out test set
    test_preds = best_clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_preds)
    test_cm = confusion_matrix(y_test, test_preds)
    
    print("\n--- Held-out Test Set Evaluation ---")
    print(f"Accuracy on Held-out Test Set: {test_acc:.4f}")
    print("Confusion Matrix:")
    print(test_cm)
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=["REAL", "SCREEN"]))
    
    # Print Feature Importance / Coefficients
    print("\n--- Feature Analysis ---")
    if hasattr(best_clf, "coef_"):
        coefs = best_clf.coef_[0]
        sorted_indices = np.argsort(np.abs(coefs))[::-1]
        print(f"Top feature coefficients for {best_clf_name}:")
        for i in sorted_indices[:15]:
            print(f"  {FEATURE_NAMES[i]:30s} : {coefs[i]:.4f}")
    elif hasattr(best_clf, "feature_importances_"):
        importances = best_clf.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        print(f"Top feature importances for {best_clf_name}:")
        for i in sorted_indices[:15]:
            print(f"  {FEATURE_NAMES[i]:30s} : {importances[i]:.4f}")
            
    # Save the model, scaler, and features metadata to model.pkl
    model_payload = {
        "scaler": scaler,
        "model": best_clf,
        "model_name": best_clf_name,
        "feature_names": FEATURE_NAMES
    }
    
    model_path = "model.pkl"
    joblib.dump(model_payload, model_path)
    print(f"\nModel and scaler successfully saved to {model_path}")
    
    print(f"\nSUMMARY_METRICS:")
    print(f"CV_ACCURACY = {best_cv_mean:.4f}")
    print(f"TEST_ACCURACY = {test_acc:.4f}")

if __name__ == "__main__":
    main()
