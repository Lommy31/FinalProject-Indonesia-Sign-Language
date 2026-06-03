import os
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from config import (
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    CLASSES
)

def load_data(features_path):
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}. Please run preprocess.py first.")
        
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    return data["features"], data["labels"]

def main():
    print("--- Loading Preprocessed Features ---")
    try:
        X_train, y_train = load_data(TRAIN_FEATURES_PATH)
        X_val, y_val = load_data(VAL_FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    print(f"Train set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Val set:   {X_val.shape[0]} samples, {X_val.shape[1]} features")
    
    # 1. Feature Scaling (Standardization)
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # 2. Classifier Training (Support Vector Machine)
    print("\nTraining Support Vector Classifier (SVM) with RBF kernel...")
    # C=10, gamma='scale', probability=True (so we can get confidence scores in realtime webcam)
    model = SVC(kernel='rbf', C=10.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on Train
    train_preds = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Accuracy: {train_acc * 100:.2f}%")
    
    # Evaluate on Validation
    val_preds = model.predict(X_val_scaled)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    
    # Print brief classification report
    print("\nClassification Report (Validation Split):")
    target_names = [CLASSES[idx] for idx in sorted(list(set(y_val)))]
    print(classification_report(y_val, val_preds, target_names=target_names))
    
    # 3. Save model and scaler
    print(f"\nSaving model to: {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    print(f"Saving scaler to: {SCALER_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    
    print("Training process finished successfully!")

if __name__ == "__main__":
    main()
