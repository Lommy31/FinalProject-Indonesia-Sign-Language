import os
import pickle
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from src.config import (
    VAL_FEATURES_PATH,
    MODEL_PATH,
    SCALER_PATH,
    CLASSES,
    MODELS_DIR
)

def main():
    print("--- Evaluating SVM Model Performance ---")
    
    # Check if files exist
    if not os.path.exists(VAL_FEATURES_PATH):
        print(f"Error: Validation features not found at {VAL_FEATURES_PATH}. Please run preprocess.py first.")
        return
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"Error: Model or scaler not found. Please run train.py first.")
        return
        
    # Load validation data
    with open(VAL_FEATURES_PATH, "rb") as f:
        val_data = pickle.load(f)
    X_val = val_data["features"]
    y_val = val_data["labels"]
    
    # Load model and scaler
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Process and predict
    X_val_scaled = scaler.transform(X_val)
    y_pred = model.predict(X_val_scaled)
    
    # 1. Print classification report
    target_names = [CLASSES[idx] for idx in sorted(list(set(y_val)))]
    report = classification_report(y_val, y_pred, target_names=target_names)
    print("\n--- Detailed Classification Report ---")
    print(report)
    
    # Write report to a text file for documentation
    report_txt_path = os.path.join(MODELS_DIR, "classification_report.txt")
    with open(report_txt_path, "w") as f:
        f.write(report)
    print(f"Saved classification report text to: {report_txt_path}")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(15, 13))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d', ax=plt.gca())
    
    plt.title("Confusion Matrix - Bisindo Sign Language Alphabet Recognition", fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    cm_plot_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    
    print(f"Saved confusion matrix plot to: {cm_plot_path}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()
