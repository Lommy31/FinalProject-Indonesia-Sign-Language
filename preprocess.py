import os
import cv2
import pickle
import numpy as np
from src.config import (
    TRAIN_IMAGES_DIR,
    VAL_IMAGES_DIR,
    TRAIN_FEATURES_PATH,
    VAL_FEATURES_PATH,
    CLASSES,
    CLASS_TO_IDX
)
from src.feature_extractor import HandFeatureExtractor

def preprocess_dataset_split(split_name, images_dir, output_path):
    print(f"\n--- Preprocessing {split_name.upper()} split ---")
    print(f"Reading from: {images_dir}")
    print(f"Saving to: {output_path}")
    
    extractor = HandFeatureExtractor(static_image_mode=True, max_num_hands=1)
    
    X = []
    y = []
    
    total_images = 0
    detected_hands = 0
    
    # Iterate over classes A-Z
    for cls in CLASSES:
        cls_dir = os.path.join(images_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"Warning: Directory for class {cls} does not exist at {cls_dir}")
            continue
            
        cls_idx = CLASS_TO_IDX[cls]
        img_files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Processing Class {cls} ({len(img_files)} images)...")
        
        for img_name in img_files:
            total_images += 1
            img_path = os.path.join(cls_dir, img_name)
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Failed to read image: {img_path}")
                continue
                
            # Extract landmarks and features
            _, landmarks = extractor.extract_landmarks(img)
            if landmarks is not None:
                features = extractor.extract_features(landmarks)
                if features is not None:
                    X.append(features)
                    y.append(cls_idx)
                    detected_hands += 1
            
    extractor.close()
    
    X = np.array(X)
    y = np.array(y)
    
    # Calculate detection rate
    detection_rate = (detected_hands / total_images) * 100 if total_images > 0 else 0
    print(f"\nSummary for {split_name.upper()}:")
    print(f"  Total images processed: {total_images}")
    print(f"  Hands successfully detected: {detected_hands} ({detection_rate:.2f}%)")
    print(f"  Feature shape: {X.shape if len(X) > 0 else 'Empty'}")
    
    # Save features to pkl
    if len(X) > 0:
        data = {
            "features": X,
            "labels": y
        }
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved preprocessed data to {output_path}")
    else:
        print("Error: No features were extracted. Nothing to save.")

def main():
    # Preprocess train split
    preprocess_dataset_split("train", TRAIN_IMAGES_DIR, TRAIN_FEATURES_PATH)
    
    # Preprocess val split
    preprocess_dataset_split("val", VAL_IMAGES_DIR, VAL_FEATURES_PATH)

if __name__ == "__main__":
    main()
