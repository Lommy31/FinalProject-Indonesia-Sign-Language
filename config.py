import os

# Base Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Image Directories
TRAIN_IMAGES_DIR = os.path.join(BASE_DIR, "images", "train")
VAL_IMAGES_DIR = os.path.join(BASE_DIR, "images", "val")

# Target model and scaler file paths
MODEL_PATH = os.path.join(MODELS_DIR, "svm_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Preprocessed features output paths
TRAIN_FEATURES_PATH = os.path.join(DATA_DIR, "features_train.pkl")
VAL_FEATURES_PATH = os.path.join(DATA_DIR, "features_val.pkl")

# MediaPipe Hand Landmarks Settings
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Classes mapping (Alphabet A-Z)
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for idx, cls in enumerate(CLASSES)}

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
