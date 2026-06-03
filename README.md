# Bisindo Sign Language Recognition System

A real-time Indonesian Sign Language (BISINDO) alphabet recognition system. This project implements a **Pattern Recognition** method that combines spatial/geometric feature extraction with a classical machine learning classification algorithm.

## Dataset Information
The model in this system is trained using the public **[Indonesian Sign Language (BISINDO)](https://www.kaggle.com/datasets/agungmrf/indonesian-sign-language-bisindo)** dataset from Kaggle. This dataset contains thousands of images demonstrating sign language alphabets (A-Z), from which spatial features are extracted for the classification process.

## Methodology
Following pattern recognition system specifications, this project uses a **Combined Approach (Spatial + Classical Classification)**:
1. **Feature Extraction (Localization):** Utilizes **MediaPipe Hands** (CNN-based) to localize 21 hand landmarks in images/videos.
2. **Handcrafted Features:** Extracts the (X, Y, Z) coordinate points from the 21 landmarks into a 1D feature vector (totaling 63 spatial features per hand).
3. **Classification:** Uses a **Support Vector Machine (SVM) with an RBF Kernel** algorithm to classify these feature vectors into 26 alphabet classes (A-Z).

## System Features
* **Multi-Hand Tracking:** Capable of tracking and predicting signs from two hands (left and right) simultaneously and independently.
* **Temporal Smoothing:** Implements a state buffer (storing the history of the last 20 frames) with a 70% confirmation threshold to prevent screen flickering and ensure stable predictions (**LOCKED** state).
* **Real-time HUD:** An interactive webcam interface that displays bounding boxes, confidence probabilities, and tracking status in real-time.