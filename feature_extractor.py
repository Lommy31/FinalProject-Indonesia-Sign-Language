import cv2
import mediapipe as mp
import numpy as np
from config import MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE

class HandFeatureExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=1):
        """
        Initializes the MediaPipe Hands model.
        
        Args:
            static_image_mode (bool): Whether to treat the input images as a batch of static
                                      and possibly unrelated images, or a video stream.
            max_num_hands (int): Maximum number of hands to detect.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, image_bgr):
        """
        Processes a BGR image to find hand landmarks (first hand only, for compatibility).
        
        Args:
            image_bgr: OpenCV BGR image.
            
        Returns:
            results: MediaPipe hands detection result object.
            landmarks_list: List of 21 landmark tuples (x, y, z) if detected, else None.
        """
        # Convert the BGR image to RGB before processing.
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        if results.multi_hand_landmarks:
            # We assume the first detected hand is the primary hand for classification
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks_list = []
            for lm in hand_landmarks.landmark:
                landmarks_list.append((lm.x, lm.y, lm.z))
            return results, landmarks_list
        
        return results, None

    def extract_all_hands_landmarks(self, image_bgr):
        """
        Processes a BGR image and returns landmarks for ALL detected hands.
        
        Args:
            image_bgr: OpenCV BGR image.
            
        Returns:
            results: MediaPipe hands detection result object.
            hands_data: List of dicts with keys 'landmarks' and 'handedness' ('Left'/'Right').
                        Empty list if no hands detected.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        hands_data = []
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_list = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                handedness = "Right"
                if results.multi_handedness and idx < len(results.multi_handedness):
                    # MediaPipe flips labels when image is mirrored
                    label = results.multi_handedness[idx].classification[0].label
                    handedness = "Left" if label == "Right" else "Right"
                hands_data.append({
                    "landmarks": landmarks_list,
                    "handedness": handedness,
                })
        return results, hands_data

    def get_hand_bbox(self, landmarks_list, img_h, img_w):
        """
        Calculates pixel bounding box coordinates for hand drawing/cropping.
        """
        xs = [lm[0] * img_w for lm in landmarks_list]
        ys = [lm[1] * img_h for lm in landmarks_list]
        x_min, x_max = int(min(xs)), int(max(xs))
        y_min, y_max = int(min(ys)), int(max(ys))
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img_w, x_max + padding)
        y_max = min(img_h, y_max + padding)
        
        return x_min, y_min, x_max, y_max

    def draw_landmarks(self, image_bgr, mp_results):
        """
        Draws the detected hand landmarks and connections on the BGR image.
        """
        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    image_bgr, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        return image_bgr

    def extract_features(self, landmarks_list):
        """
        Extracts translation- and scale-normalized handcrafted features from hand landmarks.
        
        Args:
            landmarks_list (list): List of 21 (x, y, z) tuples.
            
        Returns:
            np.array: Feature vector of shape (93,).
        """
        if landmarks_list is None or len(landmarks_list) != 21:
            return None
        
        landmarks = np.array(landmarks_list) # shape (21, 3)
        
        # 1. Translation Invariance: Shift all coordinates relative to wrist (index 0)
        wrist = landmarks[0]
        shifted_landmarks = landmarks - wrist
        
        # 2. Scale Invariance: Divide by hand size scale.
        # We define scale as distance from wrist (0) to middle finger MCP (9)
        scale = np.linalg.norm(shifted_landmarks[9])
        if scale < 1e-6:
            scale = 1.0
        normalized_landmarks = shifted_landmarks / scale
        
        # Flatten normalized landmarks coordinates -> 63 features
        coord_features = normalized_landmarks.flatten()
        
        # 3. Handcrafted Feature: Fingertip-to-Wrist normalized distances -> 5 features
        # Fingertips indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
        fingertip_indices = [4, 8, 12, 16, 20]
        wrist_dist = [np.linalg.norm(normalized_landmarks[idx]) for idx in fingertip_indices]
        
        # 4. Handcrafted Feature: Pairwise Fingertip distances -> 10 features
        fingertip_pairs = []
        for i in range(len(fingertip_indices)):
            for j in range(i + 1, len(fingertip_indices)):
                idx1 = fingertip_indices[i]
                idx2 = fingertip_indices[j]
                dist = np.linalg.norm(normalized_landmarks[idx1] - normalized_landmarks[idx2])
                fingertip_pairs.append(dist)
                
        # 5. Handcrafted Feature: Finger joint angles (flexion) -> 15 features (3 angles per finger)
        # Define joints indices for the 5 fingers (from base to tip)
        finger_chains = [
            [0, 1, 2, 3, 4],     # Thumb
            [0, 5, 6, 7, 8],     # Index
            [0, 9, 10, 11, 12],  # Middle
            [0, 13, 14, 15, 16], # Ring
            [0, 17, 18, 19, 20]  # Pinky
        ]
        
        joint_angles = []
        for chain in finger_chains:
            # Compute vectors between successive points in the chain
            v0 = normalized_landmarks[chain[1]] - normalized_landmarks[chain[0]]
            v1 = normalized_landmarks[chain[2]] - normalized_landmarks[chain[1]]
            v2 = normalized_landmarks[chain[3]] - normalized_landmarks[chain[2]]
            v3 = normalized_landmarks[chain[4]] - normalized_landmarks[chain[3]]
            
            vectors = [v0, v1, v2, v3]
            
            # Calculate angle between consecutive vectors
            for k in range(len(vectors) - 1):
                u = vectors[k]
                v = vectors[k+1]
                
                # Cosine angle: (u . v) / (|u| * |v|)
                cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-6)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.arccos(cos_theta) # in radians
                joint_angles.append(angle)
                
        # Concatenate all features
        feature_vector = np.concatenate([
            coord_features,       # 63 features
            wrist_dist,           # 5 features
            fingertip_pairs,      # 10 features
            joint_angles          # 15 features
        ])
        
        return feature_vector

    def close(self):
        self.hands.close()
