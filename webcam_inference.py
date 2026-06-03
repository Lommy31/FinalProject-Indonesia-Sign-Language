import cv2
import joblib
import numpy as np
import os
from collections import deque, Counter
from datetime import datetime
from src.config import MODEL_PATH, SCALER_PATH, CLASSES
from src.feature_extractor import HandFeatureExtractor

# ─── Stability Settings ───────────────────────────────────────────────────────
BUFFER_SIZE        = 20    # Number of recent frames to consider for smoothing
CONFIRM_THRESHOLD  = 0.70  # Fraction of buffer frames that must agree (70%)
MIN_CONFIDENCE     = 0.45  # Minimum SVM probability to accept prediction
MAX_HANDS          = 2     # How many hands to track simultaneously

# Colors (BGR)
COLOR_LOCKED  = (0, 230, 100)   # Green when locked
COLOR_PENDING = (0, 200, 255)   # Amber when accumulating
COLOR_LEFT    = (255, 140, 0)   # Blue-orange for left hand label
COLOR_RIGHT   = (0, 140, 255)   # Red-orange for right hand label
# ─────────────────────────────────────────────────────────────────────────────


class HandState:
    """Tracks prediction state and smoothing buffer for a single hand."""
    def __init__(self, label: str):
        self.label        = label          # "Left" / "Right"
        self.buffer       = deque(maxlen=BUFFER_SIZE)
        self.locked_letter = "—"
        self.locked_conf   = 0.0
        self.is_locked     = False
        self.prev_locked   = "—"
        self.active        = False         # Was this hand seen this frame?

    def push(self, letter_or_none, conf=0.0):
        self.buffer.append(letter_or_none)
        valid = [p for p in self.buffer if p is not None]
        if len(valid) >= int(BUFFER_SIZE * CONFIRM_THRESHOLD):
            most_common, count = Counter(valid).most_common(1)[0]
            if (count / BUFFER_SIZE) >= CONFIRM_THRESHOLD:
                self.locked_letter = most_common
                self.locked_conf   = conf
                self.is_locked     = True
                return
        self.is_locked = False

    def log_if_changed(self):
        if self.is_locked and self.locked_letter != self.prev_locked:
            now = datetime.now().strftime("%H:%M:%S")
            bar = "█" * int(self.locked_conf * 20)
            print(f"  {now:<12} [{self.label:>5}]  {self.locked_letter:<4}  "
                  f"{self.locked_conf*100:5.1f}%  {bar}")
            self.prev_locked = self.locked_letter


def draw_hand_card(frame, hand_state: HandState, x1, y1, x2, y2):
    """Draw bounding box + prediction card for one hand."""
    box_color  = COLOR_LOCKED if hand_state.is_locked else COLOR_PENDING
    hand_color = COLOR_LEFT   if hand_state.label == "Left" else COLOR_RIGHT

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Semi-transparent label card above the box
    card_h = 90
    card_y1 = max(0, y1 - card_h - 5)
    card_y2 = max(card_h, y1 - 5)
    card_x1 = x1
    card_x2 = min(frame.shape[1], x1 + 130)

    overlay = frame.copy()
    cv2.rectangle(overlay, (card_x1, card_y1), (card_x2, card_y2), (20, 20, 20), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    # Handedness label (Left / Right)
    cv2.putText(frame, f"{hand_state.label} hand",
                (card_x1 + 5, card_y1 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, hand_color, 1, cv2.LINE_AA)

    # Status badge
    status = "LOCKED" if hand_state.is_locked else "..."
    cv2.putText(frame, status,
                (card_x1 + 5, card_y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                COLOR_LOCKED if hand_state.is_locked else COLOR_PENDING,
                1, cv2.LINE_AA)

    # Giant predicted letter
    cv2.putText(frame, hand_state.locked_letter,
                (card_x1 + 10, card_y1 + 82),
                cv2.FONT_HERSHEY_SIMPLEX, 2.8,
                COLOR_LOCKED if hand_state.is_locked else (160, 160, 160),
                4, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model/scaler not found. Run:\n  /opt/anaconda3/bin/python3 -m src.train")
        return

    print("Loading SVM model and scaler...")
    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    print(f"Initializing MediaPipe Hands (max {MAX_HANDS} hands)...")
    extractor = HandFeatureExtractor(static_image_mode=False, max_num_hands=MAX_HANDS)

    print("Opening Webcam... Press 'q' to quit.\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        print("On macOS: System Settings → Privacy & Security → Camera → allow Terminal/IDE.")
        return

    # ── Per-hand tracking state ──────────────────────────────────────────────
    hand_states = {
        "Left":  HandState("Left"),
        "Right": HandState("Right"),
    }

    # ── Terminal header ──────────────────────────────────────────────────────
    print("─" * 60)
    print("  Bisindo Sign Recognition - Terminal Log (2-Hand Mode)")
    print("─" * 60)
    print(f"  {'Time':<12} {'Hand':<9} {'Sign':<6} {'Confidence'}")
    print("─" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # ── Detect all hands ─────────────────────────────────────────────────
        mp_results, hands_data = extractor.extract_all_hands_landmarks(frame)

        # Mark all hands as inactive this frame
        for hs in hand_states.values():
            hs.active = False

        # Draw all landmarks
        if mp_results.multi_hand_landmarks:
            for hand_lm in mp_results.multi_hand_landmarks:
                extractor.mp_draw.draw_landmarks(
                    frame, hand_lm, extractor.mp_hands.HAND_CONNECTIONS)

        # ── Process each detected hand ────────────────────────────────────────
        for hand in hands_data:
            side      = hand["handedness"]     # "Left" or "Right"
            landmarks = hand["landmarks"]
            hs        = hand_states[side]
            hs.active = True

            features = extractor.extract_features(landmarks)
            if features is not None:
                feat_scaled = scaler.transform(features.reshape(1, -1))
                pred_idx    = int(model.predict(feat_scaled)[0])
                probs       = model.predict_proba(feat_scaled)[0]
                conf        = float(probs[pred_idx])
                letter      = CLASSES[pred_idx]

                if conf >= MIN_CONFIDENCE:
                    hs.push(letter, conf)
                else:
                    hs.push(None)
            else:
                hs.push(None)

            # Log terminal if letter changed
            hs.log_if_changed()

            # Draw card
            x1, y1, x2, y2 = extractor.get_hand_bbox(landmarks, h, w)
            draw_hand_card(frame, hs, x1, y1, x2, y2)

        # Push None for hands not seen this frame (fade out)
        for side, hs in hand_states.items():
            if not hs.active:
                hs.push(None)

        # ── HUD (top-left) ───────────────────────────────────────────────────
        hud_overlay = frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (360, 95), (20, 20, 20), -1)
        frame = cv2.addWeighted(hud_overlay, 0.65, frame, 0.35, 0)

        cv2.putText(frame, "Bisindo Alphabet Recognition",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Tracking up to {MAX_HANDS} hands simultaneously",
                    (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "Hold sign steady until LOCKED  |  Press 'q' to quit",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 200, 255), 1, cv2.LINE_AA)

        # Active hand count badge
        n_active = sum(1 for hd in hands_data if hd)
        cv2.putText(frame, f"Hands detected: {n_active}/{MAX_HANDS}",
                    (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

        cv2.imshow("Bisindo Sign Language Recognition (2-Hand)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    extractor.close()
    print("\n" + "─" * 60)
    print("  Webcam inference stopped.")
    print("─" * 60)


if __name__ == "__main__":
    main()
