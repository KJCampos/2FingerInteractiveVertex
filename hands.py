"""
Hand tracking wrapper using MediaPipe.

We output:
- palm_px: pixel coords of a "palm center" estimate
- pinch: 0..1 (1 = thumb and index are close)
- landmarks_px: list of 21 pixel points for debugging
"""

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]

        # MediaPipe expects RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(frame_rgb)

        if not result.multi_hand_landmarks:
            return None

        hand_lms = result.multi_hand_landmarks[0]

        # Convert landmarks to pixel coords
        pts = []
        for lm in hand_lms.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            pts.append((x, y))

        # Landmark indices:
        # 0 = wrist
        # 4 = thumb tip
        # 8 = index tip
        wrist = pts[0]
        thumb_tip = pts[4]
        index_tip = pts[8]

        # Palm center estimate: average of wrist + a few knuckles
        # (works well enough for MVP)
        knuckle_ids = [0, 5, 9, 13, 17]  # wrist + finger MCP joints
        palm_x = int(sum(pts[i][0] for i in knuckle_ids) / len(knuckle_ids))
        palm_y = int(sum(pts[i][1] for i in knuckle_ids) / len(knuckle_ids))
        palm_px = (palm_x, palm_y)

        # Pinch strength from thumb-index distance (normalize by image size)
        dist = np.hypot(thumb_tip[0] - index_tip[0], thumb_tip[1] - index_tip[1])
        dist_norm = dist / max(w, h)

        close_dist = 0.02  # empirically chosen
        far_dist = 0.10

        pinch = 1.0 - np.clip((dist_norm - close_dist) / (far_dist - close_dist), 0.0, 1.0)

        return {
            "palm_px": palm_px,
            "pinch": float(pinch),
            "landmarks_px": pts
        }
