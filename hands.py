import cv2
import numpy as np

import mediapipe as mp

THUMB_TIP = 4
INDEX_TIP = 8


class Hands:
    """
    MediaPipe hands wrapper.

    Returns:
      {"hands": [hand0, hand1, ...]}

    Each hand dict contains:
      - "landmarks_px": [(x,y)*21] pixel coords
    """

    def __init__(self, max_hands=2, det_conf=0.5, track_conf=0.5):
        self.max_hands = max_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=1,
            min_detection_confidence=float(det_conf),
            min_tracking_confidence=float(track_conf),
        )

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Fixes NORM_RECT without IMAGE_DIMENSIONS warning
        self.hands._image_width, self.hands._image_height = frame_bgr.shape[1], frame_bgr.shape[0]  # type: ignore[attr-defined]

        res = self.hands.process(frame_rgb)

        if not res.multi_hand_landmarks:
            return None

        h, w = frame_bgr.shape[:2]
        out = {"hands": []}

        for hand_lms in res.multi_hand_landmarks:
            pts = []
            for lm in hand_lms.landmark:
                pts.append((lm.x * w, lm.y * h))
            out["hands"].append({"landmarks_px": pts})

        return out


# Make a default singleton to match app.py's "hands.process(frame)" fallback
_default = Hands(max_hands=2)


def process(frame):
    return _default.process(frame)
