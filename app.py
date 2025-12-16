# app.py
import time
import importlib
import cv2

from hud_cad import HUDCAD
from predictor import AlphaBetaPredictor

WINDOW_NAME = "2FingerInteractiveVertex (HUD + 2.5D + 3D GPU View)"

# Predictive smoothing
USE_PREDICTION = True
PRED_LEAD_SEC = 1.0 / 60.0
PRED_ALPHA = 0.85
PRED_BETA = 0.02
PRED_ADAPTIVE = True

# Pinch tuning (UV dist)
PINCH_OPEN_DIST = 0.12
PINCH_ACTIVE_THRESH = 0.10


def open_camera(max_index=6):
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"✅ Using camera index: {i}")
                return cap
        cap.release()
    raise RuntimeError(f"❌ No working camera found (0–{max_index-1}).")


def _pick_hand_tracker():
    hands_mod = importlib.import_module("hands")

    common = ["HandTracker", "Hands", "HandTracking", "Tracker", "HandDetector"]
    for name in common:
        obj = getattr(hands_mod, name, None)
        if isinstance(obj, type):
            try:
                inst = obj()
            except Exception:
                continue
            if callable(getattr(inst, "process", None)):
                return inst

    for name, obj in vars(hands_mod).items():
        if isinstance(obj, type):
            try:
                inst = obj()
            except Exception:
                continue
            if callable(getattr(inst, "process", None)):
                print(f"✅ Using hands.{name}() as tracker")
                return inst

    if callable(getattr(hands_mod, "process", None)):

        class _FuncTracker:
            def process(self, frame):
                return hands_mod.process(frame)

        print("✅ Using hands.process(frame)")
        return _FuncTracker()

    raise ImportError("Could not find a tracker in hands.py (needs .process(frame)).")


def _as_hands_list(hand_result):
    if hand_result is None:
        return []
    if isinstance(hand_result, dict) and isinstance(hand_result.get("hands"), list):
        return hand_result["hands"]
    if isinstance(hand_result, (list, tuple)):
        return list(hand_result)
    return []


def _get_landmarks_px(hand):
    if hand is None:
        return None
    if isinstance(hand, dict):
        return hand.get("landmarks_px", hand.get("landmarks", None))
    lms = getattr(hand, "landmarks_px", None)
    if lms is not None:
        return lms
    lms = getattr(hand, "landmarks", None)
    if lms is not None:
        return lms
    return None


def _tip_uv(lms, idx, w, h):
    x, y = lms[idx][0], lms[idx][1]
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        uv = (float(x), float(y))
        px = (int(x * w), int(y * h))
        return uv, px
    px = (int(x), int(y))
    uv = (float(px[0]) / float(w), float(px[1]) / float(h))
    return uv, px


def _pinch_from_lms(lms, w, h):
    idx_uv, _ = _tip_uv(lms, 8, w, h)
    thb_uv, _ = _tip_uv(lms, 4, w, h)
    dx = idx_uv[0] - thb_uv[0]
    dy = idx_uv[1] - thb_uv[1]
    d = (dx * dx + dy * dy) ** 0.5
    pinch = 1.0 - min(1.0, max(0.0, d / float(PINCH_OPEN_DIST)))
    if d > PINCH_ACTIVE_THRESH:
        pinch *= 0.85
    return float(max(0.0, min(1.0, pinch)))


def main():
    cap = open_camera()
    tracker = _pick_hand_tracker()

    hud = HUDCAD()

    pred = [
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
    ]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    #cv2.setMouseCallback(WINDOW_NAME, hud.on_mouse)

    prev = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        now = time.time()
        dt = max(1e-6, now - prev)
        prev = now
        fps = 1.0 / dt
        fps_smooth = fps if fps_smooth == 0 else 0.9 * fps_smooth + 0.1 * fps

        res = tracker.process(frame)
        hands = _as_hands_list(res)

        # default
        active = [0, 0]
        pinch = [0.0, 0.0]
        pos_uv = [(0.5, 0.5), (0.5, 0.5)]
        vel_uv = [(0.0, 0.0), (0.0, 0.0)]

        # pointers mapping:
        # - 2 hands: both index tips
        # - 1 hand: pointer0=index tip, pointer1=thumb tip
        if len(hands) >= 2:
            for hid in range(2):
                lms = _get_landmarks_px(hands[hid])
                if lms is None:
                    continue
                uv, _ = _tip_uv(lms, 8, W, H)
                active[hid] = 1
                pinch[hid] = _pinch_from_lms(lms, W, H)
                pos_uv[hid] = uv
        elif len(hands) == 1:
            lms = _get_landmarks_px(hands[0])
            if lms is not None:
                uv0, _ = _tip_uv(lms, 8, W, H)  # index tip
                uv1, _ = _tip_uv(lms, 4, W, H)  # thumb tip
                active = [1, 1]
                pinch0 = _pinch_from_lms(lms, W, H)
                pinch = [pinch0, pinch0]
                pos_uv = [uv0, uv1]

        # predictive smoothing (optional)
        pos_uv_send = []
        vel_uv_send = []
        t_now = now
        for i in range(2):
            if active[i]:
                pred[i].update(pos_uv[i], t_now)
                if USE_PREDICTION:
                    p = pred[i].predict(t_now + PRED_LEAD_SEC)
                else:
                    p = pos_uv[i]
                v = pred[i].velocity()
                pos_uv_send.append(p)
                vel_uv_send.append(v)
            else:
                pred[i].reset()
                pos_uv_send.append(pos_uv[i])
                vel_uv_send.append((0.0, 0.0))

        # update HUD + geometry + physics
        hud.update(pos_uv_send, vel_uv_send, pinch, active, W, H, dt=dt)

        # render
        composed = hud.render(frame)

        # FPS overlay
        cv2.putText(composed, f"{fps_smooth:5.1f} FPS", (12, composed.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 240, 255), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, composed)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key != 255:
            hud.on_key(key)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
