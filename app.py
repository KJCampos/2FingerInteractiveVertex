# app.py (CAD HUD only — no Taichi)
import time
import importlib
import cv2

from hud_cad import HUDCAD
from predictor import AlphaBetaPredictor
from voice_cmd import VoiceCommands

WINDOW_NAME = "2FingerInteractiveVertex - CAD HUD"

# Visual toggles
DRAW_POINTERS = True
DRAW_ALL_LANDMARKS = False

# Predictive smoothing (feels good)
USE_PREDICTION = True
PRED_LEAD_SEC = 1.0 / 60.0
PRED_ALPHA = 0.85
PRED_BETA = 0.02
PRED_ADAPTIVE = True

# Pinch tuning in normalized UV space (index-tip to thumb-tip distance)
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

        print("✅ Using hands.process(frame) function as tracker")
        return _FuncTracker()

    available = sorted([k for k in vars(hands_mod).keys() if not k.startswith("_")])
    raise ImportError(
        "Could not find a hand-tracker class/function in hands.py.\n"
        f"Exports found: {available}\n"
        "Expected either a class with .process(frame) or a module function process(frame)."
    )


def _as_hands_list(hand_result):
    if hand_result is None:
        return []
    if isinstance(hand_result, dict) and isinstance(hand_result.get("hands"), list):
        return hand_result["hands"]
    if isinstance(hand_result, (list, tuple)):
        return list(hand_result)
    return []


def _get_landmarks_px(hand):
    """
    Your hands.py spec says it returns:
      {"hands": [{"landmarks_px": [(x,y)*21]}, ...]}
    But we also accept older shapes for safety.
    """
    if hand is None:
        return None
    if isinstance(hand, dict):
        for k in ("landmarks_px", "landmarks", "lm", "lms"):
            if k in hand:
                return hand[k]
        return None

    # object style
    for attr in ("landmarks_px", "landmarks", "lm", "lms"):
        if hasattr(hand, attr):
            return getattr(hand, attr)

    if isinstance(hand, (list, tuple)) and len(hand) >= 10:
        return hand

    return None


def _tip_uv(lms, idx, w, h):
    x, y = lms[idx][0], lms[idx][1]
    # if normalized
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        uv = (float(x), float(y))
        px = (int(x * w), int(y * h))
        return uv, px
    # assume pixels
    px = (int(x), int(y))
    uv = (float(px[0]) / float(w), float(px[1]) / float(h))
    return uv, px


def _pinch_from_lms(lms, w, h):
    idx_uv, _ = _tip_uv(lms, 8, w, h)   # index tip
    thb_uv, _ = _tip_uv(lms, 4, w, h)   # thumb tip

    dx = idx_uv[0] - thb_uv[0]
    dy = idx_uv[1] - thb_uv[1]
    d = (dx * dx + dy * dy) ** 0.5

    pinch = 1.0 - min(1.0, max(0.0, d / float(PINCH_OPEN_DIST)))
    if d > PINCH_ACTIVE_THRESH:
        pinch *= 0.85
    return float(max(0.0, min(1.0, pinch)))


def main():
    cap = open_camera()
    hand_tracker = _pick_hand_tracker()

    hud = HUDCAD()
    voice = None
    try:
        voice = VoiceCommands()
        voice.start()
        print("✅ Voice commands ON")
    except Exception as e:
        print(f"⚠️ Voice commands OFF ({e})")
        
    pred = [
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
    ]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("❌ Frame grab failed")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        now = time.time()
        dt = max(1e-6, now - prev_time)
        prev_time = now

        fps = 1.0 / dt
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        # ---- hand tracking ----
        res = hand_tracker.process(frame)
        hands = _as_hands_list(res)

        active = [0, 0]
        pinch_val = [0.0, 0.0]
        pos_uv_raw = [(0.0, 0.0), (0.0, 0.0)]

        if len(hands) >= 2:
            for hid in range(2):
                lms = _get_landmarks_px(hands[hid])
                if lms is None:
                    continue

                idx_uv, idx_px = _tip_uv(lms, 8, w, h)
                active[hid] = 1
                pos_uv_raw[hid] = idx_uv
                pinch_val[hid] = _pinch_from_lms(lms, w, h)

                if DRAW_POINTERS:
                    cv2.circle(frame, idx_px, 7, (235, 245, 255), -1, lineType=cv2.LINE_AA)

                if DRAW_ALL_LANDMARKS:
                    for (lx, ly, *rest) in lms:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (160, 220, 255), -1)

        elif len(hands) == 1:
            lms = _get_landmarks_px(hands[0])
            if lms is not None:
                idx_uv, idx_px = _tip_uv(lms, 8, w, h)
                thb_uv, thb_px = _tip_uv(lms, 4, w, h)
                pinch = _pinch_from_lms(lms, w, h)

                active[0] = 1
                active[1] = 1
                pos_uv_raw[0] = idx_uv
                pos_uv_raw[1] = thb_uv
                pinch_val[0] = pinch
                pinch_val[1] = pinch

                if DRAW_POINTERS:
                    cv2.circle(frame, idx_px, 7, (235, 245, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(frame, thb_px, 7, (180, 240, 255), -1, lineType=cv2.LINE_AA)

        # ---- prediction (the “after this feels good” part) ----
        pos_uv_send = [(0.0, 0.0), (0.0, 0.0)]
        vel_uv_send = [(0.0, 0.0), (0.0, 0.0)]
        for hid in range(2):
            if active[hid]:
                if USE_PREDICTION:
                    pred[hid].update(pos_uv_raw[hid], now)
                    pos_uv_send[hid] = pred[hid].predict(now + PRED_LEAD_SEC)
                    vel_uv_send[hid] = pred[hid].velocity()
                else:
                    pos_uv_send[hid] = pos_uv_raw[hid]
                    vel_uv_send[hid] = (0.0, 0.0)
            else:
                pred[hid].reset()

        # HUD update + render (dt supported now)
        hud.update(pos_uv_send, vel_uv_send, pinch_val, active, w, h, dt=dt)
        hud.render(frame)

        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (235, 245, 255),
            2,
            cv2.LINE_AA,
        )
        if voice:
            for phrase in voice.poll():
                hud.apply_voice(phrase)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        hud.handle_key(key)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
