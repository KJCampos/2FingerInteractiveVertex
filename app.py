# app.py - FUTURISTIC TONY STARK EDITION
import time
import importlib
import cv2
import os
from collections import deque

from hud_cad import HUDCAD
from predictor import AlphaBetaPredictor

WINDOW_NAME = "Future Hud CAD System"

USE_PREDICTION = True
PRED_LEAD_SEC = 1.0 / 60.0
PRED_ALPHA = 0.85
PRED_BETA = 0.02
PRED_ADAPTIVE = True

PINCH_OPEN_DIST = 0.12
PINCH_ACTIVE_THRESH = 0.10

ENABLE_VOICE = True


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


def _init_voice():
    if not ENABLE_VOICE:
        return None

    try:
        from voice_cmd import VoiceCommands

        model_path = "models/vosk-model-small-en-us-0.15"
        if not os.path.exists(model_path):
            print("⚠️  Voice model not found. Download from:")
            print("   https://alphacephei.com/vosk/models")
            print("   Extract to models/")
            return None

        voice = VoiceCommands(model_path=model_path, wake_word="robin", arm_seconds=2.0)
        voice.start()
        print("✅ Voice commands enabled (wake word: 'robin')")
        return voice
    except ImportError:
        print("⚠️  voice_cmd.py not found - voice disabled")
        return None
    except Exception as e:
        print(f"⚠️  Voice init failed: {e}")
        return None


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


class WaveDetector:
    """Detect a left-right wave with an open hand to clear the overlay."""

    def __init__(self):
        self.history = deque()
        self.cooldown = 1.0
        self.last_trigger = 0.0

    def update(self, lms, w, h, pinch: float, t_now: float) -> bool:
        if lms is None:
            self.history.clear()
            return False

        # Only listen when not pinching (open hand)
        if pinch > 0.35:
            self.history.clear()
            return False

        def _uv(idx):
            uv, _ = _tip_uv(lms, idx, w, h)
            return uv

        tips = [_uv(i) for i in (4, 8, 12, 16, 20)]
        xs = [u for u, _ in tips]
        ys = [v for _, v in tips]

        # Require fingers to be reasonably spread (open hand)
        spread_x = max(xs) - min(xs)
        spread_y = max(ys) - min(ys)
        if spread_x < 0.12 or spread_y < 0.08:
            self.history.clear()
            return False

        # Palm center approximation using fingertips
        cx = sum(xs) / len(xs)
        self.history.append((t_now, cx))

        # Keep ~1.2s of history
        while self.history and (t_now - self.history[0][0] > 1.2):
            self.history.popleft()

        if len(self.history) < 5 or (t_now - self.last_trigger) < self.cooldown:
            return False

        xs_hist = [p[1] for p in self.history]
        range_x = max(xs_hist) - min(xs_hist)
        if range_x < 0.20:
            return False

        # Count direction changes with meaningful motion
        changes = 0
        last_sign = 0
        for i in range(1, len(xs_hist)):
            dx = xs_hist[i] - xs_hist[i - 1]
            if abs(dx) < 0.01:
                continue
            sign = 1 if dx > 0 else -1
            if last_sign != 0 and sign != last_sign:
                changes += 1
            last_sign = sign

        if changes >= 2:
            self.last_trigger = t_now
            self.history.clear()
            return True

        return False


def main():
    cap = open_camera()
    tracker = _pick_hand_tracker()

    hud = HUDCAD()

    voice = _init_voice()

    pred = [
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
    ]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, hud.on_mouse)

    if hud.projection_mode:
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev = time.time()
    fps_smooth = 0.0

    print("\n" + "="*60)
    print("🚀 STARK TABLE HOLOGRAPHIC CAD SYSTEM")
    print("="*60)
    print("\n📋 CONTROLS:")
    print("   T - Toggle Line/Curve tool")
    print("   +/- or mouse wheel - Zoom viewport, 0 to reset")
    print("   R - Reset rotation (selected object)")
    print("   C - Clear all 3D objects | X - Clear strokes")
    print("   DELETE - Remove selected object")
    print("   Drag mouse or pinch index to rotate selection")
    print("   ESC - Exit")

    if voice:
        print("\n🎤 VOICE COMMANDS:")
        print("   Say 'robin' then:")
        print("   - 'clear' (strokes)")
        print("   - 'clear all' (all meshes)")
        print("   - 'delete' (selected)")
        print("   - 'reset' (camera/rotation)")
        print("   - 'line' / 'curve'")

    print("\n✨ GESTURE MENU:")
    print("   Pinch near ring quadrants:")
    print("   TOP: Line tool")
    print("   RIGHT: Curve tool")
    print("   BOTTOM: Clear")
    print("   LEFT: Cycle render mode")
    print("\n" + "="*60 + "\n")

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

        active = [0, 0]
        pinch = [0.0, 0.0]
        pos_uv = [(0.5, 0.5), (0.5, 0.5)]
        vel_uv = [(0.0, 0.0), (0.0, 0.0)]

        if len(hands) >= 2:
            for hid in range(2):
                lms = _get_landmarks_px(hands[hid])
                if lms is None:
                    continue
                uv, _ = _tip_uv(lms, 8, W, H)
                active[hid] = 1
                pinch[hid] = _pinch_from_lms(lms, W, H)
                pos_uv[hid] = uv
            primary_lms = _get_landmarks_px(hands[0])
        elif len(hands) == 1:
            lms = _get_landmarks_px(hands[0])
            if lms is not None:
                primary_lms = lms
                uv0, _ = _tip_uv(lms, 8, W, H)
                active = [1, 0]
                pinch0 = _pinch_from_lms(lms, W, H)
                pinch = [pinch0, pinch0]
                pos_uv = [uv0, uv1]

        pos_uv_send = []
        vel_uv_send = []
        t_now = now
        for i in range(2):
            if active[i]:
                pred[i].update(pos_uv[i], t_now)
                p = pred[i].predict(t_now + PRED_LEAD_SEC) if USE_PREDICTION else pos_uv[i]
                v = pred[i].velocity()
                pos_uv_send.append(p)
                vel_uv_send.append(v)
            else:
                pred[i].reset()
                pos_uv_send.append(pos_uv[i])
                vel_uv_send.append((0.0, 0.0))

        hud.update(pos_uv_send, vel_uv_send, pinch, active, W, H, dt=dt)

        composed = hud.render(frame)

        fps_text = f"FPS: {fps_smooth:5.1f}"
        cv2.putText(composed, fps_text, (12, composed.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 245, 0), 3, cv2.LINE_AA)
        cv2.putText(composed, fps_text, (12, composed.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 128), 2, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, composed)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        if key != 255:
            hud.handle_key(key)

        if voice:
            cmd = voice.pop_command()
            if cmd:
                print(f"🎤 Voice command: {cmd}")
                hud.apply_voice(cmd)

    cap.release()
    cv2.destroyAllWindows()

    if voice:
        voice.stop()

    print("\n✅ STARK TABLE system shutdown complete")


if __name__ == "__main__":
    main()
