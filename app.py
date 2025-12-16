# app.py
import os
import sys
import time
import importlib
import cv2
import requests

from sim_taichi import ParticleSimTaichi
from params import Params
from predictor import AlphaBetaPredictor
from hud_lines import HUDLines

STREAM_TO_SERVER = True
SERVER_HAND_URL = "http://127.0.0.1:8765/hand"
WINDOW_NAME = "Hand Fluid Particles (Two Fingertips + Predict)"

# Visual toggles
DRAW_POINTERS = True
DRAW_ALL_LANDMARKS = False

# Predictive smoothing (feels way better)
USE_PREDICTION = True
PRED_LEAD_SEC = 1.0 / 60.0  # predict ~1 frame ahead
PRED_ALPHA = 0.85
PRED_BETA = 0.02
PRED_ADAPTIVE = True

# Pinch tuning in normalized UV space (index-tip to thumb-tip distance)
PINCH_OPEN_DIST = 0.12  # bigger => easier to reach pinch=1
PINCH_ACTIVE_THRESH = 0.10  # below this, you can treat it as "not pinching"


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


def init_taichi():
    import taichi as ti
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32)
        print("✅ Taichi using CUDA")
    except Exception as e:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        print(f"ℹ️ Taichi using CPU (CUDA init failed: {e})")
    return ti


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
    if isinstance(hand_result, dict):
        return [hand_result]
    return []


def _get_landmarks_px(hand_dict):
    for k in ("landmarks_px", "lm_px", "landmarks"):
        v = hand_dict.get(k)
        if isinstance(v, list) and len(v) >= 9:
            return v
    return None


def _tip_uv(lms_px, idx, w, h):
    x, y = lms_px[idx]
    return (x / w, y / h), (int(x), int(y))


def _pinch_from_lms(lms_px, w, h):
    idx_uv, _ = _tip_uv(lms_px, 8, w, h)
    thb_uv, _ = _tip_uv(lms_px, 4, w, h)
    dx = idx_uv[0] - thb_uv[0]
    dy = idx_uv[1] - thb_uv[1]
    dist = (dx * dx + dy * dy) ** 0.5
    pinch = 1.0 - min(dist / PINCH_OPEN_DIST, 1.0)
    return float(pinch)


def main():
    init_taichi()
    print("RUNNING app.py", "PID:", os.getpid(), "PY:", sys.executable)

    params = Params()

    cap = open_camera()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    tracker = _pick_hand_tracker()
    sim = ParticleSimTaichi(params=params)

    # Tony HUD interface layer (2-finger line tool)
    hud = HUDLines(
        grid_w=getattr(params, "grid_w", 240),
        grid_h=getattr(params, "grid_h", 135),
    )

    pred = [
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
        AlphaBetaPredictor(alpha=PRED_ALPHA, beta=PRED_BETA, adaptive=PRED_ADAPTIVE),
    ]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    session = requests.Session()

    prev_time = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        now = time.time()
        dt = max(1e-4, min(now - prev_time, 1 / 20))
        prev_time = now

        fps = 1.0 / dt
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps if fps_smooth > 0 else fps

        hand_result = tracker.process(frame)
        hands = _as_hands_list(hand_result)

        active = [0, 0]
        pinch_val = [0.0, 0.0]
        pos_uv_raw = [(0.0, 0.0), (0.0, 0.0)]

        # Pointer selection
        if len(hands) >= 2:
            for hid in range(2):
                lms = _get_landmarks_px(hands[hid])
                if lms is None:
                    continue
                idx_uv, idx_px = _tip_uv(lms, 8, w, h)
                thb_uv, thb_px = _tip_uv(lms, 4, w, h)

                active[hid] = 1
                pos_uv_raw[hid] = idx_uv
                pinch_val[hid] = _pinch_from_lms(lms, w, h)

                if DRAW_POINTERS:
                    cv2.circle(frame, idx_px, 7, (255, 255, 0), -1)
                    cv2.circle(frame, thb_px, 7, (255, 0, 255), -1)

                if DRAW_ALL_LANDMARKS:
                    for (lx, ly) in lms:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -1)

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
                    cv2.circle(frame, idx_px, 8, (255, 255, 0), -1)
                    cv2.circle(frame, thb_px, 8, (255, 0, 255), -1)

                if DRAW_ALL_LANDMARKS:
                    for (lx, ly) in lms:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        # Predict + send to sim
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
                pos_uv_send[hid] = (0.0, 0.0)
                vel_uv_send[hid] = (0.0, 0.0)

            sim.set_hand_input(
                pos_uv_send[hid],
                vel_uv_send[hid],
                pinch_val[hid],
                hand_present=bool(active[hid]),
                hand_id=hid,
            )

        if DRAW_POINTERS and USE_PREDICTION:
            for hid in range(2):
                if active[hid]:
                    px = int(pos_uv_send[hid][0] * w)
                    py = int(pos_uv_send[hid][1] * h)
                    cv2.circle(frame, (px, py), 6, (255, 0, 0), -1)

        # ---- step + render sim ----
        sim.step(dt)
        sim.render_on_frame(frame)

        # ---- HUD interface update + render ----
        hud.update(pos_uv_send, pinch_val, active, w, h)
        hud.render(frame)

        # ---- Optional stream ----
        if STREAM_TO_SERVER:
            payload = {
                "type": "hand",
                "t": now,
                "pointers": [
                    {
                        "id": 0,
                        "active": bool(active[0]),
                        "pos_uv": [float(pos_uv_send[0][0]), float(pos_uv_send[0][1])],
                        "vel_uv": [float(vel_uv_send[0][0]), float(vel_uv_send[0][1])],
                        "pinch": float(pinch_val[0]),
                    },
                    {
                        "id": 1,
                        "active": bool(active[1]),
                        "pos_uv": [float(pos_uv_send[1][0]), float(pos_uv_send[1][1])],
                        "vel_uv": [float(vel_uv_send[1][0]), float(vel_uv_send[1][1])],
                        "pinch": float(pinch_val[1]),
                    },
                ],
            }
            try:
                session.post(SERVER_HAND_URL, json=payload, timeout=0.05)
            except Exception:
                pass

        cv2.putText(
            frame,
            f"FPS: {fps_smooth:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        # forward keys to BOTH systems
        hud.handle_key(key)
        sim.handle_key(key)

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
