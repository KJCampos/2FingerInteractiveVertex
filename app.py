import os
import sys
import time
import importlib
import cv2
import requests

from sim_taichi import ParticleSimTaichi
from params import Params

STREAM_TO_SERVER = True
SERVER_HAND_URL = "http://127.0.0.1:8765/hand"
WINDOW_NAME = "Hand Fluid Particles (Two Fingertips)"

# Visual toggles
DRAW_POINTERS = True      # show fingertip dots + midpoint dot
DRAW_ALL_LANDMARKS = False  # keep this False to avoid noisy dots


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
    """
    Fixes your import error by NOT assuming a class name.

    It will:
      - import hands.py as a module
      - find any class with a .process(frame) method
      - OR wrap a module-level function hands.process(frame)
    """
    hands_mod = importlib.import_module("hands")

    # 1) common names first (if you happen to have them)
    common = ["HandTracker", "Hands", "HandTracking", "Tracker", "HandDetector"]
    for name in common:
        obj = getattr(hands_mod, name, None)
        if isinstance(obj, type):
            inst = obj()
            if callable(getattr(inst, "process", None)):
                return inst

    # 2) any class that has process()
    for name, obj in vars(hands_mod).items():
        if isinstance(obj, type):
            try:
                inst = obj()
            except Exception:
                continue
            if callable(getattr(inst, "process", None)):
                print(f"✅ Using hands.{name}() as tracker")
                return inst

    # 3) module-level process(frame) function
    if callable(getattr(hands_mod, "process", None)):

        class _FuncTracker:
            def process(self, frame):
                return hands_mod.process(frame)

        print("✅ Using hands.process(frame) function as tracker")
        return _FuncTracker()

    # If we got here: show what *is* in hands.py
    available = sorted([k for k in vars(hands_mod).keys() if not k.startswith("_")])
    raise ImportError(
        "Could not find a hand-tracker class/function in hands.py.\n"
        f"Exports found: {available}\n"
        "Expected either a class with .process(frame) or a module function process(frame)."
    )


def _as_hands_list(hand_result):
    """
    Normalizes whatever your tracker returns into a list of hand dicts.

    Accepts:
      - None
      - {"landmarks_px":[...], ...}  (single hand)
      - {"hands":[{...},{...}]}      (multi-hand)
    """
    if hand_result is None:
        return []
    if isinstance(hand_result, dict) and isinstance(hand_result.get("hands"), list):
        return hand_result["hands"]
    if isinstance(hand_result, dict):
        return [hand_result]
    return []


def _get_landmarks_px(hand_dict):
    # Try a few likely keys
    for k in ("landmarks_px", "lm_px", "landmarks"):
        v = hand_dict.get(k)
        if isinstance(v, list) and len(v) >= 9:
            return v
    return None


def _tip_uv(lms_px, idx, w, h):
    x, y = lms_px[idx]
    return (x / w, y / h), (int(x), int(y))


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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    session = requests.Session()

    prev_time = time.time()
    fps_smooth = 0.0

    # per-pointer velocity tracking (2 pointers)
    prev_uv = [None, None]
    prev_t = [None, None]

    # pinch tuning (in normalized UV space)
    PINCH_OPEN_DIST = 0.12  # bigger => easier to reach pinch=1

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

        # Default: deactivate both pointers
        active = [0, 0]
        pinch_val = [0.0, 0.0]
        pos_uv = [(0.0, 0.0), (0.0, 0.0)]
        vel_uv = [(0.0, 0.0), (0.0, 0.0)]

        # ---- Pointer selection logic ----
        # If 2 hands: use each hand's INDEX_TIP (8) as pointer0/pointer1
        # If 1 hand: use INDEX_TIP (8) as pointer0 and THUMB_TIP (4) as pointer1 (only active if pinching)
        if len(hands) >= 2:
            for hid in range(2):
                lms = _get_landmarks_px(hands[hid])
                if lms is None:
                    continue

                idx_uv, idx_px = _tip_uv(lms, 8, w, h)
                thb_uv, thb_px = _tip_uv(lms, 4, w, h)

                dx = idx_uv[0] - thb_uv[0]
                dy = idx_uv[1] - thb_uv[1]
                dist = (dx * dx + dy * dy) ** 0.5
                pinch = 1.0 - min(dist / PINCH_OPEN_DIST, 1.0)

                pos_uv[hid] = idx_uv
                pinch_val[hid] = float(pinch)
                active[hid] = 1

                # velocity
                if prev_uv[hid] is not None and prev_t[hid] is not None:
                    dtv = max(1e-4, now - prev_t[hid])
                    vel_uv[hid] = ((idx_uv[0] - prev_uv[hid][0]) / dtv,
                                   (idx_uv[1] - prev_uv[hid][1]) / dtv)
                prev_uv[hid] = idx_uv
                prev_t[hid] = now

                if DRAW_POINTERS:
                    cv2.circle(frame, idx_px, 7, (255, 255, 0), -1)  # index tip
                    cv2.circle(frame, thb_px, 7, (255, 0, 255), -1)  # thumb tip

        elif len(hands) == 1:
            lms = _get_landmarks_px(hands[0])
            if lms is not None:
                idx_uv, idx_px = _tip_uv(lms, 8, w, h)
                thb_uv, thb_px = _tip_uv(lms, 4, w, h)

                # pinch amount from tip distance
                dx = idx_uv[0] - thb_uv[0]
                dy = idx_uv[1] - thb_uv[1]
                dist = (dx * dx + dy * dy) ** 0.5
                pinch = 1.0 - min(dist / PINCH_OPEN_DIST, 1.0)

                # midpoint dot = the "grab point" (blue dot replacement)
                mid_uv = ((idx_uv[0] + thb_uv[0]) * 0.5, (idx_uv[1] + thb_uv[1]) * 0.5)
                mid_px = (int(mid_uv[0] * w), int(mid_uv[1] * h))

                # pointer0 = midpoint (feels like pinch-grab)
                pos_uv[0] = mid_uv
                pinch_val[0] = float(pinch)
                active[0] = 1

                # pointer1 = thumb tip (only active while actually pinching)
                if pinch > 0.15:
                    pos_uv[1] = thb_uv
                    pinch_val[1] = float(pinch)
                    active[1] = 1
                else:
                    pos_uv[1] = (0.0, 0.0)
                    pinch_val[1] = 0.0
                    active[1] = 0

                # velocity per pointer id
                for hid in (0, 1):
                    if active[hid] == 0:
                        prev_uv[hid] = None
                        prev_t[hid] = None
                        vel_uv[hid] = (0.0, 0.0)
                        continue

                    if prev_uv[hid] is not None and prev_t[hid] is not None:
                        dtv = max(1e-4, now - prev_t[hid])
                        vel_uv[hid] = ((pos_uv[hid][0] - prev_uv[hid][0]) / dtv,
                                       (pos_uv[hid][1] - prev_uv[hid][1]) / dtv)

                    prev_uv[hid] = pos_uv[hid]
                    prev_t[hid] = now

                if DRAW_POINTERS:
                    cv2.circle(frame, idx_px, 7, (255, 255, 0), -1)   # index tip
                    cv2.circle(frame, thb_px, 7, (255, 0, 255), -1)   # thumb tip
                    cv2.circle(frame, mid_px, 9, (255, 0, 0), -1)     # NEW blue dot at pinch midpoint

                if DRAW_ALL_LANDMARKS:
                    for (lx, ly) in lms:
                        cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 0), -1)

        else:
            # no hands
            prev_uv = [None, None]
            prev_t = [None, None]

        # ---- Send pointers to sim (2 hands / 2 pointers) ----
        for hid in range(2):
            sim.set_hand_input(
                pos_uv[hid],
                vel_uv[hid],
                pinch_val[hid],                   # 3rd arg is pinch now
                hand_present=bool(active[hid]),
                hand_id=hid
            )

        sim.step(dt)
        sim.render_on_frame(frame)

        # ---- Optional stream to server ----
        if STREAM_TO_SERVER:
            payload = {
                "type": "hand",
                "t": now,
                "pointers": [
                    {"id": 0, "active": bool(active[0]), "pos_uv": [pos_uv[0][0], pos_uv[0][1]],
                     "vel_uv": [vel_uv[0][0], vel_uv[0][1]], "pinch": float(pinch_val[0])},
                    {"id": 1, "active": bool(active[1]), "pos_uv": [pos_uv[1][0], pos_uv[1][1]],
                     "vel_uv": [vel_uv[1][0], vel_uv[1][1]], "pinch": float(pinch_val[1])},
                ],
            }
            try:
                session.post(SERVER_HAND_URL, json=payload, timeout=0.05)
            except Exception:
                pass

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
