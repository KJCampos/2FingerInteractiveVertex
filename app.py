import os
import sys
import time
import cv2
import requests


from hands import HandTracker
from sim_taichi import ParticleSimTaichi
from params import Params
from matrix import Matrix, Vector


STREAM_TO_SERVER = True
SERVER_HAND_URL = "http://127.0.0.1:8765/hand"
WINDOW_NAME = "Hand Fluid Particles (MVP)"


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


def main():
    import taichi as ti
    ti.init(arch=ti.cpu, default_fp=ti.f32)
    print("RUNNING app.py", "PID:", os.getpid(), "PY:", sys.executable)

    params = Params()
    cap = open_camera()
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    hand_tracker = HandTracker()
    sim = ParticleSimTaichi(params=params)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    session = requests.Session()

    prev_time = time.time()
    prev_palm_uv = None
    prev_palm_time = None
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

        hand = hand_tracker.process(frame)

        hand_present = hand is not None
        pinch = 0.0
        palm_uv = (0.0, 0.0)
        vel_uv = (0.0, 0.0)

        if hand_present:
            palm_px = hand["palm_px"]
            pinch = float(hand["pinch"])
            palm_uv = (palm_px[0] / w, palm_px[1] / h)

            if prev_palm_uv is not None and prev_palm_time is not None:
                dtv = max(1e-4, now - prev_palm_time)
                vel_uv = ((palm_uv[0] - prev_palm_uv[0]) / dtv,
                          (palm_uv[1] - prev_palm_uv[1]) / dtv)

            prev_palm_uv = palm_uv
            prev_palm_time = now

            sim.set_hand_input(palm_uv, vel_uv, pinch, hand_present=True)

            for (lx, ly) in hand["landmarks_px"]:
                cv2.circle(frame, (lx, ly), 2, (0, 255, 0), -1)
            cv2.circle(frame, palm_px, 8, (255, 255, 0), -1)
        else:
            prev_palm_uv = None
            prev_palm_time = None
            sim.set_hand_input((0.0, 0.0), (0.0, 0.0), 0.0, hand_present=False)

        sim.step(dt)
        sim.render_on_frame(frame)

        if STREAM_TO_SERVER:
            payload = {
                "type": "hand",
                "t": now,
                "present": bool(hand_present),
                "palm_uv": [float(palm_uv[0]), float(palm_uv[1])],
                "vel_uv": [float(vel_uv[0]), float(vel_uv[1])],
                "pinch": float(pinch),
            }
            try:
                session.post(SERVER_HAND_URL, json=payload, timeout=0.05)
            except Exception:
                pass

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (20,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # MUST be 1 (or small). 0 = freeze-until-keypress.
        key = cv2.waitKey(1) & 0xFF

        # close with X
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
