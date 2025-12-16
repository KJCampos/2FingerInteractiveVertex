# hud_lines.py
import math
import cv2


def _clamp(x, a, b):
    return a if x < a else b if x > b else x


class HUDLines:
    """
    Two-pinched-fingers interface tool:
      - Pinch both pointers -> preview line from finger0 to finger1
      - Live distance label (px + grid units)
      - Default axis-lock (H/V). Toggle with L.
      - Release either pinch -> commit segment
      - Magnet snapping to existing vertices
      - Build boxes by chaining segments; endpoints snap together cleanly
    """

    def __init__(self, grid_w=240, grid_h=135):
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)

        # ---- toggles ----
        self.enabled = True
        self.axis_lock = True          # default ON
        self.curve_mode = False        # reserved (not implemented)
        self.grid_snap = False         # optional
        self.show_units = True
        self.show_preview = True

        # ---- thresholds ----
        self.pinch_on = 0.78
        self.pinch_off = 0.62

        # ---- snapping ----
        self.snap_px = 18              # magnet radius in pixels
        self.grid_step = 1             # grid snap step (cells)

        # ---- state ----
        self._placing = False
        self._a_uv = (0.0, 0.0)
        self._b_uv = (0.0, 0.0)
        self._axis = 0  # 0 none, 1 horizontal, 2 vertical

        # geometry storage
        self.verts_uv = []     # [(u,v), ...]
        self.segs = []         # [(i0,i1), ...]

        # subtle “glow” palette
        self.col_core = (245, 255, 255)   # BGR
        self.col_line = (230, 245, 255)
        self.col_shadow = (10, 10, 10)

    # ---------- controls ----------
    def handle_key(self, key: int):
        if key is None:
            return

        if key in (ord("i"), ord("I")):
            self.enabled = not self.enabled
            self._placing = False
            return

        if key in (ord("l"), ord("L")):
            self.axis_lock = not self.axis_lock
            self._axis = 0
            return

        if key in (ord("k"), ord("K")):
            self.curve_mode = not self.curve_mode
            self._axis = 0
            return

        if key in (ord("g"), ord("G")):
            self.grid_snap = not self.grid_snap
            return

        if key in (ord("u"), ord("U")):
            self.show_units = not self.show_units
            return

        if key in (ord("p"), ord("P")):
            self.show_preview = not self.show_preview
            return

        if key in (ord("x"), ord("X")):
            self.clear()
            return

        if key == ord("["):
            self.snap_px = max(6, self.snap_px - 2)
            return

        if key == ord("]"):
            self.snap_px = min(60, self.snap_px + 2)
            return

    def clear(self):
        self._placing = False
        self.verts_uv.clear()
        self.segs.clear()

    # ---------- update ----------
    def update(self, pos_uv_send, pinch_val, active, frame_w, frame_h):
        """
        Call once per frame.
        pos_uv_send: [(u,v),(u,v)] from app.py
        pinch_val: [p0,p1] 0..1
        active: [0/1, 0/1]
        """
        if not self.enabled:
            self._placing = False
            return

        a_ok = bool(active[0])
        b_ok = bool(active[1])
        if not (a_ok and b_ok):
            self._placing = False
            return

        p0 = float(pinch_val[0])
        p1 = float(pinch_val[1])
        both_pinching = (p0 >= self.pinch_on) and (p1 >= self.pinch_on)

        # Start placing
        if (not self._placing) and both_pinching:
            self._placing = True
            self._a_uv = tuple(pos_uv_send[0])
            self._b_uv = tuple(pos_uv_send[1])
            self._axis = 0
            self._decide_axis_lock(frame_w, frame_h)

        # Update placing
        if self._placing:
            self._a_uv = tuple(pos_uv_send[0])
            self._b_uv = tuple(pos_uv_send[1])

            if self.axis_lock and (not self.curve_mode):
                self._apply_axis_lock(frame_w, frame_h)

            # Commit on release
            if (p0 <= self.pinch_off) or (p1 <= self.pinch_off):
                self._commit_segment(frame_w, frame_h)
                self._placing = False

    # ---------- render ----------
    def render(self, frame_bgr):
        if not self.enabled or frame_bgr is None:
            return frame_bgr

        h, w = frame_bgr.shape[:2]

        # Committed segments
        for (i0, i1) in self.segs:
            a = self.verts_uv[i0]
            b = self.verts_uv[i1]
            ax, ay = int(a[0] * w), int(a[1] * h)
            bx, by = int(b[0] * w), int(b[1] * h)

            # shadow
            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 0, 0), 4, cv2.LINE_AA)
            # core
            cv2.line(frame_bgr, (ax, ay), (bx, by), self.col_line, 2, cv2.LINE_AA)

            cv2.circle(frame_bgr, (ax, ay), 4, self.col_core, -1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (bx, by), 4, self.col_core, -1, cv2.LINE_AA)

        # Preview segment
        if self._placing and self.show_preview:
            a2 = self._snap_uv(self._a_uv, w, h, commit=False)
            b2 = self._snap_uv(self._b_uv, w, h, commit=False)

            ax, ay = int(a2[0] * w), int(a2[1] * h)
            bx, by = int(b2[0] * w), int(b2[1] * h)

            cv2.line(frame_bgr, (ax, ay), (bx, by), (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(frame_bgr, (ax, ay), (bx, by), self.col_core, 2, cv2.LINE_AA)
            cv2.circle(frame_bgr, (ax, ay), 6, self.col_core, 1, cv2.LINE_AA)
            cv2.circle(frame_bgr, (bx, by), 6, self.col_core, 1, cv2.LINE_AA)

            self._draw_dimension_text(frame_bgr, (ax, ay), (bx, by))

        # HUD text
        cv2.putText(
            frame_bgr,
            f"HUD(I) {'ON' if self.enabled else 'OFF'} | AxisLock(L) {'ON' if self.axis_lock else 'OFF'} | Snap([ ]) {self.snap_px}px | GridSnap(G) {'ON' if self.grid_snap else 'OFF'} | Clear(X)",
            (10, h - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        return frame_bgr

    # ---------- internals ----------
    def _decide_axis_lock(self, w, h):
        if (not self.axis_lock) or self.curve_mode:
            self._axis = 0
            return
        ax, ay = self._a_uv[0] * w, self._a_uv[1] * h
        bx, by = self._b_uv[0] * w, self._b_uv[1] * h
        dx = abs(bx - ax)
        dy = abs(by - ay)
        self._axis = 1 if dx >= dy else 2  # 1 horizontal, 2 vertical

    def _apply_axis_lock(self, w, h):
        if self._axis == 0:
            self._decide_axis_lock(w, h)
        if self._axis == 1:
            self._b_uv = (self._b_uv[0], self._a_uv[1])
        elif self._axis == 2:
            self._b_uv = (self._a_uv[0], self._b_uv[1])

    def _snap_uv(self, uv, w, h, commit: bool):
        u, v = uv
        u = _clamp(u, 0.0, 1.0)
        v = _clamp(v, 0.0, 1.0)

        # grid snap (optional)
        if self.grid_snap:
            # snap to integer grid cells (or step)
            gx = round((u * (self.grid_w - 1)) / self.grid_step) * self.grid_step
            gy = round((v * (self.grid_h - 1)) / self.grid_step) * self.grid_step
            u = gx / max(1, (self.grid_w - 1))
            v = gy / max(1, (self.grid_h - 1))

        if not self.verts_uv:
            return (u, v)

        px = u * w
        py = v * h
        r2 = float(self.snap_px * self.snap_px)

        best_i = -1
        best_d2 = 1e30

        for i, (uu, vv) in enumerate(self.verts_uv):
            qx = uu * w
            qy = vv * h
            dx = px - qx
            dy = py - qy
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_i = i

        if best_i >= 0 and best_d2 <= r2:
            return self.verts_uv[best_i]

        return (u, v)

    def _get_or_add_vertex(self, uv, w, h):
        uv = self._snap_uv(uv, w, h, commit=True)

        # reuse identical vertex
        for i, (uu, vv) in enumerate(self.verts_uv):
            if abs(uu - uv[0]) < 1e-6 and abs(vv - uv[1]) < 1e-6:
                return i

        self.verts_uv.append(uv)
        return len(self.verts_uv) - 1

    def _commit_segment(self, w, h):
        a = self._snap_uv(self._a_uv, w, h, commit=True)
        b = self._snap_uv(self._b_uv, w, h, commit=True)

        ax, ay = a[0] * w, a[1] * h
        bx, by = b[0] * w, b[1] * h

        # ignore tiny segments
        if (ax - bx) ** 2 + (ay - by) ** 2 < 25:
            return

        i0 = self._get_or_add_vertex(a, w, h)
        i1 = self._get_or_add_vertex(b, w, h)

        if i0 == i1:
            return

        # avoid duplicates
        for (s0, s1) in self.segs:
            if (s0 == i0 and s1 == i1) or (s0 == i1 and s1 == i0):
                return

        self.segs.append((i0, i1))

    def _draw_dimension_text(self, frame, A, B):
        ax, ay = A
        bx, by = B
        dx = bx - ax
        dy = by - ay
        dist_px = math.sqrt(dx * dx + dy * dy)

        # grid distance (approx)
        u1 = ax / max(1, frame.shape[1])
        v1 = ay / max(1, frame.shape[0])
        u2 = bx / max(1, frame.shape[1])
        v2 = by / max(1, frame.shape[0])
        dist_grid = math.sqrt(((u2 - u1) * (self.grid_w - 1)) ** 2 + ((v2 - v1) * (self.grid_h - 1)) ** 2)

        # label position: normal offset above segment
        mx = int((ax + bx) * 0.5)
        my = int((ay + by) * 0.5)

        nx, ny = -dy, dx
        nlen = math.sqrt(nx * nx + ny * ny) + 1e-6
        nx, ny = nx / nlen, ny / nlen
        tx = int(mx + nx * 14) - 50
        ty = int(my + ny * 14) - 8

        label = f"{dist_px:.1f}px"
        if self.show_units:
            label += f"  |  {dist_grid:.1f} grid"

        # shadow then bright text
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.col_shadow, 3, cv2.LINE_AA)
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.col_core, 1, cv2.LINE_AA)
