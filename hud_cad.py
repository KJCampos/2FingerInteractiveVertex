# hud_cad.py
# Two-pointer CAD HUD:
# - Pinch both pointers: preview line between them
# - Release both: commit line (with magnet snap)
# - Close a loop: auto-create a simple extruded 3D wireframe you can grab/rotate/scale
# - Gesture menu: pinch pointer0 near midpoint ring to switch tools (Top/Right/Bottom/Left)

from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np


def _clamp(x, a, b):
    return a if x < a else (b if x > b else x)


def _dist(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.hypot(dx, dy)


def _lerp(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _mul(a, s: float):
    return (a[0] * s, a[1] * s)


def _norm(v):
    n = math.hypot(v[0], v[1]) + 1e-9
    return (v[0] / n, v[1] / n)


def _perp(v):
    return (-v[1], v[0])


@dataclass
class LineSeg:
    a: tuple[int, int]
    b: tuple[int, int]
    curved: bool = False
    ctrl: tuple[int, int] | None = None
    length_px: float = 0.0


@dataclass
class Solid3D:
    # Local 3D vertices (Nx3) + faces via edge list; we render as wireframe.
    verts: np.ndarray          # shape (N*2, 3) for extrude (bottom + top)
    edges: list[tuple[int, int]]
    center: np.ndarray         # shape (3,)
    yaw: float = 0.0
    pitch: float = 0.0
    scale: float = 1.0
    pos: np.ndarray = None     # world translation (3,)

    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class HUDCAD:
    TOOL_LINE = 0
    TOOL_CURVE = 1
    TOOL_GRAB_2D = 2
    TOOL_SOLID_3D = 3

    RENDER_WIRE = 0
    RENDER_NEON = 1

    def __init__(self):
        # --- user feel/tuning ---
        self.pinch_on = 0.75
        self.pinch_off = 0.55

        self.snap_px = 18
        self.axis_lock = True
        self.axis_lock_px = 14

        self.grid_snap = False
        self.grid_px = 18

        self.auto_extrude_on_close = True
        self.extrude_depth = 60.0  # px-like depth

        # --- state ---
        self.tool = self.TOOL_LINE
        self.render_mode = self.RENDER_WIRE

        self.lines: list[LineSeg] = []
        self.vertices: list[tuple[int, int]] = []      # snapped endpoints
        self._chain_active = False
        self._chain: list[int] = []                    # vertex indices of current poly chain

        self.solids: list[Solid3D] = []
        self._active_solid = -1

        # drawing preview
        self._drawing = False
        self._prev_both_pinched = False
        self._prev_pinch0 = 0.0

        self._p0 = (0, 0)
        self._p1 = (0, 0)
        self._v0 = (0.0, 0.0)
        self._v1 = (0.0, 0.0)
        self._vel0 = (0.0, 0.0)
        self._vel1 = (0.0, 0.0)

        # gesture menu (midpoint ring)
        self.menu_enabled = True
        self.menu_radius = 60
        self._menu_cxcy = (0, 0)

        # 3D manipulation
        self._prev_solid_angle = None
        self._prev_solid_dist = None
        self._prev_solid_mid = None

        # soft palette (BGR)
        self.col_line = (235, 245, 255)
        self.col_line_dim = (190, 205, 215)
        self.col_snap = (200, 255, 245)
        self.col_text = (235, 245, 255)
        self.col_shadow = (25, 25, 25)
        self.col_hint = (160, 190, 210)
        
        # ---- UI styling / scaling ----
        self.ui_scale = 1.0
        self.panel_alpha = 0.55
        self.panel_col = (18, 18, 22)   # dark glass
        self.panel_edge = (90, 115, 135)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_base = 0.62
        self.font_small_base = 0.48

        # Put the ring in a stable place (top-right) instead of between fingers
        self.menu_anchor_fixed = True


        # ---------- public API ----------
    def update(self, *args, **kwargs):
        """
        Supports BOTH call styles:
        A) update(pos_uv, pinch, active, w, h, dt=...)
        B) update(pos_uv, vel_uv, pinch, active, w, h, dt=...)
        """
        dt = float(kwargs.get("dt", 1.0 / 60.0))

        # ---- parse args FIRST (defines w,h) ----
        if len(args) == 5:
            pos_uv, pinch, active, w, h = args
            vel_uv = [(0.0, 0.0), (0.0, 0.0)]
        elif len(args) >= 6:
            pos_uv = args[0]
            looks_vel = (
                isinstance(args[1], (list, tuple))
                and len(args[1]) == 2
                and isinstance(args[1][0], (list, tuple))
                and len(args[1][0]) == 2
            )
            if looks_vel:
                vel_uv, pinch, active, w, h = args[1], args[2], args[3], args[4], args[5]
            else:
                vel_uv = [(0.0, 0.0), (0.0, 0.0)]
                pinch, active, w, h = args[1], args[2], args[3], args[4]
        else:
            raise TypeError("HUDCAD.update expected 5 or 6+ positional args")

        # ---- now safe: define W/H and use them everywhere ----
        W = int(w)
        H = int(h)

        # ---- adaptive UI scale ----
        self.ui_scale = float(_clamp(min(W, H) / 720.0, 0.85, 1.8))

        # scale interaction thresholds too
        self.snap_px = int(18 * self.ui_scale)
        self.axis_lock_px = int(14 * self.ui_scale)

        # ring radius scales
        self.menu_radius = int(60 * self.ui_scale)

        # ---- store pointer states ----
        self._v0 = (float(pos_uv[0][0]), float(pos_uv[0][1]))
        self._v1 = (float(pos_uv[1][0]), float(pos_uv[1][1]))
        self._p0 = (int(self._v0[0] * W), int(self._v0[1] * H))
        self._p1 = (int(self._v1[0] * W), int(self._v1[1] * H))

        self._vel0 = (float(vel_uv[0][0]), float(vel_uv[0][1])) if vel_uv else (0.0, 0.0)
        self._vel1 = (float(vel_uv[1][0]), float(vel_uv[1][1])) if vel_uv else (0.0, 0.0)

        a0 = 1 if active[0] else 0
        a1 = 1 if active[1] else 0

        p0 = float(pinch[0]) if a0 else 0.0
        p1 = float(pinch[1]) if a1 else 0.0

        # ---- menu anchor ----
        if self.menu_anchor_fixed:
            margin = int(22 * self.ui_scale)
            cx = W - margin - self.menu_radius
            cy = margin + self.menu_radius + int(18 * self.ui_scale)
            self._menu_cxcy = (int(cx), int(cy))
        else:
            mx = (self._p0[0] + self._p1[0]) // 2
            my = (self._p0[1] + self._p1[1]) // 2
            self._menu_cxcy = (mx, my)

        # ---- pinch hysteresis ----
        both_pinched = self._pinch_bool(p0, p1, a0, a1)

        # ---- gesture menu switching ----
        if self.menu_enabled and a0:
            self._gesture_menu_switch(p0, self._p0, self._menu_cxcy)

        # ---- tools ----
        if self.tool in (self.TOOL_LINE, self.TOOL_CURVE):
            self._tool_draw_line(W, H, dt, p0, p1, a0, a1, both_pinched)
        elif self.tool == self.TOOL_GRAB_2D:
            self._tool_grab_2d(p0, a0)
        elif self.tool == self.TOOL_SOLID_3D:
            self._tool_solid_3d(p0, p1, a0, a1, both_pinched)

        self._prev_both_pinched = both_pinched
        self._prev_pinch0 = p0


    def render(self, frame):
        # committed lines
        self._draw_lines(frame)

        # preview line (if drawing)
        if self._drawing:
            self._draw_preview(frame)

        # 3D solids wireframe
        self._draw_solids(frame)

        # HUD overlay
        self._draw_hud(frame)

        return frame

    def handle_key(self, key: int):
        if key == ord("t"):
            self.tool = (self.tool + 1) % 4
        elif key == ord("m"):
            self.render_mode = (self.render_mode + 1) % 2
        elif key == ord("l"):
            self.axis_lock = not self.axis_lock
        elif key == ord("g"):
            self.grid_snap = not self.grid_snap
        elif key == ord("["):
            self.grid_px = max(6, self.grid_px - 2)
        elif key == ord("]"):
            self.grid_px = min(80, self.grid_px + 2)
        elif key == ord("h"):
            self.menu_enabled = not self.menu_enabled
        elif key == ord("c") or key == ord("x"):
            self.clear()

    def clear(self):
        self.lines.clear()
        self.vertices.clear()
        self._chain_active = False
        self._chain = []
        self.solids.clear()
        self._active_solid = -1
        self._drawing = False
        self._prev_solid_angle = None
        self._prev_solid_dist = None
        self._prev_solid_mid = None

    # ---------- internals ----------
    def _pinch_bool(self, p0, p1, a0, a1) -> bool:
        if not (a0 and a1):
            return False
        # hysteresis: once active, keep until both drop below pinch_off
        if self._prev_both_pinched:
            return (p0 > self.pinch_off) and (p1 > self.pinch_off)
        return (p0 > self.pinch_on) and (p1 > self.pinch_on)

    def _gesture_menu_switch(self, p0, p0_px, center_px):
        # pinch edge near ring triggers action
        rising = (self._prev_pinch0 < self.pinch_off) and (p0 > self.pinch_on)
        if not rising:
            return

        if _dist(p0_px, center_px) > self.menu_radius * 1.35:
            return

        dx = p0_px[0] - center_px[0]
        dy = p0_px[1] - center_px[1]

        # quadrant selection
        if abs(dy) > abs(dx):
            if dy < 0:
                # TOP: LINE tool
                self.tool = self.TOOL_LINE
            else:
                # BOTTOM: CLEAR
                self.clear()
        else:
            if dx > 0:
                # RIGHT: CURVE tool
                self.tool = self.TOOL_CURVE
            else:
                # LEFT: cycle render mode
                self.render_mode = (self.render_mode + 1) % 2

    def _tool_draw_line(self, w, h, dt, p0, p1, a0, a1, both_pinched):
        if both_pinched:
            # start drawing
            if not self._drawing:
                self._drawing = True

            # snap endpoints
            a = self._apply_snaps(self._p0, w, h)
            b = self._apply_snaps(self._p1, w, h)

            # axis lock (optional)
            if self.axis_lock:
                a, b = self._axis_lock(a, b)

            self._preview_a = a
            self._preview_b = b
            self._preview_len = _dist(a, b)

            # curve control (optional)
            self._preview_curved = (self.tool == self.TOOL_CURVE)
            self._preview_ctrl = None
            if self._preview_curved:
                mid = _lerp(a, b, 0.5)
                # curvature from relative motion (creates “buttery swirls” when you arc your hands)
                rv = _sub(self._vel1, self._vel0)
                ab = _sub(b, a)
                perp = _perp(_norm(ab))
                k = _clamp((rv[0] * perp[0] + rv[1] * perp[1]) * 900.0, -120.0, 120.0)
                ctrl = _add(mid, _mul(perp, k))
                self._preview_ctrl = (int(ctrl[0]), int(ctrl[1]))

        # commit on release edge
        if self._prev_both_pinched and (not both_pinched) and self._drawing:
            self._drawing = False

            a = tuple(map(int, self._preview_a))
            b = tuple(map(int, self._preview_b))

            if _dist(a, b) < 6:
                return  # ignore tiny segments

            curved = bool(self._preview_curved)
            ctrl = self._preview_ctrl if curved else None
            seg = LineSeg(a=a, b=b, curved=curved, ctrl=ctrl, length_px=_dist(a, b))
            self.lines.append(seg)

            # chain logic for polygon creation
            ia = self._get_or_add_vertex(a)
            ib = self._get_or_add_vertex(b)
            self._update_chain(ia, ib)

    def _tool_grab_2d(self, p0, a0):
        if not a0:
            return
        pinched = (p0 > self.pinch_on) if not self._prev_both_pinched else (p0 > self.pinch_off)
        if not pinched:
            return

        # (simple) drag nearest vertex
        if len(self.vertices) == 0:
            return
        # find closest vertex to pointer0
        best_i = -1
        best_d = 1e9
        for i, v in enumerate(self.vertices):
            d = _dist(v, self._p0)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i >= 0 and best_d < self.snap_px * 1.5:
            # move that vertex to pointer position (respect grid if enabled)
            newp = self._apply_grid(self._p0)
            self.vertices[best_i] = newp

            # update lines endpoints that match that vertex exactly
            for seg in self.lines:
                if seg.a == seg.a and seg.a == seg.a:  # no-op; just clarity
                    pass
            for seg in self.lines:
                if seg.a == seg.a:
                    pass
            for seg in self.lines:
                if seg.a == self.vertices[best_i]:
                    pass

            # rebuild lines by snapping endpoints to nearest vertex index (cheap: per seg)
            self._rebind_lines_to_vertices()

    def _tool_solid_3d(self, p0, p1, a0, a1, both_pinched):
        # Select solid by pinching near it with pointer0
        if not a0:
            return

        if self._active_solid < 0 and (p0 > self.pinch_on) and (not both_pinched):
            self._active_solid = self._pick_solid(self._p0)

        if self._active_solid < 0:
            return

        s = self.solids[self._active_solid]

        # Move with single pinch
        if (p0 > self.pinch_on) and not both_pinched:
            # translate by pointer motion (screen-space)
            s.pos[0] = float(self._p0[0])
            s.pos[1] = float(self._p0[1])

        # Rotate/scale with both pinched
        if both_pinched and a1:
            v = _sub(self._p1, self._p0)
            ang = math.atan2(v[1], v[0])
            dist = math.hypot(v[0], v[1])

            mid = ((self._p0[0] + self._p1[0]) * 0.5, (self._p0[1] + self._p1[1]) * 0.5)

            if self._prev_solid_angle is not None:
                d_ang = ang - self._prev_solid_angle
                d_ang = (d_ang + math.pi) % (2 * math.pi) - math.pi
                s.yaw += d_ang * 1.2

            if self._prev_solid_dist is not None:
                if self._prev_solid_dist > 1e-3:
                    scale_ratio = dist / self._prev_solid_dist
                    s.scale = float(_clamp(s.scale * scale_ratio, 0.3, 3.0))

            if self._prev_solid_mid is not None:
                dm = _sub(mid, self._prev_solid_mid)
                s.pos[0] += float(dm[0])
                s.pos[1] += float(dm[1])

            self._prev_solid_angle = ang
            self._prev_solid_dist = dist
            self._prev_solid_mid = mid
        else:
            self._prev_solid_angle = None
            self._prev_solid_dist = None
            self._prev_solid_mid = None

    def _apply_grid(self, p):
        if not self.grid_snap:
            return (int(p[0]), int(p[1]))
        gx = int(round(p[0] / self.grid_px) * self.grid_px)
        gy = int(round(p[1] / self.grid_px) * self.grid_px)
        return (gx, gy)

    def _panel(self, frame, x, y, w, h):
        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(frame.shape[1], int(x + w))
        y1 = min(frame.shape[0], int(y + h))
        if x1 <= x0 or y1 <= y0:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), self.panel_col, -1)
        cv2.addWeighted(overlay, self.panel_alpha, frame, 1.0 - self.panel_alpha, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), self.panel_edge, 1, cv2.LINE_AA)


    def _apply_snaps(self, p, w, h):
        p = self._apply_grid(p)

        # magnet to nearest vertex
        best = None
        best_d = 1e9
        for v in self.vertices:
            d = _dist(p, v)
            if d < best_d:
                best_d = d
                best = v
        if best is not None and best_d <= self.snap_px:
            return best
        return p

    def _axis_lock(self, a, b):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        if abs(dx) < self.axis_lock_px:
            return a, (a[0], b[1])
        if abs(dy) < self.axis_lock_px:
            return a, (b[0], a[1])
        return a, b

    def _get_or_add_vertex(self, p):
        # exact match first
        for i, v in enumerate(self.vertices):
            if v[0] == p[0] and v[1] == p[1]:
                return i
        self.vertices.append(p)
        return len(self.vertices) - 1

    def _update_chain(self, ia, ib):
        if not self._chain_active:
            self._chain_active = True
            self._chain = [ia, ib]
            return

        # continue chain if connected to tail
        tail = self._chain[-1]
        head = self._chain[0]

        if ia == tail and ib != tail:
            self._chain.append(ib)
        elif ib == tail and ia != tail:
            self._chain.append(ia)
        else:
            # new disconnected segment -> start a new chain
            self._chain_active = True
            self._chain = [ia, ib]
            return

        # close loop if tail snapped to head
        if self._chain[-1] == head and len(self._chain) >= 4:
            loop = self._chain[:-1]  # drop repeated head
            self._chain_active = False
            self._chain = []

            if self.auto_extrude_on_close:
                self._make_solid_from_loop(loop)

    def _make_solid_from_loop(self, loop_vids):
        pts2 = np.array([self.vertices[i] for i in loop_vids], dtype=np.float32)

        # center the polygon
        cx = float(np.mean(pts2[:, 0]))
        cy = float(np.mean(pts2[:, 1]))
        base2 = pts2 - np.array([[cx, cy]], dtype=np.float32)

        # create 3D verts (bottom z=0, top z=depth)
        n = base2.shape[0]
        bottom = np.concatenate([base2, np.zeros((n, 1), dtype=np.float32)], axis=1)
        top = np.concatenate([base2, np.full((n, 1), self.extrude_depth, dtype=np.float32)], axis=1)
        verts = np.concatenate([bottom, top], axis=0)

        edges = []
        # perimeter bottom/top
        for i in range(n):
            j = (i + 1) % n
            edges.append((i, j))
            edges.append((i + n, j + n))
            edges.append((i, i + n))  # verticals

        s = Solid3D(
            verts=verts,
            edges=edges,
            center=np.array([0.0, 0.0, self.extrude_depth * 0.5], dtype=np.float32),
            yaw=0.3,
            pitch=0.65,
            scale=1.0,
            pos=np.array([cx, cy, 0.0], dtype=np.float32),
        )
        self.solids.append(s)
        self._active_solid = len(self.solids) - 1

    def _pick_solid(self, p):
        if len(self.solids) == 0:
            return -1
        best_i = -1
        best_d = 1e9
        for i, s in enumerate(self.solids):
            sx, sy = float(s.pos[0]), float(s.pos[1])
            d = math.hypot(p[0] - sx, p[1] - sy)
            if d < best_d:
                best_d = d
                best_i = i
        return best_i if best_d < 140 else -1

    def _rebind_lines_to_vertices(self):
        # For each segment endpoint, snap to nearest vertex within snap radius
        for seg in self.lines:
            seg.a = self._snap_point_to_vertex(seg.a)
            seg.b = self._snap_point_to_vertex(seg.b)
            seg.length_px = _dist(seg.a, seg.b)

    def _snap_point_to_vertex(self, p):
        best = None
        best_d = 1e9
        for v in self.vertices:
            d = _dist(p, v)
            if d < best_d:
                best_d = d
                best = v
        if best is not None and best_d <= self.snap_px * 1.2:
            return best
        return p

    # ---------- drawing ----------
    def _draw_lines(self, frame):
        for seg in self.lines:
            if self.render_mode == self.RENDER_NEON:
                self._draw_neon_line(frame, seg)
            else:
                self._draw_wire_line(frame, seg)

            # length label
            mid = _lerp(seg.a, seg.b, 0.5)
            self._text(frame, f"{seg.length_px:.1f}px", (int(mid[0] + 10), int(mid[1] - 10)), 0.55)

        # vertices
        for v in self.vertices:
            cv2.circle(frame, v, 3, self.col_snap, -1, lineType=cv2.LINE_AA)

    def _draw_wire_line(self, frame, seg: LineSeg):
        if not seg.curved or seg.ctrl is None:
            cv2.line(frame, seg.a, seg.b, self.col_line, 2, cv2.LINE_AA)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 24)
            cv2.polylines(frame, [pts], False, self.col_line, 2, cv2.LINE_AA)

    def _draw_neon_line(self, frame, seg: LineSeg):
        # soft glow
        glow = (120, 170, 210)
        if not seg.curved or seg.ctrl is None:
            cv2.line(frame, seg.a, seg.b, glow, 6, cv2.LINE_AA)
            cv2.line(frame, seg.a, seg.b, self.col_line, 2, cv2.LINE_AA)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 28)
            cv2.polylines(frame, [pts], False, glow, 6, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, self.col_line, 2, cv2.LINE_AA)

    def _draw_preview(self, frame):
        a = self._preview_a
        b = self._preview_b

        # endpoints
        cv2.circle(frame, a, 5, self.col_snap, -1, cv2.LINE_AA)
        cv2.circle(frame, b, 5, self.col_snap, -1, cv2.LINE_AA)

        if not self._preview_curved or self._preview_ctrl is None:
            cv2.line(frame, a, b, self.col_line_dim, 2, cv2.LINE_AA)
        else:
            c = self._preview_ctrl
            cv2.circle(frame, c, 4, self.col_hint, -1, cv2.LINE_AA)
            pts = self._quad_bezier(a, c, b, 24)
            cv2.polylines(frame, [pts], False, self.col_line_dim, 2, cv2.LINE_AA)

        mid = _lerp(a, b, 0.5)
        self._text(frame, f"{self._preview_len:.1f}px", (int(mid[0] + 10), int(mid[1] - 10)), 0.6)

    def _draw_solids(self, frame):
        for i, s in enumerate(self.solids):
            pts2 = self._project_solid(s)

            # draw edges
            col = (235, 245, 255) if i == self._active_solid else (180, 205, 220)
            glow = (120, 170, 210)

            for (a, b) in s.edges:
                pa = tuple(map(int, pts2[a]))
                pb = tuple(map(int, pts2[b]))
                if self.render_mode == self.RENDER_NEON:
                    cv2.line(frame, pa, pb, glow, 5, cv2.LINE_AA)
                cv2.line(frame, pa, pb, col, 2, cv2.LINE_AA)

            # label
            self._text(frame, "SOLID", (int(s.pos[0] + 10), int(s.pos[1] - 10)), 0.6)

    def _project_solid(self, s: Solid3D):
        # basic 3D transform + perspective-ish projection to 2D screen
        V = s.verts.copy()

        # scale
        V *= float(s.scale)

        # rotate yaw (z) + pitch (x) for a nice “Tony” tilt
        cy = math.cos(s.yaw)
        sy = math.sin(s.yaw)
        cp = math.cos(s.pitch)
        sp = math.sin(s.pitch)

        # yaw around Z
        x = V[:, 0] * cy - V[:, 1] * sy
        y = V[:, 0] * sy + V[:, 1] * cy
        z = V[:, 2]

        # pitch around X
        y2 = y * cp - z * sp
        z2 = y * sp + z * cp
        x2 = x

        # simple depth scale
        depth = 520.0
        zf = z2 + depth
        px = x2 / (zf / depth)
        py = y2 / (zf / depth)

        # translate to screen position
        px += float(s.pos[0])
        py += float(s.pos[1])

        return np.stack([px, py], axis=1)

    def _draw_hud(self, frame):
        H, W = frame.shape[:2]
        pad = int(14 * self.ui_scale)

        # Top glass bar
        top_h = int(62 * self.ui_scale)
        self._panel(frame, pad, pad, W - 2 * pad, top_h)

        tool_name = ["LINE", "CURVE", "GRAB2D", "SOLID3D"][self.tool]
        mode_name = ["WIRE", "NEON"][self.render_mode]
        msg1 = f"Tool: {tool_name}   Render: {mode_name}"
        msg2 = f"Snap: {self.snap_px}px   AxisLock: {'ON' if self.axis_lock else 'OFF'}   Grid: {'ON' if self.grid_snap else 'OFF'}"

        self._text(frame, msg1, (pad + int(16*self.ui_scale), pad + int(26*self.ui_scale)), self.font_base)
        self._text(frame, msg2, (pad + int(16*self.ui_scale), pad + int(50*self.ui_scale)), self.font_small_base)

        # Menu ring
        if self.menu_enabled:
            cx, cy = self._menu_cxcy
            r = self.menu_radius

            cv2.circle(frame, (cx, cy), r, (90, 115, 135), 1, cv2.LINE_AA)
            cv2.line(frame, (cx - r, cy), (cx + r, cy), (70, 92, 110), 1, cv2.LINE_AA)
            cv2.line(frame, (cx, cy - r), (cx, cy + r), (70, 92, 110), 1, cv2.LINE_AA)

            # compact labels
            self._text(frame, "LINE",  (cx - int(16*self.ui_scale), cy - r - int(10*self.ui_scale)), self.font_small_base)
            self._text(frame, "CURVE", (cx + r + int(8*self.ui_scale),  cy + int(6*self.ui_scale)), self.font_small_base)
            self._text(frame, "CLR",   (cx - int(12*self.ui_scale), cy + r + int(18*self.ui_scale)), self.font_small_base)
            self._text(frame, "MODE",  (cx - r - int(44*self.ui_scale), cy + int(6*self.ui_scale)), self.font_small_base)

        # Bottom hint bar (short + clean)
        bottom_h = int(38 * self.ui_scale)
        self._panel(frame, pad, H - pad - bottom_h, W - 2 * pad, bottom_h)
        hint = "Voice/Keys: T tool  M mode  L axis  G grid  C clear   |   Gesture ring: pinch pointer0 near ring"
        self._text(frame, hint, (pad + int(16*self.ui_scale), H - pad - int(12*self.ui_scale)), self.font_small_base)


    def _text(self, frame, s, org, scale):
        sc = float(scale) * self.ui_scale
        thick = 1 if sc < 0.9 else 2

        # subtle shadow
        cv2.putText(frame, s, (org[0] + 1, org[1] + 1),
                    self.font, sc, self.col_shadow, thick + 1, cv2.LINE_AA)
        cv2.putText(frame, s, org, self.font, sc,
                    self.col_text, thick, cv2.LINE_AA)

    def _quad_bezier(self, a, c, b, steps):
        pts = []
        for i in range(steps + 1):
            t = i / steps
            p0 = _lerp(a, c, t)
            p1 = _lerp(c, b, t)
            p = _lerp(p0, p1, t)
            pts.append([int(p[0]), int(p[1])])
        return np.array(pts, dtype=np.int32)
    
    def apply_voice(self, text: str):
        t = text.lower().strip()

        if "line" in t:
            self.tool = self.TOOL_LINE
        elif "curve" in t:
            self.tool = self.TOOL_CURVE
        elif "grab" in t:
            self.tool = self.TOOL_GRAB_2D
        elif "solid" in t or "3d" in t:
            self.tool = self.TOOL_SOLID_3D
        elif "clear" in t or "reset" in t:
            self.clear()
        elif "grid on" in t:
            self.grid_snap = True
        elif "grid off" in t:
            self.grid_snap = False
        elif "axis on" in t:
            self.axis_lock = True
        elif "axis off" in t:
            self.axis_lock = False
        elif "neon" in t:
            self.render_mode = self.RENDER_NEON
        elif "wire" in t:
            self.render_mode = self.RENDER_WIRE
        elif "menu on" in t:
            self.menu_enabled = True
        elif "menu off" in t:
            self.menu_enabled = False

