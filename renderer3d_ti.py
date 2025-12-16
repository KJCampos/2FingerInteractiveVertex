# hud_cad.py
# Futuristic HUD + geometry pipeline -> 2.5D mesh + 3D viewport render + physics.

from __future__ import annotations
import math
import numpy as np
import cv2

from mesh_25d import build_polygon_from_points, extrude_25d
from physics3d_ti import Physics3D
from renderer3d_ti import Renderer3D

def _clamp(x, a, b):
    return a if x < a else b if x > b else x


def _alpha_blend(dst_bgr, src_bgr, alpha):
    # alpha float 0..1
    return cv2.addWeighted(dst_bgr, 1.0 - alpha, src_bgr, alpha, 0)


def _dist2(a, b):
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return dx * dx + dy * dy


class HUDCAD:
    TOOL_LINE = 0
    TOOL_CURVE = 1

    def __init__(self):
        # modes
        self.tool = self.TOOL_LINE
        self.show_3d = True

        # UI state
        self.ui_scale = 1.0
        self.snap_px = 18
        self.close_px = 22

        # construction geometry
        self.segments = []          # list[(p0,p1)] pixels
        self.current_line = None    # (p0,p1) while pinched
        self.curve_pts = []         # current freehand stroke

        # model (after extrusion)
        self.model_poly_px = None   # (N,2) float32
        self.model_mesh = None      # (verts,tris) numpy
        self.model_ready = False

        # 3D systems
        self.physics = Physics3D(max_verts=4096)
        self.renderer = Renderer3D(width=520, height=520, max_verts=4096, max_tris=8192)

        # rotation controls
        self.dragging = False
        self.last_mouse = None
        self.mouse_sens = 0.010

        # nicer colors (not abrasive)
        self.col_bg_panel = (14, 18, 28)      # dark slate
        self.col_cyan = (160, 240, 255)       # soft cyan
        self.col_cyan_dim = (90, 170, 200)
        self.col_accent = (210, 160, 255)     # soft magenta accent
        self.col_text = (220, 235, 245)
        self.col_line = (120, 220, 245)
        self.col_hint = (120, 160, 180)

    # ---------------- input + update ----------------

    def update(self, pos_uv, vel_uv, pinch, active, w, h, dt=1/60):
        """
        Called every frame from app.py.
        pos_uv: [(u,v),(u,v)]
        pinch : [0..1, 0..1] (1 means pinched/closed)
        active: [0/1, 0/1]
        """
        W = int(w); H = int(h)
        self.ui_scale = float(_clamp(min(W, H) / 720.0, 0.85, 1.6))
        self.snap_px = int(18 * self.ui_scale)
        self.close_px = int(22 * self.ui_scale)

        # convert pointers to px
        p0 = (int(pos_uv[0][0] * W), int(pos_uv[0][1] * H))
        p1 = (int(pos_uv[1][0] * W), int(pos_uv[1][1] * H))
        a0 = 1 if active[0] else 0
        a1 = 1 if active[1] else 0
        pin0 = float(pinch[0]) if a0 else 0.0
        pin1 = float(pinch[1]) if a1 else 0.0

        both_pinched = (pin0 > 0.55) and (pin1 > 0.55) and a0 and a1

        # If model already exists, just advance physics
        if self.model_ready:
            self.physics.step(dt, substeps=2)
            return

        # ---------------- construction behavior ----------------
        if self.tool == self.TOOL_LINE:
            self._tool_line(p0, p1, pin0, pin1, a0, a1, both_pinched)

        elif self.tool == self.TOOL_CURVE:
            self._tool_curve(p0, pin0, a0)

        # closure check: if endpoints meet start within tolerance
        self._try_close_and_extrude(W, H)

    def _snap_point(self, p):
        # snaps to existing endpoints (magnet)
        best = p
        best_d2 = float(self.snap_px * self.snap_px)

        # endpoints from segments
        for (a, b) in self.segments:
            for q in (a, b):
                d2 = _dist2(p, q)
                if d2 < best_d2:
                    best_d2 = d2
                    best = q

        return best

    def _tool_line(self, p0, p1, pin0, pin1, a0, a1, both_pinched):
        # Two-pointer line: both pinch to start; release to commit
        if both_pinched:
            a = self._snap_point(p0)
            b = self._snap_point(p1)
            self.current_line = (a, b)
        else:
            if self.current_line is not None:
                self.segments.append(self.current_line)
                self.current_line = None

    def _tool_curve(self, p0, pin0, a0):
        # single-pointer freehand stroke: pinch to draw
        drawing = (a0 == 1) and (pin0 > 0.55)
        if drawing:
            sp = self._snap_point(p0)
            if not self.curve_pts or _dist2(sp, self.curve_pts[-1]) > (self.snap_px * 0.5) ** 2:
                self.curve_pts.append(sp)
        else:
            # commit curve into segments (polyline)
            if len(self.curve_pts) >= 2:
                for i in range(len(self.curve_pts) - 1):
                    self.segments.append((self.curve_pts[i], self.curve_pts[i + 1]))
            self.curve_pts = []

    def _ordered_polygon_from_segments(self):
        """
        Build a single ordered loop from segments if possible.
        Assumes you are drawing a loop (box-like, etc.).
        Returns list of points in order (pixels) or None.
        """
        if len(self.segments) < 3:
            return None

        # Build adjacency
        adj = {}
        def add_edge(a, b):
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)

        for (a, b) in self.segments:
            add_edge(a, b)

        # Find a start with degree 2 (loop) or any point
        start = None
        for k, v in adj.items():
            if len(v) == 2:
                start = k
                break
        if start is None:
            start = next(iter(adj.keys()))

        # Walk the loop
        loop = [start]
        prev = None
        cur = start
        guard = 0
        while guard < 5000:
            guard += 1
            neigh = adj.get(cur, [])
            if not neigh:
                return None
            # choose next not equal prev, else fallback
            nxt = None
            if prev is None:
                nxt = neigh[0]
            else:
                if len(neigh) == 1:
                    nxt = neigh[0]
                else:
                    nxt = neigh[0] if neigh[1] == prev else neigh[1]

            if nxt == start:
                # closed
                return loop
            if nxt in loop:
                # premature cycle
                return None

            loop.append(nxt)
            prev, cur = cur, nxt

        return None

    def _try_close_and_extrude(self, W, H):
        # Determine closure: either current line snaps to start, or segments already closed
        ordered = self._ordered_polygon_from_segments()
        if ordered is None:
            return
        if len(ordered) < 3:
            return

        # if last is near first (should already be a loop); extra safety:
        if _dist2(ordered[-1], ordered[0]) > self.close_px * self.close_px:
            # still allow but require explicit closure
            pass

        poly = build_polygon_from_points(ordered, min_dup_px=6 * self.ui_scale, spacing_px=8 * self.ui_scale)
        if len(poly) < 3:
            return

        mesh = extrude_25d(poly, W, H, scale=1.6, thickness=0.25)
        if len(mesh.verts) < 3 or len(mesh.tris) < 1:
            return

        # ---- requirement (3): remove construction lines once model is created ----
        self.segments = []
        self.current_line = None
        self.curve_pts = []

        self.model_poly_px = poly
        self.model_mesh = mesh
        self.model_ready = True

        # feed physics + renderer
        self.physics.set_mesh_vertices(mesh.verts)
        self.renderer.set_mesh(mesh.verts, mesh.tris)

    # ---------------- rendering ----------------

    def render(self, frame_bgr):
        """
        Draw HUD + construction or model overlay + optional 3D viewport.
        Returns composed frame (may be wider if 3D is enabled).
        """
        H, W = frame_bgr.shape[:2]
        ui = np.zeros_like(frame_bgr)
        self._draw_hud_base(ui, W, H)

        if not self.model_ready:
            self._draw_construction(ui)
            self._draw_tool_panel(ui, W, H)
        else:
            self._draw_model_overlay(ui, W, H)
            self._draw_model_panel(ui, W, H)

        out = _alpha_blend(frame_bgr, ui, 0.82)

        if self.show_3d and self.model_ready:
            # right viewport render
            pos, ang = self.physics.get_pose()
            self.renderer.set_pose(pos, ang)
            self.renderer.clear()
            self.renderer.render()
            view = self.renderer.get_image_bgr_u8()

            # match height
            vh, vw = view.shape[:2]
            scale = float(H) / float(vh)
            view_rs = cv2.resize(view, (int(vw * scale), H), interpolation=cv2.INTER_LINEAR)

            # subtle border
            cv2.rectangle(view_rs, (0, 0), (view_rs.shape[1] - 1, H - 1), self.col_cyan_dim, 1, cv2.LINE_AA)

            out = np.hstack([out, view_rs])

        return out

    def _draw_hud_base(self, img, W, H):
        # subtle gradient background panels
        # top bar
        bar_h = int(58 * self.ui_scale)
        panel = img.copy()
        cv2.rectangle(panel, (0, 0), (W, bar_h), self.col_bg_panel, -1)
        img[:] = _alpha_blend(img, panel, 0.85)

        # corner brackets
        self._corner_bracket(img, 10, 10, 70, 70)
        self._corner_bracket(img, W - 80, 10, W - 10, 70)
        self._corner_bracket(img, 10, H - 70, 70, H - 10)
        self._corner_bracket(img, W - 80, H - 70, W - 10, H - 10)

        # reticle center
        cx, cy = W // 2, H // 2
        cv2.circle(img, (cx, cy), int(18 * self.ui_scale), self.col_cyan_dim, 1, cv2.LINE_AA)
        cv2.line(img, (cx - int(28 * self.ui_scale), cy), (cx + int(28 * self.ui_scale), cy), self.col_cyan_dim, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - int(28 * self.ui_scale)), (cx, cy + int(28 * self.ui_scale)), self.col_cyan_dim, 1, cv2.LINE_AA)

        # title
        self._text(img, "2Finger CAD", 18, 36, self.col_text, scale=0.9)

    def _corner_bracket(self, img, x0, y0, x1, y1):
        t = int(2 * self.ui_scale)
        l = int(18 * self.ui_scale)
        c = self.col_cyan_dim
        cv2.line(img, (x0, y0), (x0 + l, y0), c, t, cv2.LINE_AA)
        cv2.line(img, (x0, y0), (x0, y0 + l), c, t, cv2.LINE_AA)
        cv2.line(img, (x1, y0), (x1 - l, y0), c, t, cv2.LINE_AA)
        cv2.line(img, (x1, y0), (x1, y0 + l), c, t, cv2.LINE_AA)
        cv2.line(img, (x0, y1), (x0 + l, y1), c, t, cv2.LINE_AA)
        cv2.line(img, (x0, y1), (x0, y1 - l), c, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1 - l, y1), c, t, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 - l), c, t, cv2.LINE_AA)

    def _text(self, img, s, x, y, col, scale=0.6, thick=1):
        cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                    float(scale * self.ui_scale), col, thick, cv2.LINE_AA)

    def _draw_tool_panel(self, img, W, H):
        y = int(36 * self.ui_scale)
        mode = "LINE" if self.tool == self.TOOL_LINE else "CURVE"
        self._text(img, f"MODE: {mode}", 160, y, self.col_cyan, scale=0.62)
        self._text(img, "Pinch both pointers to place a segment.", 160, y + int(18 * self.ui_scale), self.col_hint, scale=0.50)
        self._text(img, "Close the loop to extrude (any polygon).", 160, y + int(34 * self.ui_scale), self.col_hint, scale=0.50)

    def _draw_model_panel(self, img, W, H):
        y = int(36 * self.ui_scale)
        self._text(img, "MODEL READY", 160, y, self.col_cyan, scale=0.62)
        self._text(img, "Drag mouse on right viewport to rotate.", 160, y + int(18 * self.ui_scale), self.col_hint, scale=0.50)
        self._text(img, "Press R to reset rotation.", 160, y + int(34 * self.ui_scale), self.col_hint, scale=0.50)

    def _draw_construction(self, img):
        # segments
        for (a, b) in self.segments:
            cv2.line(img, a, b, self.col_line, 2, cv2.LINE_AA)
            cv2.circle(img, a, int(3 * self.ui_scale), self.col_cyan_dim, -1, cv2.LINE_AA)
            cv2.circle(img, b, int(3 * self.ui_scale), self.col_cyan_dim, -1, cv2.LINE_AA)

        # current line
        if self.current_line is not None:
            a, b = self.current_line
            cv2.line(img, a, b, self.col_accent, 2, cv2.LINE_AA)
            d = math.hypot(b[0] - a[0], b[1] - a[1])
            mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
            self._text(img, f"{d:.1f}px", mid[0] + 8, mid[1] - 8, self.col_text, scale=0.55)

        # curve preview
        if len(self.curve_pts) >= 2:
            cv2.polylines(img, [np.array(self.curve_pts, np.int32)], False, self.col_accent, 2, cv2.LINE_AA)

    def _draw_model_overlay(self, img, W, H):
        # Draw a pseudo-3D overlay on the left viewport (simple projection).
        # We do NOT draw old construction lines (requirement 3).
        if self.model_mesh is None:
            return

        verts = self.model_mesh.verts
        tris = self.model_mesh.tris
        # simple camera projection for overlay
        # This is intentionally lightweight; "true 3D" is right viewport.
        pos, ang = self.physics.get_pose()
        Rx, Ry, Rz = ang
        cx, cy = W // 2, H // 2

        def rot(v):
            x, y, z = v
            # y-rot
            c = math.cos(Ry); s = math.sin(Ry)
            x, z = x * c + z * s, -x * s + z * c
            # x-rot
            c = math.cos(Rx); s = math.sin(Rx)
            y, z = y * c - z * s, y * s + z * c
            return (x, y, z)

        # map to pixels
        pts2 = []
        for v in verts:
            x, y, z = rot(v)
            x += pos[0]; y += pos[1]; z += pos[2]
            # perspective-ish
            zz = (z + 3.2)
            sx = int(cx + x / zz * (W * 0.55))
            sy = int(cy - y / zz * (H * 0.55))
            pts2.append((sx, sy))
        pts2 = np.array(pts2, np.int32)

        # soft fill faces (front-to-back not sorted; okay for HUD overlay)
        face_col = (80, 175, 205)
        outline = (130, 220, 245)
        for (i0, i1, i2) in tris[: min(len(tris), 900)]:
            tri = np.array([pts2[i0], pts2[i1], pts2[i2]], np.int32)
            cv2.fillConvexPoly(img, tri, face_col, cv2.LINE_AA)

        # outline polygon
        # highlight top face boundary by projecting original poly if available
        if self.model_poly_px is not None:
            cv2.polylines(img, [self.model_poly_px.astype(np.int32)], True, outline, 2, cv2.LINE_AA)

    # ---------------- controls ----------------

    def on_key(self, key: int):
        if key == ord("1"):
            self.tool = self.TOOL_LINE
        elif key == ord("2"):
            self.tool = self.TOOL_CURVE
        elif key in (ord("c"), ord("C")):
            self.clear_all()
        elif key in (ord("v"), ord("V")):
            self.show_3d = not self.show_3d
        elif key in (ord("r"), ord("R")):
            self.physics.reset_rotation()

    def clear_all(self):
        self.segments = []
        self.current_line = None
        self.curve_pts = []
        self.model_poly_px = None
        self.model_mesh = None
        self.model_ready = False
        self.physics.enabled[None] = 0

    def on_mouse(self, event, x, y, flags, userdata=None):
        # mouse interacts primarily with right viewport (3D)
        # app shows combined frame: left=W, right=renderW
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            self.last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.last_mouse is not None:
            lx, ly = self.last_mouse
            dx = x - lx
            dy = y - ly
            self.last_mouse = (x, y)

            # apply rotation
            self.physics.add_rotation(dx * self.mouse_sens, dy * self.mouse_sens)
