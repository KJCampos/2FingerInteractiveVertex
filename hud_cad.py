from __future__ import annotations
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np

from mesh_25d import build_polygon_from_chain, extrude_polygon, Mesh25D
from physics3d_ti import Physics3D
from renderer3d_ti import Renderer3D


class HUDCAD:
    TOOL_LINE = 0

    def __init__(self):
        self.tool = self.TOOL_LINE
        self.render_mode = 0
        self.projection_mode = False

        self.snap_px = 18
        self.close_px = 20
        self.pinch_on = 0.65
        self.pinch_off = 0.55

        self.lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.vertices: List[Tuple[int, int]] = []
        self.chain: List[int] = []

        self.preview: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.drawing = False
        self.prev_both = False

        self.mesh_ready: Optional[Mesh25D] = None
        self._mesh_new = False

        self.physics = Physics3D()
        self.renderer = Renderer3D(width=460, height=460)
        self._dragging = False
        self._last_mouse = None

        self._ui_time = 0.0

    # ---------- input / update ----------
    def update(self, pos_uv, vel_uv, pinch, active, w, h, dt: float = 1 / 60.0):
        self._ui_time += float(dt)
        W = int(w); H = int(h)
        scale = max(0.85, min(1.6, min(W, H) / 720.0))
        self.snap_px = int(18 * scale)
        self.close_px = int(20 * scale)

        p0 = (int(pos_uv[0][0] * W), int(pos_uv[0][1] * H))
        p1 = (int(pos_uv[1][0] * W), int(pos_uv[1][1] * H))
        a0 = 1 if active[0] else 0
        a1 = 1 if active[1] else 0
        pin0 = float(pinch[0]) if a0 else 0.0
        pin1 = float(pinch[1]) if a1 else 0.0

        both_pinched = (pin0 > self.pinch_on and pin1 > self.pinch_on and a0 and a1) if not self.prev_both else (pin0 > self.pinch_off and pin1 > self.pinch_off and a0 and a1)

        if self.mesh_ready:
            self.physics.step(dt, substeps=2)
            pos, ang = self.physics.get_pose()
            self.renderer.set_pose(pos, ang)
            return

        if both_pinched:
            a = self._snap_point(p0)
            b = self._snap_point(p1)
            self.preview = (a, b)
            self.drawing = True
        else:
            if self.drawing and self.preview:
                a, b = self.preview
                if math.hypot(a[0] - b[0], a[1] - b[1]) > 4:
                    self._commit_segment(a, b)
            self.preview = None
            self.drawing = False

        self.prev_both = both_pinched

    def render(self, frame):
        h, w = frame.shape[:2]
        overlay = np.zeros_like(frame)
        self._draw_grid(overlay, w, h)
        self._draw_lines(overlay)
        self._draw_preview(overlay)
        self._draw_ui(overlay, w, h)

        composed = cv2.addWeighted(frame, 0.5, overlay, 0.7, 0)

        if self.mesh_ready:
            pos, ang = self.physics.get_pose()
            self.renderer.set_pose(pos, ang)
            view = self.renderer.render()
            view = cv2.resize(view, (view.shape[1], h))
            cv2.rectangle(view, (0, 0), (view.shape[1]-1, h-1), (120, 200, 240), 1, cv2.LINE_AA)
            composed = np.hstack([composed, view])

        return composed

    def handle_key(self, key: int):
        if key in (ord('c'), ord('x')):
            self.clear()
        if key in (ord('r'), ord('R')):
            self.physics.reset_rotation()
        if key in (ord('m'), ord('M')):
            self.render_mode = (self.render_mode + 1) % 4

    def handle_voice(self, cmd: str):
        cmd = cmd.lower().strip()
        if 'clear' in cmd or 'reset' in cmd:
            self.clear()
        elif 'projector' in cmd:
            self.projection_mode = not self.projection_mode

    def apply_voice(self, text: str):
        self.handle_voice(text)

    def on_mouse(self, event, x, y, flags, param=None):
        if not self.mesh_ready:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging and self._last_mouse:
            dx = (x - self._last_mouse[0]) * 0.005
            dy = (y - self._last_mouse[1]) * 0.005
            self.physics.add_rotation(dx, dy)
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._last_mouse = None

    def get_mesh_if_ready(self):
        if self._mesh_new and self.mesh_ready:
            self._mesh_new = False
            return self.mesh_ready.verts, self.mesh_ready.tris
        return None

    def clear(self):
        self.lines.clear()
        self.vertices.clear()
        self.chain.clear()
        self.preview = None
        self.mesh_ready = None
        self._mesh_new = False
        self.drawing = False
        self.prev_both = False

    # ---------- internals ----------
    def _snap_point(self, p):
        best = p
        best_d2 = self.snap_px * self.snap_px
        for v in self.vertices:
            d2 = (p[0] - v[0]) ** 2 + (p[1] - v[1]) ** 2
            if d2 < best_d2:
                best = v
                best_d2 = d2
        return best

    def _record_vertex(self, p) -> int:
        for i, v in enumerate(self.vertices):
            if v == p:
                return i
        self.vertices.append(p)
        return len(self.vertices) - 1

    def _commit_segment(self, a, b):
        ia = self._record_vertex(a)
        ib = self._record_vertex(b)
        self.lines.append((a, b))

        if not self.chain:
            self.chain = [ia, ib]
        else:
            if self.chain[-1] == ia:
                self.chain.append(ib)
            elif self.chain[-1] == ib:
                self.chain.append(ia)
            else:
                self.chain = [ia, ib]

        if len(self.chain) >= 3:
            first_v = self.vertices[self.chain[0]]
            last_v = self.vertices[self.chain[-1]]
            if math.hypot(first_v[0] - last_v[0], first_v[1] - last_v[1]) <= self.close_px:
                self.chain[-1] = self.chain[0]
                self._create_mesh_from_chain()

    def _create_mesh_from_chain(self):
        pts = [self.vertices[i] for i in self.chain[:-1]]
        poly = build_polygon_from_chain(pts, min_dist_px=self.snap_px * 0.5)
        if len(poly) < 3:
            return
        mesh = extrude_polygon(poly, scale=1.4, thickness=0.6 * max(20.0, self.snap_px))
        self.mesh_ready = mesh
        self._mesh_new = True
        self.lines.clear()
        self.vertices.clear()
        self.chain.clear()
        self.preview = None
        self.physics.set_mesh(mesh.verts, mesh.tris)
        self.renderer.set_mesh(mesh.verts, mesh.tris)

    # ---------- drawing helpers ----------
    def _draw_grid(self, img, w, h):
        step = max(16, self.snap_px * 2)
        color = (80, 120, 160)
        for x in range(0, w, step):
            cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
        for y in range(0, h, step):
            cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
        cv2.rectangle(img, (8, 8), (w - 8, h - 8), (120, 200, 240), 1, cv2.LINE_AA)

    def _draw_lines(self, img):
        for a, b in self.lines:
            cv2.line(img, a, b, (200, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, a, 4, (255, 200, 200), -1, cv2.LINE_AA)
            cv2.circle(img, b, 4, (255, 200, 200), -1, cv2.LINE_AA)

    def _draw_preview(self, img):
        if self.preview:
            a, b = self.preview
            cv2.line(img, a, b, (255, 180, 120), 2, cv2.LINE_AA)

    def _draw_ui(self, img, w, h):
        txt = "PINCH BOTH TO DRAW | Close loop to extrude"
        cv2.putText(img, txt, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 2, cv2.LINE_AA)
        if self.mesh_ready:
            cv2.putText(img, "Mesh ready - drag viewport to rotate, R to reset", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 255), 2, cv2.LINE_AA)
