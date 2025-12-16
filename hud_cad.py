from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np

from mesh_25d import build_polygon_from_chain, extrude_polygon, Mesh25D
from physics3d_ti import Physics3D
from renderer3d_ti import Renderer3D


@dataclass
class SceneObject:
    oid: int
    mesh: Mesh25D
    physics: Physics3D
    color: Tuple[int, int, int]

    @property
    def position(self):
        return self.physics.position

    @property
    def rotation(self):
        return self.physics.rotation

    def step(self, dt):
        self.physics.step(dt, substeps=2)

    def reset_rotation(self):
        self.physics.reset_rotation()

    def nudge(self, dx, dy, inertial: bool = False):
        if inertial:
            self.physics.add_rotation(dx, dy)
        else:
            self.physics.nudge_rotation(dx, dy)


class HUDCAD:
    TOOL_LINE = 0
    TOOL_CURVE = 1

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

        self.objects: List[SceneObject] = []
        self.selected_id: Optional[int] = None
        self._next_oid = 1
        self._last_centers = {}
        self._last_centers_cam = {}

        self.renderer = Renderer3D(width=460, height=460)
        self.zoom = 1.0
        self.zoom_min, self.zoom_max = 0.45, 2.5
        self.renderer.zoom = self.zoom

        self._dragging = False
        self._last_mouse = None
        self._last_finger = None
        self._rot_smooth = np.zeros(2, dtype=np.float32)

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

        # Update physics for all objects
        for obj in self.objects:
            obj.step(dt)

        rotating = False
        if self.selected_id is not None and not both_pinched and pin0 > self.pinch_on * 0.9:
            # Finger-based rotation using index finger delta
            dx = vel_uv[0][0] * W
            dy = vel_uv[0][1] * H
            self._rot_smooth = 0.65 * self._rot_smooth + 0.35 * np.array([dx, dy], dtype=np.float32)
            rot_scale = 0.008
            self._apply_rotation(self.selected_id, self._rot_smooth[0] * rot_scale, self._rot_smooth[1] * rot_scale, inertial=False)
            rotating = True
        else:
            self._rot_smooth *= 0.0

        if not rotating:
            self._last_finger = p0

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

        # Hover + pinch selection
        if pin0 > self.pinch_on and not both_pinched and self.objects:
            self._select_nearby(p0)

        self.prev_both = both_pinched

    def render(self, frame):
        h, w = frame.shape[:2]
        self._camera_width = w
        overlay = np.zeros_like(frame)
        self._draw_lines(overlay)
        self._draw_preview(overlay)
        self._draw_ui(overlay, w, h)

        composed = cv2.addWeighted(frame, 0.6, overlay, 0.9, 0)

        view, centers = self.renderer.render_scene(self.objects, selected_id=self.selected_id)
        scale = h / float(self.renderer.height)
        centers_scaled = {k: (v[0] * scale, v[1] * scale) for k, v in centers.items()}
        self._last_centers_cam = centers_scaled
        self._last_centers = {k: (v[0] + w, v[1]) for k, v in centers_scaled.items()}
        view = cv2.resize(view, (int(self.renderer.width * scale), h))
        composed = np.hstack([composed, view])

        self._draw_selection_bar(composed, w, h)

        return composed

    def handle_key(self, key: int):
        if key in (ord('c'), ord('C')):
            self.objects.clear()
            self.selected_id = None
        if key in (ord('x'), ord('X')):
            self.clear_strokes()
        if key in (ord('r'), ord('R')):
            if self.selected_id is not None:
                self._get_obj(self.selected_id).reset_rotation()
        if key in (ord('-'), 45):
            self._adjust_zoom(-0.08)
        if key in (ord('='), ord('+')):
            self._adjust_zoom(0.08)
        if key == ord('0'):
            self.zoom = 1.0
            self.renderer.zoom = self.zoom
        if key in (ord('m'), ord('M')):
            self.render_mode = (self.render_mode + 1) % 4
        if key == 255 or key == 0:
            return
        if key == 127:  # Delete
            self.delete_selected()
        if key in (ord('t'), ord('T')):
            self.tool = self.TOOL_CURVE if self.tool == self.TOOL_LINE else self.TOOL_LINE

    def handle_voice(self, cmd: str):
        cmd = cmd.lower().strip()
        if 'clear all' in cmd:
            self.objects.clear()
            self.selected_id = None
        elif cmd == 'clear' or 'clear ' in cmd:
            self.clear_strokes()
        elif 'delete' in cmd:
            self.delete_selected()
        elif 'reset' in cmd:
            if self.selected_id is not None:
                self._get_obj(self.selected_id).reset_rotation()
            self.zoom = 1.0
            self.renderer.zoom = self.zoom
        elif 'line' in cmd:
            self.tool = self.TOOL_LINE
        elif 'curve' in cmd:
            self.tool = self.TOOL_CURVE

    def apply_voice(self, text: str):
        self.handle_voice(text)

    def on_mouse(self, event, x, y, flags, param=None):
        # Mouse interactions target the 3D viewport on the right
        is_in_view = x > self._camera_width if hasattr(self, '_camera_width') else False
        view_x = x - (self._camera_width if is_in_view else 0)

        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self._adjust_zoom(delta * 0.08)
            return

        if not is_in_view:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._select_nearby((x, y))
            self._dragging = True
            self._last_mouse = (view_x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging and self._last_mouse and self.selected_id is not None:
            dx = (view_x - self._last_mouse[0]) * 0.02
            dy = (y - self._last_mouse[1]) * 0.02
            self._apply_rotation(self.selected_id, dx, dy, inertial=False)
            self._last_mouse = (view_x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._last_mouse = None

    def get_mesh_if_ready(self):
        return None

    def clear_strokes(self):
        self.lines.clear()
        self.vertices.clear()
        self.chain.clear()
        self.preview = None
        self.drawing = False
        self.prev_both = False

    # ---------- internals ----------
    def _apply_rotation(self, oid: int, dx: float, dy: float, inertial: bool):
        obj = self._get_obj(oid)
        if obj:
            obj.nudge(dx, dy, inertial=inertial)

    def _adjust_zoom(self, delta: float):
        self.zoom = max(self.zoom_min, min(self.zoom_max, self.zoom + delta))
        self.renderer.zoom = self.zoom

    def _select_nearby(self, pt):
        if not self.objects:
            return
        best = None
        best_d2 = (self.snap_px * 2) ** 2
        centers = self._last_centers if pt[0] > getattr(self, '_camera_width', 0) else self._last_centers_cam
        for oid, center in centers.items():
            dx = center[0] - pt[0]
            dy = center[1] - pt[1]
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best = oid
                best_d2 = d2
        if best is None:
            return
        self.selected_id = best

    def delete_selected(self):
        if self.selected_id is None:
            return
        self.objects = [o for o in self.objects if o.oid != self.selected_id]
        self.selected_id = self.objects[-1].oid if self.objects else None

    def _get_obj(self, oid: int) -> Optional[SceneObject]:
        for obj in self.objects:
            if obj.oid == oid:
                return obj
        return None

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
        self._add_object(mesh)
        self.clear_strokes()

    def _add_object(self, mesh: Mesh25D):
        color = random.choice([
            (140, 220, 255),
            (180, 200, 255),
            (160, 240, 210),
            (255, 210, 160),
        ])
        phys = Physics3D()
        phys.set_mesh(mesh.verts, mesh.tris)
        obj = SceneObject(oid=self._next_oid, mesh=mesh, physics=phys, color=color)
        self._next_oid += 1
        self.objects.append(obj)
        self.selected_id = obj.oid

    # ---------- drawing helpers ----------
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
        cv2.putText(img, "Mouse/finger to rotate, wheel or +/- to zoom, R reset", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 255), 2, cv2.LINE_AA)

    def _draw_selection_bar(self, composed, cam_w, h):
        # Store for mouse hit testing
        self._camera_width = cam_w
        if not self.objects:
            return
        bar_y = h - 50
        cv2.rectangle(composed, (10, bar_y - 26), (cam_w - 10, bar_y + 20), (40, 60, 80), 1, cv2.LINE_AA)
        x = 20
        for obj in self.objects:
            label = f"OBJ {obj.oid}"
            color = (255, 255, 180) if obj.oid == self.selected_id else (180, 220, 240)
            cv2.putText(composed, label, (x, bar_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            x += 90
