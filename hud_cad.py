# hud_cad.py - FUTURISTIC TONY STARK EDITION
# Holographic projection-ready CAD interface with advanced visual effects

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
    glow_intensity: float = 1.0
    creation_time: float = 0.0


@dataclass
class Solid3D:
    verts: np.ndarray
    edges: list[tuple[int, int]]
    center: np.ndarray
    yaw: float = 0.0
    pitch: float = 0.0
    scale: float = 1.0
    pos: np.ndarray = None
    energy: float = 1.0

    def __post_init__(self):
        if self.pos is None:
            self.pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class HUDCAD:
    TOOL_LINE = 0
    TOOL_CURVE = 1
    TOOL_GRAB_2D = 2
    TOOL_SOLID_3D = 3

    RENDER_HOLO = 0
    RENDER_NEON = 1
    RENDER_WIREFRAME = 2
    RENDER_XRAY = 3

    def __init__(self):
        # --- Visual modes ---
        self.tool = self.TOOL_LINE
        self.render_mode = self.RENDER_HOLO
        
        # --- Holographic effects ---
        self.holo_scan_speed = 2.5
        self.holo_scan_pos = 0.0
        self.holo_flicker_time = 0.0
        self.holo_flicker_intensity = 0.0
        self.projection_mode = True  # optimized for projector
        
        # --- Energy system ---
        self.global_energy = 1.0
        self.energy_pulse_phase = 0.0
        
        # --- Advanced visual parameters ---
        self.glow_layers = 3
        self.glow_blur_base = 15
        self.ambient_particles = []
        self.particle_spawn_time = 0.0
        
        # --- Interaction tuning ---
        self.pinch_on = 0.75
        self.pinch_off = 0.55
        self.snap_px = 22
        self.axis_lock = True
        self.axis_lock_px = 16
        self.grid_snap = False
        self.grid_px = 20
        self.auto_extrude_on_close = True
        self.extrude_depth = 70.0
        
        # --- State ---
        self.lines: list[LineSeg] = []
        self.vertices: list[tuple[int, int]] = []
        self._chain_active = False
        self._chain: list[int] = []
        self.solids: list[Solid3D] = []
        self._active_solid = -1
        
        # --- Drawing state ---
        self._drawing = False
        self._prev_both_pinched = False
        self._prev_pinch0 = 0.0
        self._p0 = (0, 0)
        self._p1 = (0, 0)
        self._v0 = (0.0, 0.0)
        self._v1 = (0.0, 0.0)
        self._vel0 = (0.0, 0.0)
        self._vel1 = (0.0, 0.0)
        
        # --- Menu ---
        self.menu_enabled = True
        self.menu_radius = 70
        self._menu_cxcy = (0, 0)
        self.menu_anchor_fixed = True
        
        # --- 3D manipulation ---
        self._prev_solid_angle = None
        self._prev_solid_dist = None
        self._prev_solid_mid = None
        
        # --- FUTURISTIC COLOR PALETTE (BGR) ---
        # Primary holographic cyan
        self.col_holo_cyan = (255, 245, 0)      # Electric cyan
        self.col_holo_blue = (255, 180, 0)      # Bright blue
        self.col_holo_teal = (255, 220, 80)     # Teal accent
        
        # Energy colors
        self.col_energy_core = (255, 255, 128)  # Bright energy
        self.col_energy_glow = (255, 200, 0)    # Blue-white glow
        
        # Accent colors
        self.col_accent_mag = (255, 100, 255)   # Magenta
        self.col_accent_orange = (0, 165, 255)  # Orange
        
        # UI elements
        self.col_ui_primary = (255, 255, 200)   # Light cyan text
        self.col_ui_secondary = (200, 200, 140) # Dimmed cyan
        self.col_ui_tertiary = (160, 160, 100)  # Dark cyan
        
        # Backgrounds
        self.col_bg_dark = (20, 15, 5)          # Deep blue-black
        self.col_bg_panel = (35, 25, 8)         # Panel dark
        
        # Grid and guides
        self.col_grid = (120, 100, 30)          # Subtle grid
        self.col_snap = (255, 255, 128)         # Snap highlight
        
        # --- UI styling ---
        self.ui_scale = 1.0
        self.panel_alpha = 0.75
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_base = 0.65
        self.font_small_base = 0.50
        
        # --- Time tracking ---
        self._time = 0.0

    # ==================== PUBLIC API ====================
    
    def update(self, *args, **kwargs):
        dt = float(kwargs.get("dt", 1.0 / 60.0))
        
        # Parse arguments
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
        
        W = int(w)
        H = int(h)
        
        # Update time-based effects
        self._time += dt
        self.holo_scan_pos = (self.holo_scan_pos + self.holo_scan_speed * dt) % 1.0
        self.energy_pulse_phase = (self.energy_pulse_phase + dt * 3.0) % (2 * math.pi)
        
        # Holographic flicker effect
        if np.random.random() < 0.02:
            self.holo_flicker_time = 0.15
            self.holo_flicker_intensity = np.random.uniform(0.3, 0.8)
        self.holo_flicker_time = max(0.0, self.holo_flicker_time - dt)
        
        # Adaptive UI scale
        self.ui_scale = float(_clamp(min(W, H) / 720.0, 0.85, 2.0))
        self.snap_px = int(22 * self.ui_scale)
        self.axis_lock_px = int(16 * self.ui_scale)
        self.menu_radius = int(70 * self.ui_scale)
        
        # Store pointer states
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
        
        # Menu anchor
        if self.menu_anchor_fixed:
            margin = int(30 * self.ui_scale)
            cx = W - margin - self.menu_radius
            cy = margin + self.menu_radius + int(20 * self.ui_scale)
            self._menu_cxcy = (int(cx), int(cy))
        else:
            mx = (self._p0[0] + self._p1[0]) // 2
            my = (self._p0[1] + self._p1[1]) // 2
            self._menu_cxcy = (mx, my)
        
        # Pinch detection
        both_pinched = self._pinch_bool(p0, p1, a0, a1)
        
        # Gesture menu
        if self.menu_enabled and a0:
            self._gesture_menu_switch(p0, self._p0, self._menu_cxcy)
        
        # Tool behavior
        if self.tool in (self.TOOL_LINE, self.TOOL_CURVE):
            self._tool_draw_line(W, H, dt, p0, p1, a0, a1, both_pinched)
        elif self.tool == self.TOOL_GRAB_2D:
            self._tool_grab_2d(p0, a0)
        elif self.tool == self.TOOL_SOLID_3D:
            self._tool_solid_3d(p0, p1, a0, a1, both_pinched)
        
        self._prev_both_pinched = both_pinched
        self._prev_pinch0 = p0
        
        # Update ambient particles
        self._update_particles(W, H, dt)

    def render(self, frame):
        h, w = frame.shape[:2]
        
        # Create holographic background layer
        holo_bg = self._create_holographic_background(w, h)
        frame = cv2.addWeighted(frame, 0.4, holo_bg, 0.6, 0)
        
        # Draw grid
        if self.grid_snap or self.render_mode == self.RENDER_HOLO:
            self._draw_holographic_grid(frame, w, h)
        
        # Draw ambient particles
        self._draw_particles(frame)
        
        # Draw geometry
        self._draw_lines(frame)
        
        # Draw preview
        if self._drawing:
            self._draw_preview(frame)
        
        # Draw 3D solids
        self._draw_solids(frame)
        
        # Draw scan lines
        if self.render_mode == self.RENDER_HOLO:
            self._draw_scan_lines(frame, w, h)
        
        # Draw HUD overlay
        self._draw_hud(frame, w, h)
        
        # Apply holographic flicker
        if self.holo_flicker_time > 0:
            frame = self._apply_flicker(frame)
        
        return frame

    def handle_key(self, key: int):
        if key == ord("t"):
            self.tool = (self.tool + 1) % 4
        elif key == ord("m"):
            self.render_mode = (self.render_mode + 1) % 4
        elif key == ord("l"):
            self.axis_lock = not self.axis_lock
        elif key == ord("g"):
            self.grid_snap = not self.grid_snap
        elif key == ord("p"):
            self.projection_mode = not self.projection_mode
        elif key == ord("["):
            self.grid_px = max(8, self.grid_px - 2)
        elif key == ord("]"):
            self.grid_px = min(100, self.grid_px + 2)
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
        self.ambient_particles = []

    def apply_voice(self, text: str):
        t = text.lower().strip()
        
        if "line" in t:
            self.tool = self.TOOL_LINE
        elif "curve" in t:
            self.tool = self.TOOL_CURVE
        elif "grab" in t:
            self.tool = self.TOOL_GRAB_2D
        elif "solid" in t or "3d" in t or "three d" in t:
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
        elif "holo" in t or "hologram" in t:
            self.render_mode = self.RENDER_HOLO
        elif "neon" in t:
            self.render_mode = self.RENDER_NEON
        elif "wire" in t or "wireframe" in t:
            self.render_mode = self.RENDER_WIREFRAME
        elif "xray" in t or "x-ray" in t or "x ray" in t:
            self.render_mode = self.RENDER_XRAY
        elif "menu on" in t:
            self.menu_enabled = True
        elif "menu off" in t:
            self.menu_enabled = False
        elif "projector" in t or "projection" in t:
            self.projection_mode = not self.projection_mode

    # ==================== VISUAL EFFECTS ====================
    
    def _create_holographic_background(self, w, h):
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:] = self.col_bg_dark
        
        # Add subtle gradient
        for y in range(h):
            alpha = y / h
            color = tuple(int(self.col_bg_dark[i] + alpha * 15) for i in range(3))
            bg[y, :] = color
        
        # Add energy pulses
        pulse = 0.5 + 0.5 * math.sin(self.energy_pulse_phase)
        pulse_color = tuple(int(self.col_bg_dark[i] + pulse * 8) for i in range(3))
        
        return bg
    
    def _draw_holographic_grid(self, frame, w, h):
        grid_step = self.grid_px if self.grid_snap else int(40 * self.ui_scale)
        
        # Perspective grid effect
        horizon_y = h // 3
        
        for x in range(0, w, grid_step):
            # Vertical lines with perspective fade
            alpha = 1.0 - abs(x - w // 2) / (w // 2) * 0.7
            color = tuple(int(self.col_grid[i] * alpha) for i in range(3))
            cv2.line(frame, (x, horizon_y), (x, h), color, 1, cv2.LINE_AA)
        
        for y in range(horizon_y, h, grid_step):
            alpha = 1.0 - (y - horizon_y) / (h - horizon_y) * 0.5
            color = tuple(int(self.col_grid[i] * alpha) for i in range(3))
            cv2.line(frame, (0, y), (w, y), color, 1, cv2.LINE_AA)
    
    def _draw_scan_lines(self, frame, w, h):
        scan_y = int(self.holo_scan_pos * h)
        
        # Main scan line
        for offset in range(-3, 4):
            y = scan_y + offset
            if 0 <= y < h:
                alpha = 1.0 - abs(offset) / 3.0
                color = tuple(int(self.col_holo_cyan[i] * alpha * 0.3) for i in range(3))
                cv2.line(frame, (0, y), (w, y), color, 1, cv2.LINE_AA)
    
    def _apply_flicker(self, frame):
        factor = 1.0 - self.holo_flicker_intensity * (self.holo_flicker_time / 0.15)
        return (frame * factor).astype(np.uint8)
    
    def _update_particles(self, w, h, dt):
        # Spawn new particles
        self.particle_spawn_time += dt
        if self.particle_spawn_time > 0.1:
            self.particle_spawn_time = 0.0
            if len(self.ambient_particles) < 50:
                self.ambient_particles.append({
                    'x': np.random.randint(0, w),
                    'y': np.random.randint(0, h),
                    'vx': np.random.uniform(-20, 20),
                    'vy': np.random.uniform(-20, 20),
                    'life': np.random.uniform(2.0, 4.0),
                    'size': np.random.uniform(1, 3)
                })
        
        # Update particles
        for p in self.ambient_particles:
            p['x'] += p['vx'] * dt
            p['y'] += p['vy'] * dt
            p['life'] -= dt
            
            # Wrap around
            if p['x'] < 0: p['x'] = w
            if p['x'] > w: p['x'] = 0
            if p['y'] < 0: p['y'] = h
            if p['y'] > h: p['y'] = 0
        
        # Remove dead particles
        self.ambient_particles = [p for p in self.ambient_particles if p['life'] > 0]
    
    def _draw_particles(self, frame):
        for p in self.ambient_particles:
            alpha = min(1.0, p['life'] / 2.0)
            color = tuple(int(self.col_holo_cyan[i] * alpha * 0.3) for i in range(3))
            center = (int(p['x']), int(p['y']))
            size = int(p['size'])
            cv2.circle(frame, center, size, color, -1, cv2.LINE_AA)

    # ==================== GEOMETRY RENDERING ====================
    
    def _draw_lines(self, frame):
        for seg in self.lines:
            if self.render_mode == self.RENDER_HOLO:
                self._draw_holographic_line(frame, seg)
            elif self.render_mode == self.RENDER_NEON:
                self._draw_neon_line(frame, seg)
            elif self.render_mode == self.RENDER_XRAY:
                self._draw_xray_line(frame, seg)
            else:
                self._draw_wire_line(frame, seg)
            
            # Enhanced label
            mid = _lerp(seg.a, seg.b, 0.5)
            self._draw_holographic_text(frame, f"{seg.length_px:.0f}", 
                                       (int(mid[0] + 10), int(mid[1] - 10)), 0.5)
        
        # Enhanced vertices
        for v in self.vertices:
            self._draw_vertex(frame, v)
    
    def _draw_holographic_line(self, frame, seg: LineSeg):
        if not seg.curved or seg.ctrl is None:
            pts = np.array([seg.a, seg.b], np.int32)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 32)
        
        # Multi-layer glow for holographic effect
        for layer in range(self.glow_layers):
            thickness = 2 + layer * 4
            alpha = (1.0 - layer / self.glow_layers) * 0.6
            color = tuple(int(self.col_holo_cyan[i] * alpha) for i in range(3))
            
            if len(pts.shape) == 1:
                cv2.line(frame, tuple(pts[0]), tuple(pts[1]), color, thickness, cv2.LINE_AA)
            else:
                cv2.polylines(frame, [pts], False, color, thickness, cv2.LINE_AA)
        
        # Core bright line
        if len(pts.shape) == 1:
            cv2.line(frame, tuple(pts[0]), tuple(pts[1]), self.col_energy_core, 2, cv2.LINE_AA)
        else:
            cv2.polylines(frame, [pts], False, self.col_energy_core, 2, cv2.LINE_AA)
    
    def _draw_neon_line(self, frame, seg: LineSeg):
        if not seg.curved or seg.ctrl is None:
            pts = [seg.a, seg.b]
            # Outer glow
            cv2.line(frame, seg.a, seg.b, self.col_holo_blue, 12, cv2.LINE_AA)
            cv2.line(frame, seg.a, seg.b, self.col_holo_cyan, 6, cv2.LINE_AA)
            cv2.line(frame, seg.a, seg.b, self.col_energy_core, 2, cv2.LINE_AA)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 32)
            cv2.polylines(frame, [pts], False, self.col_holo_blue, 12, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, self.col_holo_cyan, 6, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, self.col_energy_core, 2, cv2.LINE_AA)
    
    def _draw_xray_line(self, frame, seg: LineSeg):
        if not seg.curved or seg.ctrl is None:
            # Dashed X-ray effect
            self._draw_dashed_line(frame, seg.a, seg.b, self.col_holo_teal, 2)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 32)
            cv2.polylines(frame, [pts], False, self.col_holo_teal, 2, cv2.LINE_AA)
    
    def _draw_wire_line(self, frame, seg: LineSeg):
        if not seg.curved or seg.ctrl is None:
            cv2.line(frame, seg.a, seg.b, self.col_ui_primary, 2, cv2.LINE_AA)
        else:
            pts = self._quad_bezier(seg.a, seg.ctrl, seg.b, 24)
            cv2.polylines(frame, [pts], False, self.col_ui_primary, 2, cv2.LINE_AA)
    
    def _draw_vertex(self, frame, v):
        # Pulsing vertex
        pulse = 0.5 + 0.5 * math.sin(self._time * 5.0)
        size = int(4 + pulse * 2)
        
        # Outer glow
        cv2.circle(frame, v, size + 4, self.col_holo_cyan, 1, cv2.LINE_AA)
        cv2.circle(frame, v, size + 2, self.col_holo_cyan, 1, cv2.LINE_AA)
        
        # Core
        cv2.circle(frame, v, size, self.col_energy_core, -1, cv2.LINE_AA)
    
    def _draw_dashed_line(self, frame, a, b, color, thickness):
        dist = _dist(a, b)
        dash_length = 10
        gap_length = 5
        num_dashes = int(dist / (dash_length + gap_length))
        
        for i in range(num_dashes):
            t1 = i * (dash_length + gap_length) / dist
            t2 = (i * (dash_length + gap_length) + dash_length) / dist
            p1 = _lerp(a, b, t1)
            p2 = _lerp(a, b, t2)
            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 
                    color, thickness, cv2.LINE_AA)

    def _draw_preview(self, frame):
        a = self._preview_a
        b = self._preview_b
        
        # Enhanced preview endpoints
        for p in [a, b]:
            cv2.circle(frame, p, 8, self.col_accent_mag, 2, cv2.LINE_AA)
            cv2.circle(frame, p, 4, self.col_energy_core, -1, cv2.LINE_AA)
        
        if not self._preview_curved or self._preview_ctrl is None:
            # Animated preview line
            for i in range(3):
                offset = i * 4
                alpha = (1.0 - i / 3.0)
                color = tuple(int(self.col_accent_mag[i] * alpha) for i in range(3))
                cv2.line(frame, a, b, color, 3 + offset, cv2.LINE_AA)
            cv2.line(frame, a, b, self.col_energy_core, 2, cv2.LINE_AA)
        else:
            c = self._preview_ctrl
            cv2.circle(frame, c, 6, self.col_accent_orange, -1, cv2.LINE_AA)
            pts = self._quad_bezier(a, c, b, 32)
            
            for i in range(3):
                offset = i * 4
                alpha = (1.0 - i / 3.0)
                color = tuple(int(self.col_accent_mag[i] * alpha) for i in range(3))
                cv2.polylines(frame, [pts], False, color, 3 + offset, cv2.LINE_AA)
            cv2.polylines(frame, [pts], False, self.col_energy_core, 2, cv2.LINE_AA)
        
        mid = _lerp(a, b, 0.5)
        self._draw_holographic_text(frame, f"{self._preview_len:.0f}", 
                                    (int(mid[0] + 12), int(mid[1] - 12)), 0.6)

    def _draw_solids(self, frame):
        for i, s in enumerate(self.solids):
            pts2 = self._project_solid(s)
            
            is_active = (i == self._active_solid)
            col = self.col_energy_core if is_active else self.col_holo_cyan
            
            # Draw edges with holographic effect
            for (a, b) in s.edges:
                pa = tuple(map(int, pts2[a]))
                pb = tuple(map(int, pts2[b]))
                
                if self.render_mode == self.RENDER_HOLO or self.render_mode == self.RENDER_NEON:
                    # Multi-layer glow
                    cv2.line(frame, pa, pb, self.col_holo_blue, 8, cv2.LINE_AA)
                    cv2.line(frame, pa, pb, col, 4, cv2.LINE_AA)
                    cv2.line(frame, pa, pb, self.col_energy_core, 2, cv2.LINE_AA)
                else:
                    cv2.line(frame, pa, pb, col, 2, cv2.LINE_AA)
            
            # Label
            label_pos = (int(s.pos[0] + 15), int(s.pos[1] - 15))
            self._draw_holographic_text(frame, "SOLID-3D", label_pos, 0.55)

    def _project_solid(self, s: Solid3D):
        V = s.verts.copy()
        V *= float(s.scale)
        
        cy = math.cos(s.yaw)
        sy = math.sin(s.yaw)
        cp = math.cos(s.pitch)
        sp = math.sin(s.pitch)
        
        # Yaw (Z-axis)
        x = V[:, 0] * cy - V[:, 1] * sy
        y = V[:, 0] * sy + V[:, 1] * cy
        z = V[:, 2]
        
        # Pitch (X-axis)
        y2 = y * cp - z * sp
        z2 = y * sp + z * cp
        x2 = x
        
        # Perspective projection
        depth = 600.0
        zf = z2 + depth
        px = x2 / (zf / depth)
        py = y2 / (zf / depth)
        
        px += float(s.pos[0])
        py += float(s.pos[1])
        
        return np.stack([px, py], axis=1)

    # ==================== HUD OVERLAY ====================
    
    def _draw_hud(self, frame, w, h):
        pad = int(16 * self.ui_scale)
        
        # Top holographic panel
        self._draw_holographic_panel(frame, pad, pad, w - 2 * pad, int(80 * self.ui_scale))
        
        tool_names = ["LINE", "CURVE", "GRAB-2D", "SOLID-3D"]
        mode_names = ["HOLOGRAM", "NEON", "WIREFRAME", "X-RAY"]
        
        tool_name = tool_names[self.tool]
        mode_name = mode_names[self.render_mode]
        
        # Main status
        y = pad + int(32 * self.ui_scale)
        self._draw_holographic_text(frame, f"TOOL: {tool_name}", 
                                    (pad + int(20 * self.ui_scale), y), self.font_base)
        self._draw_holographic_text(frame, f"RENDER: {mode_name}", 
                                    (pad + int(20 * self.ui_scale), y + int(24 * self.ui_scale)), 
                                    self.font_small_base)
        
        # System info
        self._draw_holographic_text(frame, f"SNAP: {self.snap_px}px", 
                                    (pad + int(320 * self.ui_scale), y), self.font_small_base)
        self._draw_holographic_text(frame, f"AXIS: {'LOCKED' if self.axis_lock else 'FREE'}", 
                                    (pad + int(320 * self.ui_scale), y + int(24 * self.ui_scale)), 
                                    self.font_small_base)
        
        # Menu ring
        if self.menu_enabled:
            self._draw_gesture_menu(frame)
        
        # Corner brackets
        self._draw_corner_brackets(frame, w, h)
        
        # Bottom panel
        bottom_h = int(50 * self.ui_scale)
        self._draw_holographic_panel(frame, pad, h - pad - bottom_h, 
                                     w - 2 * pad, bottom_h)
        
        # Controls hint
        hint = "VOICE: 'robin' + command  |  KEYS: T tool  M mode  L axis  G grid  C clear  P projector"
        self._draw_holographic_text(frame, hint, 
                                    (pad + int(20 * self.ui_scale), 
                                     h - pad - int(16 * self.ui_scale)), 
                                    self.font_small_base)
    
    def _draw_holographic_panel(self, frame, x, y, w, h):
        x0, y0 = int(x), int(y)
        x1, y1 = int(x + w), int(y + h)
        
        # Dark background with border
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), self.col_bg_panel, -1)
        frame[:] = cv2.addWeighted(frame, 1.0 - self.panel_alpha, overlay, self.panel_alpha, 0)
        
        # Glowing border
        cv2.rectangle(frame, (x0, y0), (x1, y1), self.col_holo_cyan, 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1 + 1, y1 + 1), self.col_ui_tertiary, 1, cv2.LINE_AA)
    
    def _draw_corner_brackets(self, frame, w, h):
        bracket_len = int(40 * self.ui_scale)
        thickness = int(3 * self.ui_scale)
        pad = int(12 * self.ui_scale)
        
        corners = [
            (pad, pad),
            (w - pad, pad),
            (pad, h - pad),
            (w - pad, h - pad)
        ]
        
        for i, (cx, cy) in enumerate(corners):
            dx = 1 if i % 2 == 1 else -1
            dy = 1 if i >= 2 else -1
            
            # Horizontal
            cv2.line(frame, (cx, cy), (cx + dx * bracket_len, cy), 
                    self.col_holo_cyan, thickness, cv2.LINE_AA)
            # Vertical
            cv2.line(frame, (cx, cy), (cx, cy + dy * bracket_len), 
                    self.col_holo_cyan, thickness, cv2.LINE_AA)
            
            # Corner glow
            cv2.circle(frame, (cx, cy), 4, self.col_energy_core, -1, cv2.LINE_AA)
    
    def _draw_gesture_menu(self, frame):
        cx, cy = self._menu_cxcy
        r = self.menu_radius
        
        # Animated rings
        pulse = 0.7 + 0.3 * math.sin(self._time * 4.0)
        
        # Outer ring
        cv2.circle(frame, (cx, cy), r, self.col_holo_cyan, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * pulse), self.col_ui_tertiary, 1, cv2.LINE_AA)
        
        # Cross
        cv2.line(frame, (cx - r, cy), (cx + r, cy), self.col_ui_secondary, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), self.col_ui_secondary, 2, cv2.LINE_AA)
        
        # Center
        cv2.circle(frame, (cx, cy), 6, self.col_energy_core, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, self.col_holo_cyan, 1, cv2.LINE_AA)
        
        # Quadrant labels with icons
        label_offset = int(r + 20 * self.ui_scale)
        self._draw_holographic_text(frame, "LINE", (cx - 20, cy - label_offset), 0.5)
        self._draw_holographic_text(frame, "CURVE", (cx + label_offset - 30, cy + 6), 0.5)
        self._draw_holographic_text(frame, "CLEAR", (cx - 25, cy + label_offset + 10), 0.5)
        self._draw_holographic_text(frame, "MODE", (cx - label_offset - 10, cy + 6), 0.5)
    
    def _draw_holographic_text(self, frame, text, pos, scale):
        sc = float(scale) * self.ui_scale
        thickness = max(1, int(sc * 2))
        
        # Glow layers
        for i in range(2):
            offset = i + 1
            alpha = (1.0 - i / 2.0) * 0.5
            color = tuple(int(self.col_holo_cyan[j] * alpha) for j in range(3))
            cv2.putText(frame, text, (pos[0] + offset, pos[1] + offset),
                       self.font, sc, color, thickness + i, cv2.LINE_AA)
        
        # Core text
        cv2.putText(frame, text, pos, self.font, sc, 
                   self.col_energy_core, thickness, cv2.LINE_AA)

    # ==================== TOOL INTERNALS ====================
    
    def _pinch_bool(self, p0, p1, a0, a1) -> bool:
        if not (a0 and a1):
            return False
        if self._prev_both_pinched:
            return (p0 > self.pinch_off) and (p1 > self.pinch_off)
        return (p0 > self.pinch_on) and (p1 > self.pinch_on)
    
    def _gesture_menu_switch(self, p0, p0_px, center_px):
        rising = (self._prev_pinch0 < self.pinch_off) and (p0 > self.pinch_on)
        if not rising:
            return
        
        if _dist(p0_px, center_px) > self.menu_radius * 1.35:
            return
        
        dx = p0_px[0] - center_px[0]
        dy = p0_px[1] - center_px[1]
        
        if abs(dy) > abs(dx):
            if dy < 0:
                self.tool = self.TOOL_LINE
            else:
                self.clear()
        else:
            if dx > 0:
                self.tool = self.TOOL_CURVE
            else:
                self.render_mode = (self.render_mode + 1) % 4
    
    def _tool_draw_line(self, w, h, dt, p0, p1, a0, a1, both_pinched):
        if both_pinched:
            if not self._drawing:
                self._drawing = True
            
            a = self._apply_snaps(self._p0, w, h)
            b = self._apply_snaps(self._p1, w, h)
            
            if self.axis_lock:
                a, b = self._axis_lock(a, b)
            
            self._preview_a = a
            self._preview_b = b
            self._preview_len = _dist(a, b)
            
            self._preview_curved = (self.tool == self.TOOL_CURVE)
            self._preview_ctrl = None
            if self._preview_curved:
                mid = _lerp(a, b, 0.5)
                rv = _sub(self._vel1, self._vel0)
                ab = _sub(b, a)
                perp = _perp(_norm(ab))
                k = _clamp((rv[0] * perp[0] + rv[1] * perp[1]) * 900.0, -120.0, 120.0)
                ctrl = _add(mid, _mul(perp, k))
                self._preview_ctrl = (int(ctrl[0]), int(ctrl[1]))
        
        if self._prev_both_pinched and (not both_pinched) and self._drawing:
            self._drawing = False
            
            a = tuple(map(int, self._preview_a))
            b = tuple(map(int, self._preview_b))
            
            if _dist(a, b) < 6:
                return
            
            curved = bool(self._preview_curved)
            ctrl = self._preview_ctrl if curved else None
            seg = LineSeg(a=a, b=b, curved=curved, ctrl=ctrl, 
                         length_px=_dist(a, b), creation_time=self._time)
            self.lines.append(seg)
            
            ia = self._get_or_add_vertex(a)
            ib = self._get_or_add_vertex(b)
            self._update_chain(ia, ib)
    
    def _tool_grab_2d(self, p0, a0):
        if not a0:
            return
        pinched = (p0 > self.pinch_on) if not self._prev_both_pinched else (p0 > self.pinch_off)
        if not pinched:
            return
        
        if len(self.vertices) == 0:
            return
        
        best_i = -1
        best_d = 1e9
        for i, v in enumerate(self.vertices):
            d = _dist(v, self._p0)
            if d < best_d:
                best_d = d
                best_i = i
        
        if best_i >= 0 and best_d < self.snap_px * 1.5:
            newp = self._apply_grid(self._p0)
            self.vertices[best_i] = newp
            self._rebind_lines_to_vertices()
    
    def _tool_solid_3d(self, p0, p1, a0, a1, both_pinched):
        if not a0:
            return
        
        if self._active_solid < 0 and (p0 > self.pinch_on) and (not both_pinched):
            self._active_solid = self._pick_solid(self._p0)
        
        if self._active_solid < 0:
            return
        
        s = self.solids[self._active_solid]
        
        if (p0 > self.pinch_on) and not both_pinched:
            s.pos[0] = float(self._p0[0])
            s.pos[1] = float(self._p0[1])
        
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
    
    def _apply_snaps(self, p, w, h):
        p = self._apply_grid(p)
        
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
        
        tail = self._chain[-1]
        head = self._chain[0]
        
        if ia == tail and ib != tail:
            self._chain.append(ib)
        elif ib == tail and ia != tail:
            self._chain.append(ia)
        else:
            self._chain_active = True
            self._chain = [ia, ib]
            return
        
        if self._chain[-1] == head and len(self._chain) >= 4:
            loop = self._chain[:-1]
            self._chain_active = False
            self._chain = []
            
            if self.auto_extrude_on_close:
                self._make_solid_from_loop(loop)
    
    def _make_solid_from_loop(self, loop_vids):
        pts2 = np.array([self.vertices[i] for i in loop_vids], dtype=np.float32)
        
        cx = float(np.mean(pts2[:, 0]))
        cy = float(np.mean(pts2[:, 1]))
        base2 = pts2 - np.array([[cx, cy]], dtype=np.float32)
        
        n = base2.shape[0]
        bottom = np.concatenate([base2, np.zeros((n, 1), dtype=np.float32)], axis=1)
        top = np.concatenate([base2, np.full((n, 1), self.extrude_depth, dtype=np.float32)], axis=1)
        verts = np.concatenate([bottom, top], axis=0)
        
        edges = []
        for i in range(n):
            j = (i + 1) % n
            edges.append((i, j))
            edges.append((i + n, j + n))
            edges.append((i, i + n))
        
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
    
    def _quad_bezier(self, a, c, b, steps):
        pts = []
        for i in range(steps + 1):
            t = i / steps
            p0 = _lerp(a, c, t)
            p1 = _lerp(c, b, t)
            p = _lerp(p0, p1, t)
            pts.append([int(p[0]), int(p[1])])
        return np.array(pts, dtype=np.int32)