# pyright: reportInvalidTypeForm=false
import time
import numpy as np
import taichi as ti
import cv2


def _pget(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


@ti.data_oriented
class ParticleSimTaichi:
    TOOL_DRAW = 0
    TOOL_EXTRUDE = 1

    def __init__(self, params):
        self.params = params

        # ---------------- Grid ----------------
        self.W = int(_pget(params, "grid_w", 240))
        self.H = int(_pget(params, "grid_h", 135))

        # ---------------- Time ----------------
        self.dt_max = float(_pget(params, "dt_max", 1.0 / 30.0))
        self.advect_strength = float(_pget(params, "advect_strength", 1.0))

        # ---------------- Fluid stability ----------------
        self.viscosity = float(_pget(params, "viscosity", 0.0020))
        self.diffuse_iters = int(_pget(params, "diffuse_iters", 10))
        self.project_iters = int(_pget(params, "project_iters", 22))

        # ---------------- Dye ----------------
        self.dye_decay = float(_pget(params, "dye_decay", 0.988))
        self.dye_inject = float(_pget(params, "dye_inject", 0.9))

        # ---------------- Input scaling ----------------
        self.input_vel_scale = float(_pget(params, "input_vel_scale", 0.45))
        self.vel_limit = float(_pget(params, "vel_limit", 55.0))

        # ---------------- Pointer influence ----------------
        self.force_radius = float(_pget(params, "force_radius", 12.0))
        self.force_sigma = float(_pget(params, "force_sigma", self.force_radius * 0.60))
        self.force_sigma2 = self.force_sigma * self.force_sigma

        # softer than before
        self.push_gain = float(_pget(params, "push_gain", 0.65))
        self.swirl_gain = float(_pget(params, "swirl_gain", 1.10))
        self.pinch_gain = float(_pget(params, "pinch_gain", 0.95))

        # ---------------- Two-pointer sheet interaction ----------------
        self.pair_grab_gain = float(_pget(params, "pair_grab_gain", 1.10))
        self.pair_stretch_gain = float(_pget(params, "pair_stretch_gain", 0.60))
        self.pair_radius = float(_pget(params, "pair_radius", 10.0))
        self.pair_sigma2 = (self.pair_radius * 0.85) ** 2

        # ---------------- Wall drawing (thin strokes) ----------------
        self.draw_obstacle_when_pinch = float(_pget(params, "draw_obstacle_when_pinch", 0.80))
        self.obstacle_thickness = float(_pget(params, "obstacle_thickness", 2.0))  # grid cells

        # ---------------- Extrude tool ----------------
        self.tool_mode = int(_pget(params, "tool_mode", self.TOOL_DRAW))
        self.extrude_pinch_min = float(_pget(params, "extrude_pinch_min", 0.55))
        self.extrude_gain = float(_pget(params, "extrude_gain", 5.0))
        self.extrude_dye = float(_pget(params, "extrude_dye", 2.5))

        # ---------------- Rendering (softer) ----------------
        self.render_mode = int(_pget(params, "render_mode", 0))  # 0 dye, 1 vort, 2 pressure, 3 speed
        self.render_blur_ksize = int(_pget(params, "render_blur_ksize", 21))
        if self.render_blur_ksize % 2 == 0:
            self.render_blur_ksize += 1

        self.render_alpha = float(_pget(params, "render_alpha", 0.34))
        self.colormap = int(_pget(params, "colormap", cv2.COLORMAP_VIRIDIS))
        self.cmap_saturation = float(_pget(params, "cmap_saturation", 0.50))

        # ---------------- Fields ----------------
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))
        self.vel0 = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))
        self.vel1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))

        self.dye = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.dye0 = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        self.div = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.p = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.p0 = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        self.obst = ti.field(dtype=ti.i32, shape=(self.W, self.H))

        self.vort = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.speed = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.div_abs = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        # Hand inputs
        self.hand_pos_uv = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
        self.hand_vel_uv = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
        self.hand_pinch = ti.field(dtype=ti.f32, shape=(2,))
        self.hand_present = ti.field(dtype=ti.i32, shape=(2,))

        # Stroke memory (grid coords)
        self.draw_prev = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
        self.draw_has_prev = ti.field(dtype=ti.i32, shape=(2,))

        # ---------- gesture menu state (python-side) ----------
        self._pinch_prev = [0.0, 0.0]
        self._menu_cooldown = 0.0
        self._menu_hover = ""  # "TOP/RIGHT/BOTTOM/LEFT" for rendering

        self.reset()

    # ===================== Public API =====================

    def set_hand_input(self, pos_uv, vel_uv, pinch, hand_present=True, hand_id=0):
        hid = int(hand_id)
        if hid < 0 or hid > 1:
            return
        self.hand_pos_uv[hid] = ti.Vector([float(pos_uv[0]), float(pos_uv[1])])
        self.hand_vel_uv[hid] = ti.Vector([float(vel_uv[0]), float(vel_uv[1])])
        self.hand_pinch[hid] = float(max(0.0, min(1.0, pinch)))
        self.hand_present[hid] = 1 if hand_present else 0

    def get_positions(self) -> np.ndarray:
        return np.zeros((0, 2), dtype=np.float32)

    def reset(self):
        self._clear_all_kernel()

    def clear_obstacles(self):
        self._clear_obstacles_kernel()

    def clear_dye(self):
        self._clear_dye_kernel()

    def handle_key(self, key: int):
        if key is None:
            return

        if key in (ord('0'), ord('1'), ord('2'), ord('3')):
            self.render_mode = int(chr(key))
            return

        if key in (ord('c'), ord('C')):
            self.clear_obstacles()
            self.clear_dye()
            return

        if key in (ord('r'), ord('R')):
            self.reset()
            return

        if key in (ord('t'), ord('T')):
            self.tool_mode = self.TOOL_EXTRUDE if self.tool_mode == self.TOOL_DRAW else self.TOOL_DRAW
            return

        if key == ord('['):
            self.obstacle_thickness = max(1.0, self.obstacle_thickness - 0.5)
            return

        if key == ord(']'):
            self.obstacle_thickness = min(8.0, self.obstacle_thickness + 0.5)
            return

        if key in (ord('m'), ord('M')):
            options = [cv2.COLORMAP_VIRIDIS, cv2.COLORMAP_CIVIDIS, cv2.COLORMAP_OCEAN, cv2.COLORMAP_BONE]
            try:
                idx = options.index(self.colormap)
            except ValueError:
                idx = 0
            self.colormap = options[(idx + 1) % len(options)]
            return

    def step(self, dt):
        dt = float(dt)
        if dt <= 0:
            return
        dt = min(dt, self.dt_max)

        # gesture menu switching (python-side, zero app.py changes)
        self._gesture_menu_update(dt)

        # 1) Inputs (forces OR tools)
        self._apply_inputs(dt, int(self.tool_mode), float(self.obstacle_thickness))
        self._apply_obstacles()
        self._clamp_velocity()

        # 2) Viscosity
        if self.viscosity > 0.0 and self.diffuse_iters > 0:
            self._diffuse_velocity(dt)
            self._apply_obstacles()
            self._clamp_velocity()

        # 3) Project incompressible
        self._project()
        self._apply_obstacles()

        # 4) Advect velocity
        self._advect_velocity(dt)
        self._apply_obstacles()
        self._clamp_velocity()

        # 5) Project again
        self._project()
        self._apply_obstacles()

        # 6) Advect dye
        self._advect_dye(dt)
        self._apply_obstacles()

        # 7) Decay + diagnostics
        self._decay_dye()
        self._compute_diagnostics()

    # ===================== Gesture menu (python-side) =====================

    def _gesture_menu_update(self, dt: float):
        self._menu_hover = ""
        self._menu_cooldown = max(0.0, self._menu_cooldown - dt)

        hp = self.hand_present.to_numpy()
        if hp[0] != 1 or hp[1] != 1:
            # update pinch prev only
            self._pinch_prev[0] = float(self.hand_pinch.to_numpy()[0])
            self._pinch_prev[1] = float(self.hand_pinch.to_numpy()[1])
            return

        pos = self.hand_pos_uv.to_numpy()
        pinch = self.hand_pinch.to_numpy()

        p0 = pos[0]
        p1 = pos[1]
        mid = (p0 + p1) * 0.5

        # Use hand0 as the "cursor" for selection (works with your current pointer mapping)
        cur = p0
        dx = float(cur[0] - mid[0])
        dy = float(cur[1] - mid[1])

        r2 = dx * dx + dy * dy
        inner = 0.035
        outer = 0.160
        if r2 < inner * inner or r2 > outer * outer:
            self._pinch_prev[0] = float(pinch[0])
            self._pinch_prev[1] = float(pinch[1])
            return

        # Determine quadrant (y increases downward)
        if abs(dx) > abs(dy):
            self._menu_hover = "RIGHT" if dx > 0 else "LEFT"
        else:
            self._menu_hover = "BOTTOM" if dy > 0 else "TOP"

        # Pinch rising edge to "click"
        click_thr = 0.78
        prev0 = self._pinch_prev[0]
        now0 = float(pinch[0])

        if self._menu_cooldown <= 0.0 and prev0 < click_thr and now0 >= click_thr:
            if self._menu_hover == "TOP":
                self.tool_mode = self.TOOL_DRAW
            elif self._menu_hover == "RIGHT":
                self.tool_mode = self.TOOL_EXTRUDE
            elif self._menu_hover == "BOTTOM":
                self.clear_obstacles()
                self.clear_dye()
            elif self._menu_hover == "LEFT":
                self.render_mode = (self.render_mode + 1) % 4

            self._menu_cooldown = 0.22  # debounce

        self._pinch_prev[0] = now0
        self._pinch_prev[1] = float(pinch[1])

    # ===================== Rendering =====================

    def render_on_frame(self, frame_bgr):
        if frame_bgr is None:
            return None
        h, w = frame_bgr.shape[:2]

        # Select field
        if self.render_mode == 0:
            field = np.clip(self.dye.to_numpy().T, 0.0, 1.0)
            img = (field * 255.0).astype(np.uint8)
        elif self.render_mode == 1:
            field = self.vort.to_numpy().T
            m = float(np.max(np.abs(field)) + 1e-6)
            img = np.clip(128.0 + 92.0 * (field / m), 0, 255).astype(np.uint8)
        elif self.render_mode == 2:
            field = self.p.to_numpy().T
            m = float(np.max(np.abs(field)) + 1e-6)
            img = np.clip(128.0 + 92.0 * (field / m), 0, 255).astype(np.uint8)
        else:
            field = self.speed.to_numpy().T
            m = float(np.max(field) + 1e-6)
            img = np.clip(255.0 * (field / m), 0, 255).astype(np.uint8)

        up = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (self.render_blur_ksize, self.render_blur_ksize), 0)

        colored = cv2.applyColorMap(up, self.colormap)

        # soften saturation (blend with grayscale)
        gray3 = cv2.cvtColor(up, cv2.COLOR_GRAY2BGR)
        colored = cv2.addWeighted(colored, float(self.cmap_saturation), gray3, float(1.0 - self.cmap_saturation), 0)

        out = cv2.addWeighted(frame_bgr, 1.0, colored, float(self.render_alpha), 0.0)

        # --- neon “Tony” stroke render for obstacles ---
        obst = self.obst.to_numpy().T.astype(np.uint8) * 255
        obst_up = cv2.resize(obst, (w, h), interpolation=cv2.INTER_NEAREST)

        edges = cv2.Canny(obst_up, 20, 80)
        glow = cv2.GaussianBlur(edges, (0, 0), 1.6)
        glow2 = cv2.GaussianBlur(edges, (0, 0), 4.0)

        neon = np.zeros_like(out, dtype=np.uint8)
        neon[:, :, 0] = glow2
        neon[:, :, 1] = glow2
        neon[:, :, 2] = glow

        out = cv2.addWeighted(out, 1.0, neon, 0.55, 0.0)
        core_mask = edges > 0
        out[core_mask] = (235, 245, 255)

        # --- midpoint ring UI overlay (Tony menu) ---
        try:
            hp = self.hand_present.to_numpy()
            if hp[0] == 1 and hp[1] == 1:
                pos = self.hand_pos_uv.to_numpy()
                mid = ((pos[0] + pos[1]) * 0.5)
                mx, my = int(mid[0] * w), int(mid[1] * h)

                # rings
                cv2.circle(out, (mx, my), 44, (220, 245, 255), 2, cv2.LINE_AA)
                cv2.circle(out, (mx, my), 22, (220, 245, 255), 1, cv2.LINE_AA)

                # hover highlight
                if self._menu_hover:
                    hx, hy = mx, my
                    if self._menu_hover == "TOP":
                        hy -= 34
                    elif self._menu_hover == "BOTTOM":
                        hy += 34
                    elif self._menu_hover == "LEFT":
                        hx -= 34
                    elif self._menu_hover == "RIGHT":
                        hx += 34
                    cv2.circle(out, (hx, hy), 10, (235, 255, 255), 2, cv2.LINE_AA)

                tool_name = "DRAW" if self.tool_mode == self.TOOL_DRAW else "EXTRUDE"
                mode_name = ["DYE", "VORT", "PRES", "SPD"][self.render_mode]

                cv2.putText(out, f"{tool_name}", (mx - 44, my - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, f"{mode_name}", (mx - 40, my + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 255, 255), 2, cv2.LINE_AA)

                # quadrant labels (subtle)
                cv2.putText(out, "DRAW",   (mx - 18, my - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(out, "EXTR",   (mx + 32, my + 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(out, "CLEAR",  (mx - 22, my + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 240, 255), 1, cv2.LINE_AA)
                cv2.putText(out, "MODE",   (mx - 62, my + 6),  cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 240, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

        # HUD text (kept, but not screaming)
        vmax = float(np.max(np.abs(self.vort.to_numpy())))
        smax = float(np.max(self.speed.to_numpy()))
        drms = float(np.sqrt(np.mean(self.div_abs.to_numpy() ** 2)))

        cv2.putText(out, f"0 Dye | 1 Vort | 2 Press | 3 Speed   (LEFT quadrant cycles too)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2)
        cv2.putText(out, f"Tool: {'DRAW' if self.tool_mode==0 else 'EXTRUDE'}  (T)   Thickness: {self.obstacle_thickness:.1f}  ([ / ])",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2)
        cv2.putText(out, f"|w|max={vmax:.2f}  |u|max={smax:.2f}  div_rms={drms:.5f}   C clear  R reset  M colormap",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (235, 235, 235), 2)

        frame_bgr[:] = out
        return frame_bgr

    # ===================== Taichi helpers =====================

    @ti.func
    def _clampf(self, x, a, b):
        return ti.max(a, ti.min(b, x))

    @ti.func
    def _idx_clamp(self, i, max_i):
        return ti.max(0, ti.min(max_i, i))

    @ti.func
    def _uv_to_grid(self, uv):
        return ti.Vector([uv[0] * float(self.W - 1), uv[1] * float(self.H - 1)])

    @ti.func
    def _bilerp_vec2(self, f, x, y):
        x = self._clampf(x, 0.0, float(self.W - 1))
        y = self._clampf(y, 0.0, float(self.H - 1))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = self._idx_clamp(x0 + 1, self.W - 1)
        y1 = self._idx_clamp(y0 + 1, self.H - 1)
        sx = x - ti.cast(x0, ti.f32)
        sy = y - ti.cast(y0, ti.f32)

        v00 = f[x0, y0]
        v10 = f[x1, y0]
        v01 = f[x0, y1]
        v11 = f[x1, y1]

        a = v00 * (1.0 - sx) + v10 * sx
        b = v01 * (1.0 - sx) + v11 * sx
        return a * (1.0 - sy) + b * sy

    @ti.func
    def _bilerp_s(self, f, x, y):
        x = self._clampf(x, 0.0, float(self.W - 1))
        y = self._clampf(y, 0.0, float(self.H - 1))
        x0 = ti.cast(ti.floor(x), ti.i32)
        y0 = ti.cast(ti.floor(y), ti.i32)
        x1 = self._idx_clamp(x0 + 1, self.W - 1)
        y1 = self._idx_clamp(y0 + 1, self.H - 1)
        sx = x - ti.cast(x0, ti.f32)
        sy = y - ti.cast(y0, ti.f32)

        v00 = f[x0, y0]
        v10 = f[x1, y0]
        v01 = f[x0, y1]
        v11 = f[x1, y1]

        a = v00 * (1.0 - sx) + v10 * sx
        b = v01 * (1.0 - sx) + v11 * sx
        return a * (1.0 - sy) + b * sy

    @ti.func
    def _dist2_point_segment(self, p, a, b):
        ab = b - a
        ap = p - a
        denom = ab.dot(ab) + 1e-6
        t = ap.dot(ab) / denom
        t = self._clampf(t, 0.0, 1.0)
        c = a + ab * t
        d = p - c
        return d.dot(d)

    # ===================== Clear kernels =====================

    @ti.kernel
    def _clear_all_kernel(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.vel[i, j] = ti.Vector([0.0, 0.0])
            self.vel0[i, j] = ti.Vector([0.0, 0.0])
            self.vel1[i, j] = ti.Vector([0.0, 0.0])
            self.dye[i, j] = 0.0
            self.dye0[i, j] = 0.0
            self.div[i, j] = 0.0
            self.p[i, j] = 0.0
            self.p0[i, j] = 0.0
            self.obst[i, j] = 0
            self.vort[i, j] = 0.0
            self.speed[i, j] = 0.0
            self.div_abs[i, j] = 0.0

        for k in ti.static(range(2)):
            self.hand_pos_uv[k] = ti.Vector([0.5, 0.5])
            self.hand_vel_uv[k] = ti.Vector([0.0, 0.0])
            self.hand_pinch[k] = 0.0
            self.hand_present[k] = 0
            self.draw_prev[k] = ti.Vector([0.0, 0.0])
            self.draw_has_prev[k] = 0

    @ti.kernel
    def _clear_obstacles_kernel(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.obst[i, j] = 0

    @ti.kernel
    def _clear_dye_kernel(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.dye[i, j] = 0.0

    # ===================== Inputs / tools kernel =====================

    @ti.kernel
    def _apply_inputs(self, dt: ti.f32, tool_mode: ti.i32, thickness: ti.f32):
        # per-hand: either draw strokes OR inject forces
        for hid in ti.static(range(2)):
            if self.hand_present[hid] == 0:
                self.draw_has_prev[hid] = 0
            else:
                p_uv = self.hand_pos_uv[hid]
                v_uv = self.hand_vel_uv[hid]
                pinch = self.hand_pinch[hid]

                p = self._uv_to_grid(p_uv)
                v = ti.Vector([v_uv[0] * float(self.W), v_uv[1] * float(self.H)]) * ti.cast(self.input_vel_scale, ti.f32)

                closed = self._clampf(pinch, 0.0, 1.0)
                open_amt = 1.0 - closed

                # ---------------- Decide mode (NO continue) ----------------
                draw_cond = (tool_mode == ti.cast(self.TOOL_DRAW, ti.i32)) and (closed >= ti.cast(self.draw_obstacle_when_pinch, ti.f32))

                # ---------------- Thin wall drawing (stroke capsule) ----------------
                if draw_cond:
                    a = p
                    if self.draw_has_prev[hid] == 1:
                        a = self.draw_prev[hid]
                    b = p

                    thick = ti.cast(thickness, ti.f32)
                    thick2 = thick * thick

                    minx = ti.cast(ti.floor(ti.min(a[0], b[0]) - thick - 1.0), ti.i32)
                    maxx = ti.cast(ti.ceil(ti.max(a[0], b[0]) + thick + 1.0), ti.i32)
                    miny = ti.cast(ti.floor(ti.min(a[1], b[1]) - thick - 1.0), ti.i32)
                    maxy = ti.cast(ti.ceil(ti.max(a[1], b[1]) + thick + 1.0), ti.i32)

                    x0 = ti.max(0, minx)
                    x1 = ti.min(self.W - 1, maxx)
                    y0 = ti.max(0, miny)
                    y1 = ti.min(self.H - 1, maxy)

                    for i, j in ti.ndrange((x0, x1 + 1), (y0, y1 + 1)):
                        q = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
                        d2 = self._dist2_point_segment(q, a, b)
                        if d2 <= thick2:
                            self.obst[i, j] = 1
                            self.dye[i, j] = 0.0
                            self.vel[i, j] = ti.Vector([0.0, 0.0])

                    self.draw_prev[hid] = b
                    self.draw_has_prev[hid] = 1

                # ---------------- Fluid force injection ----------------
                else:
                    self.draw_has_prev[hid] = 0

                    r = ti.cast(self.force_radius, ti.i32)
                    cx = ti.cast(p[0], ti.i32)
                    cy = ti.cast(p[1], ti.i32)
                    x0 = ti.max(0, cx - r)
                    x1 = ti.min(self.W - 1, cx + r)
                    y0 = ti.max(0, cy - r)
                    y1 = ti.min(self.H - 1, cy + r)

                    for i, j in ti.ndrange((x0, x1 + 1), (y0, y1 + 1)):
                        if self.obst[i, j] == 1:
                            continue

                        dx = ti.cast(i, ti.f32) - p[0]
                        dy = ti.cast(j, ti.f32) - p[1]
                        r2 = dx * dx + dy * dy
                        if r2 > (self.force_radius * self.force_radius):
                            continue

                        w = ti.exp(-r2 / (self.force_sigma2 + 1e-6))

                        push = v * (ti.cast(self.push_gain, ti.f32) * w)

                        cross = dx * v[1] - dy * v[0]
                        perp = ti.Vector([-dy, dx])
                        perp_norm = perp / (ti.sqrt(r2) + 1e-4)
                        swirl = perp_norm * (ti.cast(self.swirl_gain, ti.f32) * w * cross / (r2 + 10.0))

                        dir_in = ti.Vector([-dx, -dy]) / (ti.sqrt(r2) + 1e-4)
                        pinch_force = dir_in * (ti.cast(self.pinch_gain, ti.f32) * w * (closed - 0.25 * open_amt))

                        dv = (push + swirl + pinch_force) * dt
                        ti.atomic_add(self.vel[i, j][0], dv[0])
                        ti.atomic_add(self.vel[i, j][1], dv[1])
                        ti.atomic_add(self.dye[i, j], ti.cast(self.dye_inject, ti.f32) * w * dt)

        # ---------------- Two-pointer interaction / EXTRUDE sheet ----------------
        if self.hand_present[0] == 1 and self.hand_present[1] == 1:
            pinch0 = self.hand_pinch[0]
            pinch1 = self.hand_pinch[1]

            p0 = self._uv_to_grid(self.hand_pos_uv[0])
            p1 = self._uv_to_grid(self.hand_pos_uv[1])

            v0 = ti.Vector([self.hand_vel_uv[0][0] * float(self.W), self.hand_vel_uv[0][1] * float(self.H)]) * ti.cast(self.input_vel_scale, ti.f32)
            v1 = ti.Vector([self.hand_vel_uv[1][0] * float(self.W), self.hand_vel_uv[1][1] * float(self.H)]) * ti.cast(self.input_vel_scale, ti.f32)

            d = p1 - p0
            L = ti.sqrt(d.dot(d)) + 1e-4
            dhat = d / L
            mid = (p0 + p1) * 0.5
            stretch_rate = (v1 - v0).dot(dhat)

            n = ti.Vector([-dhat[1], dhat[0]])

            r = ti.cast(self.pair_radius, ti.i32)
            cx = ti.cast(mid[0], ti.i32)
            cy = ti.cast(mid[1], ti.i32)
            x0 = ti.max(0, cx - r)
            x1 = ti.min(self.W - 1, cx + r)
            y0 = ti.max(0, cy - r)
            y1 = ti.min(self.H - 1, cy + r)

            extrude_active = 0
            if tool_mode == ti.cast(self.TOOL_EXTRUDE, ti.i32):
                if pinch0 >= ti.cast(self.extrude_pinch_min, ti.f32) and pinch1 >= ti.cast(self.extrude_pinch_min, ti.f32):
                    extrude_active = 1

            for i, j in ti.ndrange((x0, x1 + 1), (y0, y1 + 1)):
                if self.obst[i, j] == 1:
                    continue

                x = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])

                t = (x - p0).dot(dhat)
                t = self._clampf(t, 0.0, L)
                c = p0 + dhat * t

                s = x - c
                s2 = s.dot(s)
                if s2 > (self.pair_radius * self.pair_radius):
                    continue

                w = ti.exp(-s2 / (ti.cast(self.pair_sigma2, ti.f32) + 1e-6))

                grab_dir = -s / (ti.sqrt(s2) + 1e-4)
                grab = grab_dir * (ti.cast(self.pair_grab_gain, ti.f32) * w)

                along = (x - mid).dot(dhat)
                stretch = dhat * (ti.cast(self.pair_stretch_gain, ti.f32) * w * along * stretch_rate / (L + 1.0))

                dv = (grab + stretch) * dt

                if extrude_active == 1:
                    side = (x - mid).dot(n)
                    sgn = 1.0
                    if side < 0.0:
                        sgn = -1.0
                    extr = n * (ti.cast(self.extrude_gain, ti.f32) * w * stretch_rate * sgn)
                    dv += extr * dt
                    ti.atomic_add(self.dye[i, j], ti.cast(self.extrude_dye, ti.f32) * w * ti.abs(stretch_rate) * dt)

                ti.atomic_add(self.vel[i, j][0], dv[0])
                ti.atomic_add(self.vel[i, j][1], dv[1])

    @ti.kernel
    def _apply_obstacles(self):
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])
                self.dye[i, j] = 0.0
    @ti.kernel
    def _clamp_velocity(self):
        vmax = ti.cast(self.vel_limit, ti.f32)
        for i, j in ti.ndrange(self.W, self.H):
            v = self.vel[i, j]
            s = ti.sqrt(v.dot(v)) + 1e-6
            if s > vmax:
                self.vel[i, j] = v * (vmax / s)

    # ===================== Viscosity =====================

    def _diffuse_velocity(self, dt):
        a = self.viscosity * dt
        self._copy_vec2(self.vel, self.vel0)
        src = self.vel0
        dst = self.vel1
        for _ in range(self.diffuse_iters):
            self._diffuse_vec2(src, dst, a)
            src, dst = dst, src
        self._copy_vec2(src, self.vel)

    @ti.kernel
    def _copy_vec2(self, src: ti.template(), dst: ti.template()):
        for i, j in ti.ndrange(self.W, self.H):
            dst[i, j] = src[i, j]

    @ti.kernel
    def _diffuse_vec2(self, src: ti.template(), dst: ti.template(), a: ti.f32):
        denom = 1.0 + 4.0 * a
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                dst[i, j] = ti.Vector([0.0, 0.0])
                continue

            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.W - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.H - 1)

            v0 = self.vel[i, j]
            s = src[im, j] + src[ip, j] + src[i, jm] + src[i, jp]
            dst[i, j] = (v0 + a * s) / denom

    # ===================== Projection =====================

    def _project(self):
        self._compute_divergence()
        self._clear_pressure()
        for _ in range(self.project_iters):
            self._pressure_jacobi()
        self._subtract_pressure_grad()

    @ti.kernel
    def _compute_divergence(self):
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.div[i, j] = 0.0
                continue

            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.W - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.H - 1)

            vx_r = self.vel[ip, j][0]
            vx_l = self.vel[im, j][0]
            vy_u = self.vel[i, jp][1]
            vy_d = self.vel[i, jm][1]
            self.div[i, j] = 0.5 * ((vx_r - vx_l) + (vy_u - vy_d))

    @ti.kernel
    def _clear_pressure(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.p[i, j] = 0.0

    @ti.kernel
    def _pressure_jacobi(self):
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.p0[i, j] = 0.0
                continue

            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.W - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.H - 1)

            self.p0[i, j] = (self.div[i, j] + self.p[im, j] + self.p[ip, j] + self.p[i, jm] + self.p[i, jp]) * 0.25

        for i, j in ti.ndrange(self.W, self.H):
            self.p[i, j] = self.p0[i, j]

    @ti.kernel
    def _subtract_pressure_grad(self):
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])
                continue

            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.W - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.H - 1)

            gradx = 0.5 * (self.p[ip, j] - self.p[im, j])
            grady = 0.5 * (self.p[i, jp] - self.p[i, jm])
            self.vel[i, j] -= ti.Vector([gradx, grady])

    # ===================== Advection =====================

    @ti.kernel
    def _advect_velocity(self, dt: ti.f32):
        for i, j in ti.ndrange(self.W, self.H):
            self.vel0[i, j] = self.vel[i, j]

        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])
                continue

            x = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            v = self.vel0[i, j] * ti.cast(self.advect_strength, ti.f32)
            back = x - v * dt
            self.vel[i, j] = self._bilerp_vec2(self.vel0, back[0], back[1])

    @ti.kernel
    def _advect_dye(self, dt: ti.f32):
        for i, j in ti.ndrange(self.W, self.H):
            self.dye0[i, j] = self.dye[i, j]

        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.dye[i, j] = 0.0
                continue

            x = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            v = self.vel[i, j] * ti.cast(self.advect_strength, ti.f32)
            back = x - v * dt
            self.dye[i, j] = self._bilerp_s(self.dye0, back[0], back[1])

    @ti.kernel
    def _decay_dye(self):
        dec = ti.cast(self.dye_decay, ti.f32)
        for i, j in ti.ndrange(self.W, self.H):
            self.dye[i, j] *= dec
            if self.dye[i, j] < 1e-6:
                self.dye[i, j] = 0.0

    # ===================== Diagnostics =====================

    @ti.kernel
    def _compute_diagnostics(self):
        for i, j in ti.ndrange(self.W, self.H):
            im = ti.max(i - 1, 0)
            ip = ti.min(i + 1, self.W - 1)
            jm = ti.max(j - 1, 0)
            jp = ti.min(j + 1, self.H - 1)

            vx_r = self.vel[ip, j][0]
            vx_l = self.vel[im, j][0]
            vy_u = self.vel[i, jp][1]
            vy_d = self.vel[i, jm][1]

            div = 0.5 * ((vx_r - vx_l) + (vy_u - vy_d))
            self.div_abs[i, j] = ti.abs(div)

            dvy_dx = 0.5 * (self.vel[ip, j][1] - self.vel[im, j][1])
            dvx_dy = 0.5 * (self.vel[i, jp][0] - self.vel[i, jm][0])
            self.vort[i, j] = dvy_dx - dvx_dy

            v = self.vel[i, j]
            self.speed[i, j] = ti.sqrt(v.dot(v))
