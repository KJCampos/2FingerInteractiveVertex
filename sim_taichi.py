# pyright: reportInvalidTypeForm=false
import numpy as np
import taichi as ti
import cv2


def _pget(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


@ti.data_oriented
class ParticleSimTaichi:
    """
    Stable incompressible 2D fluid (grid-based) driven by up to 2 fingertip pointers.

    Keeps your API:
      sim = ParticleSimTaichi(params=params)
      sim.set_hand_input(pos_uv, vel_uv, pinch, hand_present=True/False, hand_id=0/1)
      sim.step(dt)
      sim.render_on_frame(frame)   # modifies frame in-place and returns it

    Extras:
      sim.handle_key(key)          # toggle render modes + clear/reset
    """

    def __init__(self, params):
        self.params = params

        # ---------------- Grid resolution ----------------
        self.W = int(_pget(params, "grid_w", 240))
        self.H = int(_pget(params, "grid_h", 135))

        # ---------------- Core sim params ----------------
        self.dt_max = float(_pget(params, "dt_max", 1.0 / 30.0))
        self.advect_strength = float(_pget(params, "advect_strength", 1.0))

        # viscosity / diffusion
        self.viscosity = float(_pget(params, "viscosity", 0.0025))
        self.diffuse_iters = int(_pget(params, "diffuse_iters", 12))
        self.project_iters = int(_pget(params, "project_iters", 24))

        # dye
        self.dye_decay = float(_pget(params, "dye_decay", 0.985))
        self.dye_inject = float(_pget(params, "dye_inject", 1.0))

        # pointer force shape (in grid cells)
        self.force_radius = float(_pget(params, "force_radius", 12.0))
        self.force_sigma = float(_pget(params, "force_sigma", self.force_radius * 0.55))
        self.force_sigma2 = self.force_sigma * self.force_sigma

        # force gains
        self.push_gain = float(_pget(params, "push_gain", 1.25))
        self.swirl_gain = float(_pget(params, "swirl_gain", 2.0))
        self.pinch_gain = float(_pget(params, "pinch_gain", 1.15))

        # pair grab/stretch
        self.pair_grab_gain = float(_pget(params, "pair_grab_gain", 1.35))
        self.pair_stretch_gain = float(_pget(params, "pair_stretch_gain", 0.80))
        self.pair_radius = float(_pget(params, "pair_radius", 10.0))
        self.pair_sigma2 = (self.pair_radius * 0.8) ** 2

        # ---------------- “Shape drawing” via obstacles ----------------
        # Your app pinch is 0..1 where 1 = closed (you already do that)
        self.draw_obstacle_when_pinch = float(_pget(params, "draw_obstacle_when_pinch", 0.80))
        self.obstacle_radius = float(_pget(params, "obstacle_radius", 6.0))  # in grid cells

        # ---------------- Rendering + modes ----------------
        # 0 = dye, 1 = vorticity, 2 = pressure, 3 = speed
        self.render_mode = int(_pget(params, "render_mode", 0))

        self.render_blur_ksize = int(_pget(params, "render_blur_ksize", 17))
        if self.render_blur_ksize % 2 == 0:
            self.render_blur_ksize += 1
        self.render_alpha = float(_pget(params, "render_alpha", 0.65))
        self.colormap = int(_pget(params, "colormap", cv2.COLORMAP_TURBO))

        # ---------------- Taichi fields ----------------
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))
        self.vel0 = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))
        self.vel1 = ti.Vector.field(2, dtype=ti.f32, shape=(self.W, self.H))

        self.dye = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.dye0 = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        self.div = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.p = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.p0 = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        # Obstacles: 1 = solid (no flow)
        self.obst = ti.field(dtype=ti.i32, shape=(self.W, self.H))

        # Diagnostics
        self.vort = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.speed = ti.field(dtype=ti.f32, shape=(self.W, self.H))
        self.div_abs = ti.field(dtype=ti.f32, shape=(self.W, self.H))

        # Hand inputs (2 pointers)
        self.hand_pos_uv = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
        self.hand_vel_uv = ti.Vector.field(2, dtype=ti.f32, shape=(2,))
        self.hand_pinch = ti.field(dtype=ti.f32, shape=(2,))   # 0..1 (1 closed)
        self.hand_present = ti.field(dtype=ti.i32, shape=(2,))

        self.reset()

    # ========================= Public controls =========================

    def handle_key(self, key: int):
        """
        Call this from app.py with cv2.waitKey.
        Keys:
          0/1/2/3: render modes (dye/vorticity/pressure/speed)
          c: clear obstacles
          r: reset sim
        """
        if key is None:
            return

        if key in (ord('0'), ord('1'), ord('2'), ord('3')):
            self.render_mode = int(chr(key))
        elif key in (ord('c'), ord('C')):
            self.clear_obstacles()
        elif key in (ord('r'), ord('R')):
            self.reset()

    def clear_obstacles(self):
        self._clear_obstacles_kernel()

    def reset(self):
        self._clear_all_kernel()

    def set_hand_input(self, pos_uv, vel_uv, pinch, hand_present=True, hand_id=0):
        hid = int(hand_id)
        if hid < 0 or hid > 1:
            return
        self.hand_pos_uv[hid] = ti.Vector([float(pos_uv[0]), float(pos_uv[1])])
        self.hand_vel_uv[hid] = ti.Vector([float(vel_uv[0]), float(vel_uv[1])])
        self.hand_pinch[hid] = float(max(0.0, min(1.0, pinch)))  # 1=closed
        self.hand_present[hid] = 1 if hand_present else 0

    def get_positions(self) -> np.ndarray:
        # Kept for compatibility with older codepaths.
        return np.zeros((0, 2), dtype=np.float32)

    # ========================= Core sim loop =========================

    def step(self, dt):
        dt = float(dt)
        if dt <= 0:
            return
        dt = min(dt, self.dt_max)

        # 1) Inject from fingers (or draw obstacles)
        self._apply_inputs(dt)
        self._apply_obstacles()

        # 2) Diffuse (viscosity)
        if self.viscosity > 0.0 and self.diffuse_iters > 0:
            self._diffuse_velocity(dt)
            self._apply_obstacles()

        # 3) Project (incompressible)
        self._project()
        self._apply_obstacles()

        # 4) Advect velocity
        self._advect_velocity(dt)
        self._apply_obstacles()

        # 5) Project again
        self._project()
        self._apply_obstacles()

        # 6) Advect dye
        self._advect_dye(dt)
        self._apply_obstacles()

        # 7) Dye decay + diagnostics
        self._decay_dye()
        self._compute_diagnostics()

    # ========================= Rendering =========================

    def render_on_frame(self, frame_bgr):
        if frame_bgr is None:
            return None

        h, w = frame_bgr.shape[:2]

        # Pick field for display
        if self.render_mode == 0:
            field = np.clip(self.dye.to_numpy().T, 0.0, 1.0)
            img = (field * 255.0).astype(np.uint8)

        elif self.render_mode == 1:
            field = self.vort.to_numpy().T
            m = float(np.max(np.abs(field)) + 1e-6)
            img = np.clip(128.0 + 127.0 * (field / m), 0, 255).astype(np.uint8)

        elif self.render_mode == 2:
            field = self.p.to_numpy().T
            m = float(np.max(np.abs(field)) + 1e-6)
            img = np.clip(128.0 + 127.0 * (field / m), 0, 255).astype(np.uint8)

        else:  # 3 speed
            field = self.speed.to_numpy().T
            m = float(np.max(field) + 1e-6)
            img = np.clip(255.0 * (field / m), 0, 255).astype(np.uint8)

        # Upscale to camera resolution
        up = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        # Blur to avoid pixel noise
        up = cv2.GaussianBlur(up, (self.render_blur_ksize, self.render_blur_ksize), 0)

        colored = cv2.applyColorMap(up, self.colormap)
        out = cv2.addWeighted(frame_bgr, 1.0, colored, float(self.render_alpha), 0.0)

        # Overlay obstacle mask (white outlines)
        obst = self.obst.to_numpy().T.astype(np.uint8) * 255
        obst_up = cv2.resize(obst, (w, h), interpolation=cv2.INTER_NEAREST)
        edges = cv2.Canny(obst_up, 40, 120)
        out[edges > 0] = (255, 255, 255)

        # HUD
        vort_np = self.vort.to_numpy()
        spd_np = self.speed.to_numpy()
        div_np = self.div_abs.to_numpy()

        vmax = float(np.max(np.abs(vort_np)))
        smax = float(np.max(spd_np))
        drms = float(np.sqrt(np.mean(div_np * div_np)))

        cv2.putText(out, f"Mode {self.render_mode}: 0 Dye | 1 Vorticity | 2 Pressure | 3 Speed",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(out, f"|w|max={vmax:.2f}  |u|max={smax:.2f}  div_rms={drms:.5f}",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(out, f"Pinch >= {self.draw_obstacle_when_pinch:.2f}: DRAW SOLID WALLS  |  Keys: 0-3, C clear, R reset",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

        frame_bgr[:] = out
        return frame_bgr

    # ========================= Taichi helpers =========================

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

    # ========================= Clear kernels =========================

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

        for k in range(2):
            self.hand_pos_uv[k] = ti.Vector([0.5, 0.5])
            self.hand_vel_uv[k] = ti.Vector([0.0, 0.0])
            self.hand_pinch[k] = 0.0
            self.hand_present[k] = 0

    @ti.kernel
    def _clear_obstacles_kernel(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.obst[i, j] = 0

    # ========================= Input forces + obstacle drawing =========================

    @ti.kernel
    def _apply_inputs(self, dt: ti.f32):
        for hid in range(2):
            if self.hand_present[hid] == 0:
                continue

            p_uv = self.hand_pos_uv[hid]
            v_uv = self.hand_vel_uv[hid]
            pinch = self.hand_pinch[hid]  # 0..1 where 1=closed

            p = self._uv_to_grid(p_uv)
            v = ti.Vector([v_uv[0] * float(self.W), v_uv[1] * float(self.H)])

            closed = self._clampf(pinch, 0.0, 1.0)
            open_amt = 1.0 - closed

            # --- Pinch tool: draw SOLID walls/obstacles ---
            if closed >= ti.cast(self.draw_obstacle_when_pinch, ti.f32):
                rr = ti.cast(self.obstacle_radius, ti.i32)
                cx = ti.cast(p[0], ti.i32)
                cy = ti.cast(p[1], ti.i32)
                x0 = ti.max(0, cx - rr)
                x1 = ti.min(self.W - 1, cx + rr)
                y0 = ti.max(0, cy - rr)
                y1 = ti.min(self.H - 1, cy + rr)

                for i, j in ti.ndrange((x0, x1 + 1), (y0, y1 + 1)):
                    dx = ti.cast(i, ti.f32) - p[0]
                    dy = ti.cast(j, ti.f32) - p[1]
                    if dx * dx + dy * dy <= self.obstacle_radius * self.obstacle_radius:
                        self.obst[i, j] = 1
                        self.dye[i, j] = 0.0
                        self.vel[i, j] = ti.Vector([0.0, 0.0])
                continue

            # --- Otherwise: inject forces into fluid ---
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

                # wake / steering (momentum injection)
                push = v * (self.push_gain * w)

                # swirl injection (signed tangential) based on cross(r, v)
                cross = dx * v[1] - dy * v[0]
                perp = ti.Vector([-dy, dx])
                perp_norm = perp / (ti.sqrt(r2) + 1e-4)
                swirl = perp_norm * (self.swirl_gain * w * cross / (r2 + 6.0))

                # pinch compress/expand (feels like squeezing the flow)
                dir_in = ti.Vector([-dx, -dy]) / (ti.sqrt(r2) + 1e-4)
                pinch_force = dir_in * (self.pinch_gain * w * (closed - 0.35 * open_amt))

                dv = (push + swirl + pinch_force) * dt
                ti.atomic_add(self.vel[i, j][0], dv[0])
                ti.atomic_add(self.vel[i, j][1], dv[1])

                # dye injection to visualize motion
                ti.atomic_add(self.dye[i, j], self.dye_inject * w * dt)

        # Pair grab/stretch (only if both hands present AND neither is drawing)
        if (self.hand_present[0] == 1 and self.hand_present[1] == 1 and
            self.hand_pinch[0] < ti.cast(self.draw_obstacle_when_pinch, ti.f32) and
            self.hand_pinch[1] < ti.cast(self.draw_obstacle_when_pinch, ti.f32)):

            p0 = self._uv_to_grid(self.hand_pos_uv[0])
            p1 = self._uv_to_grid(self.hand_pos_uv[1])
            v0 = ti.Vector([self.hand_vel_uv[0][0] * float(self.W), self.hand_vel_uv[0][1] * float(self.H)])
            v1 = ti.Vector([self.hand_vel_uv[1][0] * float(self.W), self.hand_vel_uv[1][1] * float(self.H)])

            d = p1 - p0
            L = ti.sqrt(d.dot(d)) + 1e-4
            dhat = d / L
            mid = (p0 + p1) * 0.5
            stretch_rate = (v1 - v0).dot(dhat)

            r = ti.cast(self.pair_radius, ti.i32)
            cx = ti.cast(mid[0], ti.i32)
            cy = ti.cast(mid[1], ti.i32)
            x0 = ti.max(0, cx - r)
            x1 = ti.min(self.W - 1, cx + r)
            y0 = ti.max(0, cy - r)
            y1 = ti.min(self.H - 1, cy + r)

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

                w = ti.exp(-s2 / (self.pair_sigma2 + 1e-6))

                grab_dir = -s / (ti.sqrt(s2) + 1e-4)
                grab = grab_dir * (self.pair_grab_gain * w)

                along = (x - mid).dot(dhat)
                stretch = dhat * (self.pair_stretch_gain * w * along * stretch_rate / (L + 1.0))

                dv = (grab + stretch) * dt
                ti.atomic_add(self.vel[i, j][0], dv[0])
                ti.atomic_add(self.vel[i, j][1], dv[1])
    @ti.kernel
    def _apply_obstacles(self):
        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])
                self.dye[i, j] = 0.0


    # ========================= Viscosity (diffuse velocity) =========================

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

    # ========================= Projection (incompressible) =========================

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

    # ========================= Advection =========================

    @ti.kernel
    def _advect_velocity(self, dt: ti.f32):
        for i, j in ti.ndrange(self.W, self.H):
            self.vel0[i, j] = self.vel[i, j]

        for i, j in ti.ndrange(self.W, self.H):
            if self.obst[i, j] == 1:
                self.vel[i, j] = ti.Vector([0.0, 0.0])
                continue

            x = ti.Vector([ti.cast(i, ti.f32), ti.cast(j, ti.f32)])
            v = self.vel0[i, j] * self.advect_strength
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
            v = self.vel[i, j] * self.advect_strength
            back = x - v * dt
            self.dye[i, j] = self._bilerp_s(self.dye0, back[0], back[1])

    @ti.kernel
    def _decay_dye(self):
        for i, j in ti.ndrange(self.W, self.H):
            self.dye[i, j] *= self.dye_decay
            if self.dye[i, j] < 1e-6:
                self.dye[i, j] = 0.0

    # ========================= Diagnostics =========================

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
