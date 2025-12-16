import taichi as ti
import numpy as np
# pyright: reportInvalidTypeForm=false


def _pget(p, key, default=None):
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


@ti.data_oriented
class ParticleSimTaichi:
    """
    Efficient Taichi particle sim (Taichi 1.7.4 friendly).

    app.py expects:
      sim = ParticleSimTaichi(params=params)
      sim.step(dt)
      sim.set_hand_input((x,y), (vx,vy), pinch, hand_present=bool, hand_id=0/1)
      sim.get_positions()  # Nx2 float32 in [0..1]
      sim.render_on_frame(frame)
    """

    def __init__(self, params):
        # --- params ---
        self.n = int(_pget(params, "n_particles", _pget(params, "n", 4096)))

        dt = float(_pget(params, "dt", 1.0 / 120.0))
        damping = float(_pget(params, "damping", 0.985))
        hand_radius = float(_pget(params, "hand_radius", 0.10))
        hand_strength = float(_pget(params, "hand_strength", 6.0))
        bounce = float(_pget(params, "bounce", 0.6))
        seed = int(_pget(params, "seed", 1234))

        # --- scalar params ---
        self.dt = ti.field(dtype=ti.f32, shape=())
        self.damping = ti.field(dtype=ti.f32, shape=())
        self.hand_radius = ti.field(dtype=ti.f32, shape=())
        self.hand_strength = ti.field(dtype=ti.f32, shape=())
        self.bounce = ti.field(dtype=ti.f32, shape=())

        self.dt[None] = dt
        self.damping[None] = damping
        self.hand_radius[None] = hand_radius
        self.hand_strength[None] = hand_strength
        self.bounce[None] = bounce

        # --- particle state ---
        self.pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.n,))
        self.vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.n,))

        # --- 2 pointers / 2 hands ---
        self.max_hands = 2
        self.hand_pos = ti.Vector.field(2, dtype=ti.f32, shape=(self.max_hands,))
        self.hand_vel = ti.Vector.field(2, dtype=ti.f32, shape=(self.max_hands,))
        self.hand_active = ti.field(dtype=ti.i32, shape=(self.max_hands,))
        self.hand_pinch = ti.field(dtype=ti.f32, shape=(self.max_hands,))

        self._init_particles(seed)

        # render buffer (allocated lazily)
        self._dens = None
        self._dens_shape = None

    @ti.func
    def _fract(self, x):
        # fract(x) = x - floor(x), works for negatives too
        return x - ti.floor(x)

    @ti.kernel
    def _init_particles(self, seed: ti.i32):
        for i in range(self.n):
            a = ti.sin(ti.cast(i * 97 + seed, ti.f32)) * 43758.5453
            b = ti.sin(ti.cast(i * 57 + seed * 3, ti.f32)) * 24634.6345
            x = self._fract(a)
            y = self._fract(b)
            self.pos[i] = ti.Vector([x, y])
            self.vel[i] = ti.Vector([0.0, 0.0])

        for h in ti.static(range(2)):
            self.hand_pos[h] = ti.Vector([0.5, 0.5])
            self.hand_vel[h] = ti.Vector([0.0, 0.0])
            self.hand_active[h] = 0
            self.hand_pinch[h] = 0.0

    # -------------------- hand control --------------------

    @ti.kernel
    def _set_hand_kernel(self, hid: ti.i32, x: ti.f32, y: ti.f32,
                         vx: ti.f32, vy: ti.f32, active: ti.i32, pinch: ti.f32):
        if 0 <= hid < 2:
            # clamp pos to [0..1]
            cx = ti.min(1.0, ti.max(0.0, x))
            cy = ti.min(1.0, ti.max(0.0, y))
            self.hand_pos[hid] = ti.Vector([cx, cy])
            self.hand_vel[hid] = ti.Vector([vx, vy])
            self.hand_active[hid] = active
            self.hand_pinch[hid] = ti.min(1.0, ti.max(0.0, pinch))

    def set_hand_input(self, pos, vel, pinch=0.0, hand_present=True, hand_id=0):
        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        p = float(pinch)
        active = 1 if hand_present else 0
        self._set_hand_kernel(int(hand_id), x, y, vx, vy, active, p)

    # -------------------- sim step --------------------

    @ti.kernel
    def step(self, dt_in: ti.f32):
        dt = dt_in
        damp = self.damping[None]
        base_r = self.hand_radius[None]
        base_strength = self.hand_strength[None]
        bounce = self.bounce[None]

        for i in range(self.n):
            p = self.pos[i]
            v = self.vel[i]

            # apply forces from both pointers
            for h in ti.static(range(2)):
                if self.hand_active[h] != 0:
                    hp = self.hand_pos[h]
                    hv = self.hand_vel[h]
                    pinch = self.hand_pinch[h]

                    # pinch affects radius + strength (feels more “grabby” when pinched)
                    r = base_r * (0.75 + 0.75 * pinch)
                    strength = base_strength * (0.50 + 1.50 * pinch)

                    d = p - hp
                    dist2 = d.dot(d) + 1e-6
                    dist = ti.sqrt(dist2)

                    if dist < r:
                        t = 1.0 - (dist / r)
                        dir = d / dist
                        v += dir * (strength * t) * dt
                        v += hv * (0.10 + 0.40 * pinch) * t * dt

            v *= damp
            p += v * dt

            # bounds
            if p.x < 0.0:
                p.x = 0.0
                v.x = -v.x * bounce
            elif p.x > 1.0:
                p.x = 1.0
                v.x = -v.x * bounce

            if p.y < 0.0:
                p.y = 0.0
                v.y = -v.y * bounce
            elif p.y > 1.0:
                p.y = 1.0
                v.y = -v.y * bounce

            self.pos[i] = p
            self.vel[i] = v

    # -------------------- data access --------------------

    def get_positions(self) -> np.ndarray:
        return self.pos.to_numpy()

    # -------------------- rendering (smoother, less pixel-noise) --------------------

    def render_on_frame(self, frame):
        import cv2

        pos = self.get_positions()  # Nx2 in [0..1]
        h, w = frame.shape[:2]

        # Downsample density grid to reduce pixel/noise look
        ds = 2  # 2 = half-res grid
        gh = max(1, h // ds)
        gw = max(1, w // ds)

        if self._dens is None or self._dens_shape != (gh, gw):
            self._dens = np.zeros((gh, gw), dtype=np.float32)
            self._dens_shape = (gh, gw)

        self._dens.fill(0.0)

        xs = (pos[:, 0] * (gw - 1)).astype(np.int32)
        ys = (pos[:, 1] * (gh - 1)).astype(np.int32)

        m = (xs >= 0) & (xs < gw) & (ys >= 0) & (ys < gh)
        xs = xs[m]
        ys = ys[m]

        # accumulate density
        np.add.at(self._dens, (ys, xs), 1.0)

        # blur more => smoother “fluid”
        dens = cv2.GaussianBlur(self._dens, (0, 0), sigmaX=3.5, sigmaY=3.5)
        dens = cv2.GaussianBlur(dens, (0, 0), sigmaX=2.0, sigmaY=2.0)

        mx = float(dens.max()) if dens.size else 1.0
        if mx < 1e-6:
            return frame

        img_small = (dens / mx * 255.0).astype(np.uint8)
        col_small = cv2.applyColorMap(img_small, cv2.COLORMAP_TURBO)

        # upsample smoothly to full frame
        col = cv2.resize(col_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # blend onto camera
        out = cv2.addWeighted(frame, 0.55, col, 0.45, 0.0)
        frame[:] = out
        return frame
