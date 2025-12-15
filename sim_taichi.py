import taichi as ti
import taichi.math as tm
import numpy as np
# pyright: reportInvalidTypeForm=false


def _pget(p, key, default=None):
    # Works for dicts and objects (dataclass / SimpleNamespace)
    if isinstance(p, dict):
        return p.get(key, default)
    return getattr(p, key, default)


@ti.data_oriented
class ParticleSimTaichi:
    """
    Efficient Taichi particle sim (Taichi 1.7.4 friendly).

    app.py expects:
      sim = ParticleSimTaichi(params=params)
      sim.step()
      sim.set_hand_input((x,y), (vx,vy), dt, hand_present=bool)
      sim.get_positions()   # Nx2 float32 in [0..1]
    """

    def __init__(self, params):
        # NOTE: ti.init() MUST be called in app.py before constructing this class.

        # --- params (dict or Params object) ---
        self.n = int(_pget(params, "n_particles", _pget(params, "n", 4096)))

        dt = float(_pget(params, "dt", 1.0 / 120.0))
        damping = float(_pget(params, "damping", 0.985))
        hand_radius = float(_pget(params, "hand_radius", 0.10))
        hand_strength = float(_pget(params, "hand_strength", 6.0))
        bounce = float(_pget(params, "bounce", 0.6))
        seed = int(_pget(params, "seed", 1234))

        # --- scalar params in 0-d fields (kernel-friendly) ---
        # (shape=() is fine for scalar ti.field; your earlier crash was Vector.field(shape=()))
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

        # --- hand input (avoid shape=()) ---
        self.hand_pos = ti.Vector.field(2, dtype=ti.f32, shape=(1,))
        self.hand_vel = ti.Vector.field(2, dtype=ti.f32, shape=(1,))
        self.hand_active = ti.field(dtype=ti.i32, shape=(1,))

        self._init_particles(seed)

    # -------------------- init --------------------

    @ti.kernel
    def _init_particles(self, seed: ti.i32):
        for i in range(self.n):
            a = ti.sin(ti.cast(i * 97 + seed, ti.f32)) * 43758.5453
            b = ti.sin(ti.cast(i * 57 + seed * 3, ti.f32)) * 24634.6345
            x = tm.fract(a)
            y = tm.fract(b)
            self.pos[i] = ti.Vector([x, y])
            self.vel[i] = ti.Vector([0.0, 0.0])

        self.hand_pos[0] = ti.Vector([0.5, 0.5])
        self.hand_vel[0] = ti.Vector([0.0, 0.0])
        self.hand_active[0] = 0

    # -------------------- hand control --------------------

    @ti.kernel
    def _set_hand_kernel(self, x: ti.f32, y: ti.f32, vx: ti.f32, vy: ti.f32, active: ti.i32):
        # Clamp position to [0..1] for stability
        self.hand_pos[0] = ti.Vector([ti.min(1.0, ti.max(0.0, x)), ti.min(1.0, ti.max(0.0, y))])
        self.hand_vel[0] = ti.Vector([vx, vy])
        self.hand_active[0] = active

    def set_hand(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0, active: int = 1):
        """Direct API (normalized coords)."""
        self._set_hand_kernel(float(x), float(y), float(vx), float(vy), int(active))

    def set_hand_input(self, pos, vel, dt_unused=0.0, hand_present=True):
        """
        Compatibility wrapper for your app.py call:
          sim.set_hand_input((x,y), (vx,vy), dt, hand_present=False/True)
        """
        x, y = float(pos[0]), float(pos[1])
        vx, vy = float(vel[0]), float(vel[1])
        active = 1 if hand_present else 0
        self.set_hand(x, y, vx, vy, active)

    # -------------------- sim step --------------------

    @ti.kernel
    def step(self, dt_in: ti.f32):
        hp = self.hand_pos[0]
        hv = self.hand_vel[0]
        ha = self.hand_active[0]

        # use dt from app.py
        dt = dt_in

        damp = self.damping[None]
        r = self.hand_radius[None]
        strength = self.hand_strength[None]
        bounce = self.bounce[None]

        for i in range(self.n):
            p = self.pos[i]
            v = self.vel[i]

            if ha != 0:
                d = p - hp
                dist2 = d.dot(d) + 1e-6
                dist = ti.sqrt(dist2)

                if dist < r:
                    dir = d / dist
                    t = 1.0 - (dist / r)
                    v += dir * (strength * t) * dt
                    v += hv * (0.25 * t) * dt

            v *= damp
            p += v * dt

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
        """Returns Nx2 float32 array in [0..1]."""
        return self.pos.to_numpy()

    def set_positions(self, positions: np.ndarray):
        """Optional: set positions from Nx2 numpy array."""
        arr = np.asarray(positions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("positions must be Nx2")
        if arr.shape[0] != self.n:
            raise ValueError(f"positions must have N={self.n} rows")

        arr = np.clip(arr, 0.0, 1.0)
        self.pos.from_numpy(arr)
        self.vel.from_numpy(np.zeros((self.n, 2), dtype=np.float32))
    def render_on_frame(self, frame):
        """
    Draw particles onto an HxWx3 uint8 frame (OpenCV BGR).
    Modifies frame in-place and also returns it.
        """
        pos = self.get_positions()  # Nx2 in [0..1]
        h, w = frame.shape[:2]
        if pos.size == 0:
            return frame

        # Convert normalized -> pixel coords
        xs = (pos[:, 0] * (w - 1)).astype(np.int32)
        ys = (pos[:, 1] * (h - 1)).astype(np.int32)

        # Clip to bounds
        m = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs = xs[m]
        ys = ys[m]

        # Draw as bright white pixels (BGR)
        frame[ys, xs] = (255, 255, 255)

        # Optional: make points a tiny bit bigger (2x2)
        x2 = np.clip(xs + 1, 0, w - 1)
        y2 = np.clip(ys + 1, 0, h - 1)
        frame[ys, x2] = (255, 255, 255)
        frame[y2, xs] = (255, 255, 255)
        frame[y2, x2] = (255, 255, 255)

        return frame
