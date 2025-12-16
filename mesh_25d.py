# physics3d_ti.py
# GPU-ish rigid approximation: one rigid body + plane contact + damping.
# Stable "settles on plane" behavior, not a full rigid-body solver.
# pyright: reportInvalidTypeForm=false

from __future__ import annotations
import taichi as ti

_TAICHI_READY = False


def ensure_ti():
    global _TAICHI_READY
    if _TAICHI_READY:
        return
    try:
        ti.init(arch=ti.cuda, device_memory_fraction=0.7)
        print("✅ Taichi CUDA (physics)")
    except Exception:
        ti.init(arch=ti.cpu)
        print("⚠️ Taichi CPU fallback (physics)")
    _TAICHI_READY = True


@ti.data_oriented
class RigidBodyPlane:
    def __init__(self, max_verts: int = 2048):
        ensure_ti()

        self.max_verts = max_verts
        self.v_body = ti.Vector.field(3, ti.f32, shape=max_verts)
        self.v_count = ti.field(ti.i32, shape=())

        # pose + motion
        self.pos = ti.Vector.field(3, ti.f32, shape=())
        self.vel = ti.Vector.field(3, ti.f32, shape=())
        self.ang = ti.Vector.field(3, ti.f32, shape=())   # Euler (simple & stable here)

        # flags
        self.enabled = ti.field(ti.i32, shape=())

        # params
        self.gravity = ti.Vector([0.0, -9.81, 0.0])
        self.lin_damp = 0.985
        self.ang_damp = 0.985
        self.friction = 0.88

        self._min_y = ti.field(ti.f32, shape=())

        # init
        self.enabled[None] = 0
        self.pos[None] = ti.Vector([0.0, 0.8, 0.0])
        self.vel[None] = ti.Vector([0.0, 0.0, 0.0])
        self.ang[None] = ti.Vector([0.0, 0.0, 0.0])
        self.v_count[None] = 0

    def set_mesh_vertices(self, verts_body):
        """
        verts_body: numpy (V,3) float32
        """
        import numpy as np
        verts_body = np.asarray(verts_body, dtype=np.float32)
        n = min(len(verts_body), self.max_verts)
        self.v_body.from_numpy(verts_body[:n])
        self.v_count[None] = n
        self.enabled[None] = 1
        # reset pose
        self.pos[None] = ti.Vector([0.0, 1.2, 0.0])
        self.vel[None] = ti.Vector([0.0, 0.0, 0.0])
        self.ang[None] = ti.Vector([0.0, 0.0, 0.0])

    @ti.func
    def _rot_x(self, a):
        c = ti.cos(a)
        s = ti.sin(a)
        return ti.Matrix([[1.0, 0.0, 0.0],
                          [0.0, c, -s],
                          [0.0, s, c]])

    @ti.func
    def _rot_y(self, a):
        c = ti.cos(a)
        s = ti.sin(a)
        return ti.Matrix([[c, 0.0, s],
                          [0.0, 1.0, 0.0],
                          [-s, 0.0, c]])

    @ti.func
    def _rot_z(self, a):
        c = ti.cos(a)
        s = ti.sin(a)
        return ti.Matrix([[c, -s, 0.0],
                          [s, c, 0.0],
                          [0.0, 0.0, 1.0]])

    @ti.func
    def _R(self):
        a = self.ang[None]
        # Z * Y * X (stable enough for our use)
        return self._rot_z(a.z) @ self._rot_y(a.y) @ self._rot_x(a.x)

    @ti.kernel
    def _compute_min_y(self):
        mn = 1e9
        R = self._R()
        p = self.pos[None]
        for i in range(self.v_count[None]):
            v = p + R @ self.v_body[i]
            mn = ti.min(mn, v.y)
        self._min_y[None] = mn

    @ti.kernel
    def _step(self, dt: ti.f32):
        if self.enabled[None] == 0:
            return

        # integrate
        v = self.vel[None]
        p = self.pos[None]
        a = self.ang[None]

        # gravity
        v += self.gravity * dt
        p += v * dt

        # damping
        v *= self.lin_damp
        a *= self.ang_damp

        self.vel[None] = v
        self.pos[None] = p
        self.ang[None] = a

    @ti.kernel
    def _resolve_plane(self):
        if self.enabled[None] == 0:
            return

        mn = self._min_y[None]
        if mn < 0.0:
            # push up
            p = self.pos[None]
            p.y -= mn
            self.pos[None] = p

            # kill downward velocity, apply friction to tangential motion
            v = self.vel[None]
            if v.y < 0:
                v.y = 0.0
            v.x *= self.friction
            v.z *= self.friction
            self.vel[None] = v

            # angular "friction"
            a = self.ang[None]
            a *= 0.92
            self.ang[None] = a

    def step(self, dt: float, substeps: int = 2):
        if self.enabled[None] == 0:
            return
        dt = float(max(1e-5, min(1.0 / 30.0, dt)))
        for _ in range(max(1, int(substeps))):
            self._step(dt / substeps)
            self._compute_min_y()
            self._resolve_plane()

    def add_rotation(self, dx: float, dy: float):
        # Called from mouse drag (CPU); apply to Euler angles
        a = self.ang[None]
        a.y += float(dx)
        a.x += float(dy)
        self.ang[None] = a

    def reset_rotation(self):
        a = self.ang[None]
        a.x = 0.0
        a.y = 0.0
        a.z = 0.0
        self.ang[None] = a

    def get_pose(self):
        # Return pose (pos, ang) for renderer
        p = self.pos[None]
        a = self.ang[None]
        return (float(p.x), float(p.y), float(p.z)), (float(a.x), float(a.y), float(a.z))
