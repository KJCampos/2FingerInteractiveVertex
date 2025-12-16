from __future__ import annotations
import math
import numpy as np


class Physics3D:
    """Simple rigid-body approximation that settles on a plane."""

    def __init__(self):
        self.position = np.array([0.0, 1.2, 0.0], dtype=np.float32)
        self.velocity = np.zeros(3, dtype=np.float32)
        self.rotation = np.zeros(3, dtype=np.float32)  # pitch, yaw, roll
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.enabled = False

    def set_mesh(self, verts, tris):
        self.enabled = len(verts) > 0
        self.position[:] = np.array([0.0, 1.2, 0.0], dtype=np.float32)
        self.velocity[:] = 0.0
        self.rotation[:] = np.array([0.4, 0.25, 0.0], dtype=np.float32)
        self.ang_vel[:] = 0.0

    def step(self, dt: float, substeps: int = 2):
        if not self.enabled:
            return
        dt = max(1e-4, min(dt, 1.0 / 20.0))
        for _ in range(max(1, int(substeps))):
            self._integrate(dt / substeps)

    def _integrate(self, dt: float):
        g = np.array([0.0, -9.81, 0.0], dtype=np.float32)
        self.velocity += g * dt
        self.position += self.velocity * dt

        self.velocity *= 0.985
        self.ang_vel *= 0.985

        if self.position[1] < 0.0:
            self.position[1] = 0.0
            if self.velocity[1] < 0:
                self.velocity[1] = 0
            self.velocity[0] *= 0.88
            self.velocity[2] *= 0.88
            self.ang_vel *= 0.9

        self.rotation += self.ang_vel * dt

    def add_rotation(self, dx: float, dy: float):
        self.ang_vel[1] += float(dx) * 2.0
        self.ang_vel[0] += float(dy) * 2.0

    def reset_rotation(self):
        self.rotation[:] = 0.0
        self.ang_vel[:] = 0.0

    def get_pose(self):
        return tuple(self.position.tolist()), tuple(self.rotation.tolist())
