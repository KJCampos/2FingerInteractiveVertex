"""
Particle simulation that feels "fluid-ish" without full SPH yet.

State:
- positions: Nx2 in [0..1] sim coordinates
- velocities: Nx2 sim units / second

Forces:
- gravity
- cohesion: particles softly attract toward nearby neighbors (simple)
- hand force: push or pull based on pinch
"""

import numpy as np
import cv2


class ParticleSim:
    def __init__(self, params):
        self.params = params
        self.rng = np.random.default_rng(0)

        # Hand input (set each frame)
        self.hand_present = False
        self.hand_pos = np.array([0.0, 0.0], dtype=np.float32)  # in 0..1
        self.hand_vel = np.array([0.0, 0.0], dtype=np.float32)  # in 0..1 / sec
        self.pinch = 0.0

        self.reset()

    def reset(self):
        n = self.params.num_particles

        # Spawn particles in a blob in the center
        center = np.array([0.5, 0.55], dtype=np.float32)
        spread = np.array([0.18, 0.18], dtype=np.float32)

        self.pos = center + (self.rng.random((n, 2)).astype(np.float32) - 0.5) * spread
        self.pos = np.clip(self.pos, 0.02, 0.98)

        self.vel = (self.rng.random((n, 2)).astype(np.float32) - 0.5) * 0.05

    def set_hand_input(self, palm_uv, vel_uv, pinch, hand_present):
        self.hand_present = hand_present
        self.hand_pos = np.array(palm_uv, dtype=np.float32)
        self.hand_vel = np.array(vel_uv, dtype=np.float32)
        self.pinch = float(pinch)

    def step(self, dt):
        p = self.params

        # --- Gravity (downward) ---
        self.vel[:, 1] += p.gravity * dt

        # --- Cohesion (simple neighbor pull) ---
        # This is O(N^2) for MVP simplicity. With 1500 particles it’s okay.
        # Later we replace this with a grid / SPH neighbor search.
        r = p.neighbor_radius
        r2 = r * r

        # Compute pairwise differences
        # diff[i, j] = pos[j] - pos[i]
        diff = self.pos[None, :, :] - self.pos[:, None, :]
        d2 = (diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)

        # neighbor mask (exclude self with d2 > 0)
        mask = (d2 > 1e-12) & (d2 < r2)

        # For each particle, compute a small pull toward neighbor average
        # direction is diff normalized, weighted by closeness
        # weight = (1 - dist/r)
        dist = np.sqrt(np.maximum(d2, 1e-12))
        weight = np.clip(1.0 - (dist / r), 0.0, 1.0)

        # Only apply for neighbors
        weight *= mask.astype(np.float32)

        # Sum weighted direction vectors
        dir_vec = diff / dist[:, :, None]
        cohesion_force = (dir_vec * weight[:, :, None]).sum(axis=1)

        self.vel += (p.cohesion * cohesion_force) * dt

        # --- Hand force field ---
        if self.hand_present:
            to_hand = self.pos - self.hand_pos[None, :]
            d = np.linalg.norm(to_hand, axis=1)
            inside = d < p.hand_radius

            if np.any(inside):
                # Normalize direction
                dirn = np.zeros_like(to_hand)
                dirn[inside] = to_hand[inside] / (d[inside][:, None] + 1e-6)

                # Falloff: closer = stronger
                falloff = (1.0 - (d / p.hand_radius))
                falloff = np.clip(falloff, 0.0, 1.0)

                # Decide push vs pull
                if p.auto_mode:
                    # pinch high => pull, open => push
                    pulling = self.pinch > 0.60
                else:
                    pulling = not p.force_mode_push

                strength = p.hand_strength

                # Push: away from hand center
                # Pull: toward hand center (flip direction)
                sign = -1.0 if pulling else 1.0

                #Clamp hand velocity impulse
                hv = np.clip(self.hand_vel, -2.0, 2.0)
                impulse = self.hand_vel[None, :] * 0.25

                self.vel[inside] += sign * dirn[inside] * (strength * falloff[inside][:, None]) * dt
                self.vel[inside] += impulse

        # --- Damping ---
        self.vel *= p.damping

        # --- Clamp speed ---
        speed = np.linalg.norm(self.vel, axis=1)
        too_fast = speed > p.max_speed
        if np.any(too_fast):
            self.vel[too_fast] *= (p.max_speed / (speed[too_fast][:, None] + 1e-6))

        # --- Integrate ---
        self.pos += self.vel * dt

        # --- Boundaries (bounce) ---
        self._solve_bounds()

    def _solve_bounds(self):
        # Keep particles within [0..1] with a simple bounce
        bounce = 0.65

        # X boundaries
        left = self.pos[:, 0] < 0.01
        right = self.pos[:, 0] > 0.99
        self.pos[left, 0] = 0.01
        self.pos[right, 0] = 0.99
        self.vel[left | right, 0] *= -bounce

        # Y boundaries
        top = self.pos[:, 1] < 0.01
        bottom = self.pos[:, 1] > 0.99
        self.pos[top, 1] = 0.01
        self.pos[bottom, 1] = 0.99
        self.vel[top | bottom, 1] *= -bounce

    def render_on_frame(self, frame_bgr):
        """
        Draw particles on top of the camera frame.
        """
        h, w = frame_bgr.shape[:2]
        r = self.params.particle_radius_px

        # Convert sim coords to pixels
        xs = (self.pos[:, 0] * w).astype(np.int32)
        ys = (self.pos[:, 1] * h).astype(np.int32)

        for x, y in zip(xs, ys):
            cv2.circle(frame_bgr, (x, y), r, (0, 140, 255), -1)
