# predictor.py
from __future__ import annotations

from dataclasses import dataclass
import math


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


@dataclass
class _State:
    x: float = 0.5
    y: float = 0.5
    vx: float = 0.0
    vy: float = 0.0
    t: float = 0.0
    init: bool = False


class AlphaBetaPredictor:
    """
    Lightweight predictive filter (constant-velocity model).
    - Smooths jitter
    - Predicts slightly ahead (lead time) to reduce perceived latency
    - Adapts a bit (optional) based on residual magnitude
    """

    def __init__(self, alpha: float = 0.85, beta: float = 0.02, adaptive: bool = True):
        # alpha: how aggressively we correct position
        # beta:  how aggressively we correct velocity
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.adaptive = bool(adaptive)
        self.s = _State()

    def reset(self) -> None:
        self.s = _State()

    def update(self, pos_uv: tuple[float, float], t: float) -> None:
        x_m = float(pos_uv[0])
        y_m = float(pos_uv[1])

        if not self.s.init:
            self.s.x, self.s.y = _clamp01(x_m), _clamp01(y_m)
            self.s.vx, self.s.vy = 0.0, 0.0
            self.s.t = float(t)
            self.s.init = True
            return

        dt = float(t) - float(self.s.t)
        if dt <= 1e-6:
            return

        # Predict
        x_p = self.s.x + self.s.vx * dt
        y_p = self.s.y + self.s.vy * dt

        # Residual (measurement - prediction)
        rx = x_m - x_p
        ry = y_m - y_p

        # Optional "learning": adapt gains when residual is large (fast motion)
        a = self.alpha
        b = self.beta
        if self.adaptive:
            r = math.sqrt(rx * rx + ry * ry)
            # Increase responsiveness when motion is bigger
            a = min(0.95, max(0.60, self.alpha + 0.8 * r))
            b = min(0.25, max(0.01, self.beta + 0.6 * r))

        # Correct
        self.s.x = _clamp01(x_p + a * rx)
        self.s.y = _clamp01(y_p + a * ry)
        self.s.vx = self.s.vx + (b * rx / dt)
        self.s.vy = self.s.vy + (b * ry / dt)
        self.s.t = float(t)

    def predict(self, t_future: float) -> tuple[float, float]:
        if not self.s.init:
            return (0.5, 0.5)
        dt = float(t_future) - float(self.s.t)
        return (_clamp01(self.s.x + self.s.vx * dt), _clamp01(self.s.y + self.s.vy * dt))

    def velocity(self) -> tuple[float, float]:
        return (float(self.s.vx), float(self.s.vy))
