from __future__ import annotations
from dataclasses import dataclass
import math
from typing import List, Tuple
import numpy as np


@dataclass
class Mesh25D:
    verts: np.ndarray  # (N,3) float32
    tris: np.ndarray   # (M,3) int32


def _area(poly: List[Tuple[float, float]]) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _is_clockwise(poly: List[Tuple[float, float]]) -> bool:
    return _area(poly) < 0


def _remove_near_duplicates(poly: List[Tuple[float, float]], min_dist: float) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for p in poly:
        if not cleaned or math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) >= min_dist:
            cleaned.append(p)
    if len(cleaned) > 2 and math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < min_dist:
        cleaned[-1] = cleaned[0]
        cleaned = cleaned[:-1]
    return cleaned


def _point_in_tri(p, a, b, c):
    px, py = p
    ax, ay = a
    bx, by = b
    cx, cy = c
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay

    dot00 = v0x * v0x + v0y * v0y
    dot01 = v0x * v1x + v0y * v1y
    dot02 = v0x * v2x + v0y * v2y
    dot11 = v1x * v1x + v1y * v1y
    dot12 = v1x * v2x + v1y * v2y

    inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-9)
    u = (dot11 * dot02 - dot01 * dot12) * inv_denom
    v = (dot00 * dot12 - dot01 * dot02) * inv_denom
    return (u >= 0) and (v >= 0) and (u + v <= 1)


def triangulate_earclip(poly: List[Tuple[float, float]]) -> List[Tuple[int, int, int]]:
    n = len(poly)
    if n < 3:
        return []

    indices = list(range(n))
    tris: List[Tuple[int, int, int]] = []

    def is_convex(i0, i1, i2):
        ax, ay = poly[i0]
        bx, by = poly[i1]
        cx, cy = poly[i2]
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0

    guard = 0
    while len(indices) > 3 and guard < 5000:
        guard += 1
        ear_found = False
        for k in range(len(indices)):
            i_prev = indices[(k - 1) % len(indices)]
            i_curr = indices[k]
            i_next = indices[(k + 1) % len(indices)]

            if not is_convex(i_prev, i_curr, i_next):
                continue

            a, b, c = poly[i_prev], poly[i_curr], poly[i_next]
            is_ear = True
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_tri(poly[j], a, b, c):
                    is_ear = False
                    break
            if is_ear:
                tris.append((i_prev, i_curr, i_next))
                indices.pop(k)
                ear_found = True
                break
        if not ear_found:
            break

    if len(indices) == 3:
        tris.append(tuple(indices))
    return tris


def build_polygon_from_chain(points: List[Tuple[int, int]], min_dist_px: float = 4.0) -> List[Tuple[float, float]]:
    if len(points) < 3:
        return []
    pts = [(float(p[0]), float(p[1])) for p in points]
    pts = _remove_near_duplicates(pts, min_dist_px)
    if len(pts) < 3:
        return []
    if _is_clockwise(pts):
        pts.reverse()
    return pts


def extrude_polygon(poly_px: List[Tuple[float, float]], scale: float, thickness: float) -> Mesh25D:
    poly = np.array(poly_px, dtype=np.float32)
    center = np.mean(poly, axis=0)
    centered = (poly - center) * float(scale)
    n = len(centered)

    bottom = np.concatenate([centered, np.zeros((n, 1), dtype=np.float32)], axis=1)
    top = np.concatenate([centered, np.full((n, 1), thickness, dtype=np.float32)], axis=1)
    verts = np.vstack([bottom, top]).astype(np.float32)

    tri_indices = triangulate_earclip(centered.tolist())
    tris: List[Tuple[int, int, int]] = []
    for a, b, c in tri_indices:
        tris.append((a, b, c))
        tris.append((a + n, c + n, b + n))

    for i in range(n):
        j = (i + 1) % n
        tris.append((i, j, i + n))
        tris.append((j, j + n, i + n))

    return Mesh25D(verts=verts, tris=np.array(tris, dtype=np.int32))
