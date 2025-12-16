from __future__ import annotations
import math
import numpy as np
import cv2


class Renderer3D:
    """Lightweight renderer that projects the mesh into its own viewport."""

    def __init__(self, width: int = 480, height: int = 480):
        self.width = int(width)
        self.height = int(height)
        self.mesh_verts = np.zeros((0, 3), dtype=np.float32)
        self.mesh_tris = np.zeros((0, 3), dtype=np.int32)
        self.pos = (0.0, 0.0, 0.0)
        self.ang = (0.0, 0.0, 0.0)

    def set_mesh(self, verts, tris):
        self.mesh_verts = np.asarray(verts, dtype=np.float32)
        self.mesh_tris = np.asarray(tris, dtype=np.int32)

    def set_pose(self, pos, ang):
        self.pos = pos
        self.ang = ang

    def render(self) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        if len(self.mesh_verts) == 0 or len(self.mesh_tris) == 0:
            return img

        yaw, pitch, roll = self.ang[1], self.ang[0], self.ang[2]

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
        R = Rz @ Ry @ Rx

        pts = (self.mesh_verts @ R.T) + np.asarray(self.pos, dtype=np.float32)

        cx, cy_img = self.width * 0.5, self.height * 0.55
        f = self.width * 0.9
        proj = []
        for x, y, z in pts:
            zz = z + 4.0
            sx = int(cx + (x / zz) * f)
            sy = int(cy_img - (y / zz) * f)
            proj.append((sx, sy, zz))

        proj_np = np.array(proj, dtype=np.float32)

        order = np.argsort(-proj_np[:, 2])
        for idx in order:
            pass

        face_color = (200, 240, 255)
        outline = (120, 200, 240)

        for tri in self.mesh_tris:
            pts2d = np.array([(proj_np[i, 0], proj_np[i, 1]) for i in tri], dtype=np.int32)
            cv2.fillConvexPoly(img, pts2d, face_color, cv2.LINE_AA)
            cv2.polylines(img, [pts2d], True, outline, 1, cv2.LINE_AA)

        cv2.circle(img, (int(cx), int(cy_img)), 4, (255, 160, 160), -1, cv2.LINE_AA)
        return img
