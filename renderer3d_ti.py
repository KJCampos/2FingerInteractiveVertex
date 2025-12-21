from __future__ import annotations
import math
import numpy as np
import cv2


class Renderer3D:
    """Lightweight renderer that projects one or more meshes into a viewport."""

    def __init__(self, width: int = 480, height: int = 480):
        self.width = int(width)
        self.height = int(height)
        self.zoom = 1.0

    def render_scene(self, objects, selected_id=None, glow: bool = True):
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        centers = {}

        if not objects:
            return img, centers

        cx, cy_img = self.width * 0.5, self.height * 0.55
        f = self.width * (0.55 + 0.9 * self.zoom)

        for obj in objects:
            verts = obj.mesh.verts
            tris = obj.mesh.tris
            if len(verts) == 0 or len(tris) == 0:
                continue

            yaw, pitch, roll = obj.rotation[1], obj.rotation[0], obj.rotation[2]
            cyaw, syaw = math.cos(yaw), math.sin(yaw)
            cp, sp = math.cos(pitch), math.sin(pitch)
            cr, sr = math.cos(roll), math.sin(roll)

            Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float32)
            Ry = np.array([[cyaw, 0, syaw], [0, 1, 0], [-syaw, 0, cyaw]], dtype=np.float32)
            Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float32)
            R = Rz @ Ry @ Rx

            pts = (verts @ R.T) + np.asarray(obj.position, dtype=np.float32)

            proj = []
            for x, y, z in pts:
                zz = z + 4.0
                sx = int(cx + (x / zz) * f)
                sy = int(cy_img - (y / zz) * f)
                proj.append((sx, sy, zz))

            proj_np = np.array(proj, dtype=np.float32)
            centers[obj.oid] = (
                float(np.mean(proj_np[:, 0])),
                float(np.mean(proj_np[:, 1])),
            )

            face_color = obj.color
            outline = (min(255, face_color[0] + 40), min(255, face_color[1] + 40), min(255, face_color[2] + 40))
            if obj.oid == selected_id:
                outline = (255, 255, 180)

            for tri in tris:
                pts2d = np.array([(proj_np[i, 0], proj_np[i, 1]) for i in tri], dtype=np.int32)
                cv2.fillConvexPoly(img, pts2d, face_color, cv2.LINE_AA)
                cv2.polylines(img, [pts2d], True, outline, 1, cv2.LINE_AA)

        if glow:
            blur = cv2.GaussianBlur(img, (0, 0), 3)
            img = cv2.addWeighted(img, 0.8, blur, 0.3, 0)

        cv2.rectangle(img, (6, 6), (self.width - 6, self.height - 6), (90, 140, 160), 1, cv2.LINE_AA)
        return img, centers
