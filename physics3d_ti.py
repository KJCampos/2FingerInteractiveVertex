# physics3d_ti.py
class Physics3D:
    def __init__(self, *args, **kwargs):
        self.enabled = True

    def set_mesh(self, verts, faces):
        self.verts = verts
        self.faces = faces

    def reset(self):
        pass

    def step(self, dt: float):
        pass
