import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import shutil


# ======================================================
#                      MESH
# ======================================================

class Mesh:

    def __init__(self, meshfile, boundaryfile, domain, blocks_num,
                 ngs_boundary=6, ngs_interior=5):

        self.vertices, self.elements = self.read_meshfile(meshfile)
        self.boundary_vertices, self.boundary_faces, self.boundary_type = (
            self.read_boundaryfile(boundaryfile)
        )

        self.interior_vertices = self.find_interior_points(
            self.vertices, self.boundary_vertices
        )

        self.domain = domain
        self.blocks_num = blocks_num
        self.ngs_interior = ngs_interior
        self.ngs_boundary = ngs_boundary

        # --------------------
        # GAUSS QUADRATURE
        # --------------------
        self.gs_boundary = {
            1: {'wts': [1], 'pts': [[1 / 3, 1 / 3, 1 / 3]]},
            3: {'wts': [1 / 3] * 3,
                'pts': [[2 / 3, 1 / 6, 1 / 6],
                        [1 / 6, 2 / 3, 1 / 6],
                        [1 / 6, 1 / 6, 2 / 3]]},
            6: {'wts': [0.109951743655322] * 3 +
                        [0.223381589678011] * 3,
                'pts': [[0.816847572980459, 0.091576213509661, 0.091576213509661],
                        [0.091576213509661, 0.816847572980459, 0.091576213509661],
                        [0.091576213509661, 0.091576213509661, 0.816847572980459],
                        [0.10810301816807, 0.445948490915965, 0.445948490915965],
                        [0.445948490915965, 0.10810301816807, 0.445948490915965],
                        [0.445948490915965, 0.445948490915965, 0.10810301816807]]}
        }

        self.gs_interior = {
            1: {'wts': [1], 'pts': [[0.25, 0.25, 0.25, 0.25]]},
            4: {'wts': [0.25]*4,
                'pts': [[0.58541019662497, 0.13819660112501, 0.13819660112501, 0.13819660112501],
                        [0.13819660112501, 0.58541019662497, 0.13819660112501, 0.13819660112501],
                        [0.13819660112501, 0.13819660112501, 0.58541019662497, 0.13819660112501],
                        [0.13819660112501, 0.13819660112501, 0.13819660112501, 0.58541019662497]]}
        }

        self.blocks = self._blocks_info()
        self.X_interior = self._get_X_interior()
        self.X_boundary = self._get_X_boundary()
        self.z_interior = self._get_Z_interior()
        self.z_blocks = self._z_blocks()

    # ======================================================
    # IO
    # ======================================================

    def read_meshfile(self, meshfile):
        with open(meshfile, 'r') as file:
            _ = next(file)
            line2 = next(file).strip()
            n_str, e_str, _, _ = line2.split(',')
            n = int(n_str.split('=')[1])
            e = int(e_str.split('=')[1])

            vertices = [list(map(float, next(file).split())) for _ in range(n)]
            tetrahedrons = [list(map(int, next(file).split())) for _ in range(e)]

        return np.array(vertices), np.array(tetrahedrons)

    def read_boundaryfile(self, boundaryfile):
        with open(boundaryfile, 'r') as file:
            _ = next(file)
            line2 = next(file)
            bnc, bec, _, _ = line2.split(',')
            bnc = int(bnc.split('=')[1])
            bec = int(bec.split('=')[1])

            boundary_vertices = [list(map(float, next(file).split())) for _ in range(bnc)]
            boundary_faces = [list(map(int, next(file).split())) for _ in range(bec)]
            boundary_type = [list(map(int, next(file).split())) for _ in range(bec)]

        return np.array(boundary_vertices), np.array(boundary_faces), np.array(boundary_type)

    # ======================================================
    # GEOMETRY
    # ======================================================

    def find_interior_points(self, vertices, boundary_vertices):
        interior = []
        for p in vertices:
            if not np.any(np.isclose(boundary_vertices, p, atol=1e-15).all(axis=1)):
                interior.append(p)
        return np.array(interior)

    def calc_area(self, vertices):
        a = vertices[1] - vertices[0]
        b = vertices[2] - vertices[0]
        return 0.5 * np.linalg.norm(np.cross(a, b))

    def calc_volume(self, vertices):
        M = np.array([vertices[1]-vertices[0],
                      vertices[2]-vertices[0],
                      vertices[3]-vertices[0]])
        return abs(np.linalg.det(M)) / 6

    # ======================================================
    # DOMAIN BLOCKS
    # ======================================================

    def _blocks_info(self):
        x0_min, x0_max, x1_min, x1_max, x2_min, x2_max = self.domain
        x0 = np.linspace(x0_min, x0_max, self.blocks_num[0] + 1)
        x1 = np.linspace(x1_min, x1_max, self.blocks_num[1] + 1)
        x2 = np.linspace(x2_min, x2_max, self.blocks_num[2] + 1)

        return np.array([
            (x0[i], x0[i+1], x1[j], x1[j+1], x2[k], x2[k+1])
            for i in range(len(x0)-1)
            for j in range(len(x1)-1)
            for k in range(len(x2)-1)
        ])

    def in_subdomain(self, z, subdomain):
        zc = z['coord']
        mask = (
            (zc[:, 0] >= subdomain[0]) & (zc[:, 0] <= subdomain[1]) &
            (zc[:, 1] >= subdomain[2]) & (zc[:, 1] <= subdomain[3]) &
            (zc[:, 2] >= subdomain[4]) & (zc[:, 2] <= subdomain[5])
        )
        return {'coord': zc[mask]}

    # ======================================================
    # GAUSS INTEGRATION
    # ======================================================

    def _get_X_boundary(self):
        Normal, Coord, Wts = [], [], []

        for fh in self.boundary_faces:
            vertices = np.vstack([self.boundary_vertices[p] for p in fh])
            v1 = vertices[1] - vertices[0]
            v2 = vertices[2] - vertices[1]
            normal = np.cross(v1, v2)
            normal /= (np.linalg.norm(normal) + 1e-12)
            area = self.calc_area(vertices)

            pts = np.dot(self.gs_boundary[self.ngs_boundary]['pts'], vertices)
            wts = area * np.array(self.gs_boundary[self.ngs_boundary]['wts'])

            Normal.append(normal)
            Coord.append(pts)
            Wts.append(wts)

        COORD = [np.array([Coord[i][k] for i in range(len(Coord))]) for k in range(self.ngs_boundary)]
        WTS = [np.array([Wts[i][k] for i in range(len(Wts))]) for k in range(self.ngs_boundary)]

        return {
            'coord': COORD,
            'wts': WTS,
            'normal': np.array(Normal),
            'boundary_type': self.boundary_type
        }

    def _get_X_interior(self):
        Coord, Wts = [], []

        for ele in self.elements:
            vertices = np.vstack([self.vertices[i] for i in ele])
            volume = self.calc_volume(vertices)
            pts = np.dot(self.gs_interior[self.ngs_interior]['pts'], vertices)
            wts = volume * np.array(self.gs_interior[self.ngs_interior]['wts'])
            Coord.append(pts)
            Wts.append(wts)

        COORD = [np.array([Coord[i][k] for i in range(len(self.elements))]) for k in range(self.ngs_interior)]
        WTS = [np.array([Wts[i][k] for i in range(len(self.elements))]) for k in range(self.ngs_interior)]

        return {'coord': COORD, 'wts': WTS}

    def _get_Z_interior(self):
        return {'coord': self.interior_vertices}

    def _z_blocks(self):
        return [self.in_subdomain(self.z_interior, block) for block in self.blocks]


# ======================================================
#                  VISUALIZATION
# ======================================================

def show_image(x, u, elems):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    tri = Triangulation(x[:, 0], x[:, 1], elems)
    surf = ax.plot_trisurf(tri, u[:, 0], cmap=plt.cm.Spectral)
    fig.colorbar(surf)
    return fig


# ======================================================
#                      LOSS
# ======================================================

class LpLoss:

    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num = x.shape[0]
        h = 1.0 / (x.shape[1] - 1.0)
        norms = h**(self.d/self.p) * torch.norm(
            x.view(num, -1) - y.view(num, -1), p=self.p, dim=1
        )
        return norms.mean() if self.size_average else norms.sum()

    def rel(self, x, y):
        num = x.shape[0]
        diff = torch.norm(x.view(num, -1) - y.view(num, -1), p=self.p, dim=1)
        norm = torch.norm(y.view(num, -1), p=self.p, dim=1)
        return (diff / norm).mean()

    def __call__(self, x, y):
        return self.abs(x, y)


# ======================================================
#                  CHECKPOINT
# ======================================================

def save_checkpoints(k, state, is_best=False, save_dir=None):
    path = os.path.join(save_dir, f'checkpoint_block{k}.pt')
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(save_dir, f'block{k}.pt')
        shutil.copyfile(path, best_path)
