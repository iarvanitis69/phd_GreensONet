import os
import torch
import numpy as np
from scipy.interpolate import griddata
import meshio


# ============================================================
# DIFFUSION - REACTION PROBLEM (PyTorch version)
# ============================================================

class DiffusionReaction(object):
    def __init__(self, data_folder, geometry):
        self.data_folder = data_folder
        self.geometry = geometry

    # ------------------------------------------------------------
    # Exact solution u(x)
    # ------------------------------------------------------------
    def u_exact(self, x, case_index):
        """
        x: torch.Tensor of shape (n, 3)
        """
        x_np = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f"res{case_index}.vtu")
        mesh = meshio.read(u_exact_file)

        points = mesh.points
        u_exact = mesh.point_data.get("u", None)

        interpolated_values = griddata(
            points=points,
            values=u_exact,
            xi=x_np,
            method="nearest"
        )

        result = torch.tensor(
            interpolated_values[..., np.newaxis],
            dtype=torch.float32,
            device=x.device
        )

        return result

    # ------------------------------------------------------------
    # Right-hand side f(x)
    # ------------------------------------------------------------
    def f(self, x, case_index):
        """
        x: torch.Tensor of shape (n, 3)
        """
        x_np = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f"res{case_index}.vtu")
        mesh = meshio.read(u_exact_file)

        points = mesh.points
        f_vals = mesh.point_data.get("f", None)

        interpolated_values = griddata(
            points=points,
            values=f_vals,
            xi=x_np,
            method="nearest"
        )

        res = torch.tensor(
            interpolated_values,
            dtype=torch.float32,
            device=x.device
        )

        return res

    # ------------------------------------------------------------
    # Diffusion coefficient a(x)
    # ------------------------------------------------------------
    def a(self, x):
        """
        x: torch.Tensor (n, 3)
        """
        if self.geometry == "plate":
            res = (
                (x[:, 0] - 0.5) ** 2
                + (x[:, 1] - 0.5) ** 2
                + (x[:, 2] - 0.1) ** 2
            )

        elif self.geometry == "pipe":
            res = torch.where(
                x[:, 0] > 0.08,
                torch.tensor(2e-2, device=x.device),
                torch.tensor(1e-2, device=x.device),
            )

        return res

    # ------------------------------------------------------------
    # Dirichlet boundary condition g(x)
    # ------------------------------------------------------------
    def g(self, x, case_index):
        """
        x: torch.Tensor of shape (n, 3)
        """
        x_np = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f"res{case_index}.vtu")
        mesh = meshio.read(u_exact_file)

        points = mesh.points
        u_exact = mesh.point_data.get("u", None)

        interpolated_values = griddata(
            points=points,
            values=u_exact,
            xi=x_np,
            method="nearest"
        )

        result = torch.tensor(
            interpolated_values,
            dtype=torch.float32,
            device=x.device
        )

        return result


# ============================================================
# POISSON = SPECIAL CASE OF DIFFUSION
# ============================================================

class Poisson(DiffusionReaction):
    def __init__(self, data_folder, geometry):
        super().__init__(data_folder, geometry)

    def a(self, x):
        return torch.ones_like(x[:, 0])


# ============================================================
# STOKES PROBLEM (VELOCITY FIELD)
# ============================================================

class Stokes(object):
    def __init__(self, data_folder, velocity_component_name):
        self.data_folder = data_folder
        self.velocity_component_name = velocity_component_name

    # ------------------------------------------------------------
    # Exact velocity
    # ------------------------------------------------------------
    def u_exact(self, x, case_index):
        x_np = x.detach().cpu().numpy()

        u_exact_file = os.path.join(self.data_folder, f"res{case_index}.vtu")
        mesh = meshio.read(u_exact_file)
        points = mesh.points

        if self.velocity_component_name == "x":
            var_name = "u"
        elif self.velocity_component_name == "y":
            var_name = "v"
        elif self.velocity_component_name == "z":
            var_name = "w"
        else:
            raise ValueError("Invalid velocity component name.")

        u_exact = mesh.point_data.get(var_name, None)

        interpolated_values = griddata(
            points=points,
            values=u_exact,
            xi=x_np,
            method="nearest"
        )

        return torch.tensor(
            interpolated_values[..., np.newaxis],
            dtype=torch.float32,
            device=x.device
        )

    # ------------------------------------------------------------
    # Right-hand force f(x)
    # ------------------------------------------------------------
    def f(self, x, case_index):
        x_np = x.detach().cpu().numpy()

        f_file = os.path.join(self.data_folder, f"res{case_index}.vtu")
        mesh = meshio.read(f_file)

        points = mesh.points
        fx = mesh.point_data.get("fx", None)

        interpolated_values = griddata(
            points=points,
            values=fx,
            xi=x_np,
            method="nearest"
        )

        return torch.tensor(
            interpolated_values[..., np.newaxis],
            dtype=torch.float32,
            device=x.device
        )

    # ------------------------------------------------------------
    # Diffusion coefficient
    # ------------------------------------------------------------
    def a(self, x):
        res = torch.ones_like(x[:, [0]]) / 100.0
        return res

    # ------------------------------------------------------------
    # Dirichlet BC
    # ------------------------------------------------------------
    def g(self, x, boundary_type):
        x_np = x.detach().cpu().numpy()
        boundary_np = boundary_type.detach().cpu().numpy()

        if x_np.shape[0] != boundary_np.shape[0]:
            raise ValueError(
                f"The number of points in x: {x_np.shape[0]} and boundary_type: {boundary_np.shape[0]} must be the same."
            )

        result = np.zeros_like(x_np[:, [0]])
        mask = boundary_np == 3
        result[mask.flatten()] = 1

        return torch.tensor(
            result, dtype=torch.float32, device=x.device
        )
