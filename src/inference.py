import os
import re
import sys
import time
import random
from types import SimpleNamespace

import torch
import meshio
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
import hydra
from omegaconf import DictConfig

from model import Net_Integral  # PyTorch έκδοση
from problem import DiffusionReaction, Poisson, Stokes  # PyTorch εκδόσεις
from utils import Mesh

sys.path.append("../external-libraries")


# ----------------------------------------------------------------------
# ΒΟΗΘΗΤΙΚΑ
# ----------------------------------------------------------------------
def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def extract_integers_from_string(input_string_list):
    pattern = re.compile(r"\d+")
    return [int(pattern.findall(s)[0]) for s in input_string_list]


def tile_for_blocks(x: torch.Tensor, z_blocks_coord: np.ndarray) -> torch.Tensor:
    """
    Αντί για paddle.tile → torch.repeat.
    x: (N, d)
    z_blocks_coord: np.array με shape (M, 3)
    Επαναλαμβάνουμε τα x M φορές:
        -> (M * N, d)
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x, dtype=torch.float32)

    if x.dim() == 1:
        x = x.unsqueeze(1)  # (N,) -> (N,1)

    num_blocks = z_blocks_coord.shape[0]
    x_expanded = x.unsqueeze(0).repeat(num_blocks, 1, 1)   # (M, N, d)
    x_tiled = x_expanded.reshape(-1, x.shape[-1])          # (M*N, d)
    return x_tiled


# ----------------------------------------------------------------------
# DATASET: VTK_Dataset (PyTorch)
# ----------------------------------------------------------------------
class VTK_Dataset(Dataset):
    def __init__(self, args, mesh, problem, train_samples=None):
        self.args = args
        self.problem = problem
        self.mesh = mesh

        # λίστα αρχείων στο data_folder του προβλήματος
        self.file_list = os.listdir(problem.data_folder)
        self.case_index_list = extract_integers_from_string(self.file_list)
        self.case_index_list.sort()
        if train_samples is not None:
            self.case_index_list = self.case_index_list[:train_samples]

        self.z_blocks = self.mesh.z_blocks

        # --- Interior: λίστες από numpy → λίστες από torch.Tensor ---
        self.x_in_wts = [
            torch.as_tensor(w, dtype=torch.float32)
            for w in self.mesh.X_interior["wts"]
        ]
        self.x_in_coord = [
            torch.as_tensor(c, dtype=torch.float32)
            for c in self.mesh.X_interior["coord"]
        ]

        # --- Boundary: λίστες από numpy → λίστες από torch.Tensor ---
        self.x_bc_wts = [
            torch.as_tensor(w, dtype=torch.float32)
            for w in self.mesh.X_boundary["wts"]
        ]
        self.x_bc_coord = [
            torch.as_tensor(c, dtype=torch.float32)
            for c in self.mesh.X_boundary["coord"]
        ]

        # --- Normals ---
        normals = self.mesh.X_boundary["normal"]
        if isinstance(normals, torch.Tensor):
            self.x_bc_normal = normals.float()
        else:
            self.x_bc_normal = torch.as_tensor(normals, dtype=torch.float32)

        # z_points στο πρώτο block (όπως στο αρχικό Paddle)
        self.z_blocks_coord = np.array(self.mesh.z_blocks[0]["coord"], dtype="float32")

        # boundary_type (μόνο για Stokes)
        if "boundary_type" in self.mesh.X_boundary.keys():
            bt = self.mesh.X_boundary["boundary_type"]
            if isinstance(bt, torch.Tensor):
                self.input_boundary_type = bt.float()
            else:
                self.input_boundary_type = torch.as_tensor(bt, dtype=torch.float32)

    def __getitem__(self, k):
        x_in_coord_list = []
        x_in_wts_list = []
        x_bc_coord_list = []
        x_bc_wts_list = []
        f_interior_list = []
        g_boundary_list = []
        a_boundary_list = []

        N_interior = self.x_in_coord[0].shape[0]
        N_boundary = self.x_bc_coord[0].shape[0]
        case_index = self.case_index_list[k]

        # Stokes: boundary_type είναι πεδίο, αλλιώς case_index
        if isinstance(self.problem, Stokes):
            input_boundary_type = tile_for_blocks(
                self.input_boundary_type, self.z_blocks_coord
            )[:, 0]
            input_boundary_list = []
            for i in range(self.args.ngs_boundary):
                coords_tiled = tile_for_blocks(self.x_bc_coord[i], self.z_blocks_coord)
                input_boundary_list.append(coords_tiled[:, :3])
        else:
            input_boundary_type = case_index
            input_boundary_list = self.x_bc_coord

        # Interior
        for i in range(self.args.ngs_interior):
            x_in_coord_list.append(self.x_in_coord[i])
            x_in_wts_list.append(self.x_in_wts[i])

            f_interior = self.problem.f(self.x_in_coord[i], case_index)
            if isinstance(self.problem, Stokes):
                f_interior = f_interior.reshape(N_interior, -1)
            f_interior_list.append(f_interior)

        # Boundary
        for i in range(self.args.ngs_boundary):
            x_bc_wts_list.append(self.x_bc_wts[i])
            x_bc_coord_list.append(self.x_bc_coord[i])

            g_boundary = self.problem.g(input_boundary_list[i], input_boundary_type)
            a_boundary = self.problem.a(input_boundary_list[i])

            if isinstance(self.problem, Stokes):
                g_boundary = g_boundary.reshape(N_boundary, -1)
                a_boundary = a_boundary.reshape(N_boundary, -1)

            g_boundary_list.append(g_boundary)
            a_boundary_list.append(a_boundary)

        return (
            x_in_coord_list,
            x_in_wts_list,
            x_bc_coord_list,
            x_bc_wts_list,
            f_interior_list,
            g_boundary_list,
            a_boundary_list,
            self.z_blocks,
            case_index,
            self.x_bc_normal,
        )

    def __len__(self):
        return len(self.case_index_list)


# ----------------------------------------------------------------------
# TESTER (PyTorch)
# ----------------------------------------------------------------------
class Tester:
    def __init__(self, args):
        self.args = args
        self.val_problem = args.val_problem

        self.mesh = Mesh(
            args.mesh_path,
            args.boundary_mesh_path,
            args.domain,
            args.blocks_num,
            ngs_boundary=self.args.ngs_boundary,
            ngs_interior=self.args.ngs_interior,
        )

        self.loss_function = nn.MSELoss()

        self.net_pde = Net_Integral(
            args.layers,
            args.shape,
            args.ngs_boundary,
            args.ngs_interior,
            args.val_problem,
            args.act,
            eval_mode=True,
        )

        # Αν υπάρχουν checkpoints από YAML
        if hasattr(self.args, "checkpoint_path"):
            for i in range(args.shape[0]):
                for j in range(args.shape[1]):
                    ckpt_path = str(self.args.checkpoint_path[j])
                    print("Loading the pre-trained model from : ", ckpt_path)

                    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
                        state_dict = torch.load(ckpt_path, map_location="cpu")
                    elif ckpt_path.endswith(".npy"):
                        checkpoint = np.load(ckpt_path, allow_pickle=True).item()
                        state_dict = {
                            "layers_list.0.weight": torch.tensor(
                                checkpoint["0"], dtype=torch.float32
                            ),
                            "layers_list.0.bias": torch.tensor(
                                checkpoint["1"].reshape(-1), dtype=torch.float32
                            ),
                            "layers_list.1.weight": torch.tensor(
                                checkpoint["2"], dtype=torch.float32
                            ),
                            "layers_list.1.bias": torch.tensor(
                                checkpoint["3"].reshape(-1), dtype=torch.float32
                            ),
                            "layers_list.2.weight": torch.tensor(
                                checkpoint["4"], dtype=torch.float32
                            ),
                            "layers_list.2.bias": torch.tensor(
                                checkpoint["5"].reshape(-1), dtype=torch.float32
                            ),
                            "layers_list.3.weight": torch.tensor(
                                checkpoint["6"], dtype=torch.float32
                            ),
                            "layers_list.3.bias": torch.tensor(
                                checkpoint["7"].reshape(-1), dtype=torch.float32
                            ),
                        }
                    else:
                        raise ValueError(f"Unknown checkpoint format: {ckpt_path}")

                    self.net_pde.G[i][j].load_state_dict(state_dict)
                    self.net_pde.G[i][j].eval()
        else:
            print("Initialize Evaluator (no checkpoint_path)")

    @torch.no_grad()
    def calculate(self, model_val=None):
        mse_all = []
        mse_bc_all = []

        for K in range(len(self.mesh.blocks)):
            if (
                self.mesh.z_blocks[K] is not None
                and len(self.mesh.z_blocks[K]["coord"]) > 0
            ):
                mse, mse_bc = self.calculate_block(K, model_val)
                mse_all.append(np.mean(mse))
                mse_bc_all.append(np.mean(mse_bc))
            else:
                raise ValueError(f"Block #{K} is empty!!!")

        return np.mean(mse_all), np.mean(mse_bc_all)

    @torch.no_grad()
    def calculate_block(self, k, model_val=None):
        loss_list = []
        mse_bc_list = []

        model = self.net_pde if model_val is None else model_val

        test_dataset = VTK_Dataset(
            self.args, self.mesh, self.val_problem, train_samples=self.args.test_samples
        )

        for data in test_dataset:
            (
                x_in_coord_list,
                x_in_wts_list,
                x_bc_coord_list,
                x_bc_wts_list,
                f_interior_list,
                g_boundary_list,
                a_boundary_list,
                z_blocks,
                case_index,
                x_bc_normal,
            ) = data

            coord_np = z_blocks[k]["coord"]
            coord = torch.as_tensor(coord_np, dtype=torch.float32)

            t0 = time.time()

            u_pred = model(
                x_in_coord_list,
                x_in_wts_list,
                x_bc_wts_list,
                x_bc_coord_list,
                coord,
                f_interior_list,
                g_boundary_list,
                a_boundary_list,
                x_bc_normal,
            )

            u_exac = self.val_problem.u_exact(coord, case_index)

            loss = self.loss_function(u_pred, u_exac)
            mse_bc = torch.sum((u_pred - u_exac) ** 2) / len(self.mesh.vertices)

            if self.args.save_vtk:
                self.save_to_vtk(case_index, coord, u_pred, u_exac)

            loss_list.append(loss.item())
            mse_bc_list.append(mse_bc.item())

        if model_val is None:
            print(
                f"MSE loss over [{len(test_dataset)}] test cases {(sum(loss_list) / len(loss_list)):.2e}"
            )
            print(
                f"MSE with BC loss over [{len(test_dataset)}] test cases {(sum(mse_bc_list) / len(mse_bc_list)):.2e}"
            )

        return loss_list, mse_bc_list

    def save_to_vtk(self, case_index, coords_tensor, u_pred_tensor, u_exac_tensor):
        coords = coords_tensor.detach().cpu().numpy()
        u_pred = u_pred_tensor.detach().cpu().numpy()
        u_exac = u_exac_tensor.detach().cpu().numpy()

        error = np.abs(u_pred - u_exac)
        sorted_data = []

        for node in self.mesh.vertices:
            if any(np.allclose(node, coord) for coord in coords):
                index = np.where(np.all(np.isclose(coords, node), axis=1))[0][0]
                sorted_data.append(
                    [
                        node[0],
                        node[1],
                        node[2],
                        u_pred[index][0],
                        u_exac[index][0],
                        error[index][0],
                    ]
                )
            else:
                node_np = node[np.newaxis, ...]
                node_t = torch.as_tensor(node_np, dtype=torch.float32)
                res = self.val_problem.u_exact(node_t, case_index)
                boundary_point_value = res.detach().cpu().numpy()
                sorted_data.append(
                    [
                        node[0],
                        node[1],
                        node[2],
                        boundary_point_value[0][0],
                        boundary_point_value[0][0],
                        0.0,
                    ]
                )

        sorted_data = np.array(sorted_data)
        mesh = meshio.Mesh(
            points=self.mesh.vertices,
            cells=[("tetra", self.mesh.elements)],
            point_data={
                "u_pred": sorted_data[:, 3],
                "u_exac": sorted_data[:, 4],
                "u_error": sorted_data[:, 5],
            },
        )
        mesh.write(
            f"../output/vtk_in_paraview/test_case{case_index + 1}_Greenonet_torch.vtu"
        )


# ----------------------------------------------------------------------
# HYDRA MAIN – ΟΛΑ ΑΠΟ ΤΟ YAML
# ----------------------------------------------------------------------
@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="diffusion_plate.yaml",
)
def main(cfg: DictConfig):

    args = SimpleNamespace()

    # Device / seed
    args.device = torch.device(cfg.device)
    args.seed = cfg.seed
    set_random_seed(args.seed)

    # PDE / Geometry
    args.pde_case = cfg.pde.case
    args.geometry = cfg.pde.geometry
    args.domain = cfg.pde.domain

    # Mesh
    args.mesh_path = cfg.mesh.interior_path
    args.boundary_mesh_path = cfg.mesh.boundary_path
    args.blocks_num = cfg.model.blocks_num
    args.ngs_boundary = cfg.mesh.ngs_boundary
    args.ngs_interior = cfg.mesh.ngs_interior

    # Model
    args.shape = cfg.model.shape
    args.layers = cfg.model.layers
    args.act = cfg.model.activation

    # Training / Testing
    args.train_samples = cfg.training.train_samples
    args.test_samples = cfg.training.test_samples

    # VTK
    args.save_vtk = cfg.vtk.save_vtk

    # Checkpoints (προαιρετικά)
    if "checkpoint" in cfg and cfg.checkpoint:
        if isinstance(cfg.checkpoint, (list, tuple)):
            args.checkpoint_path = list(cfg.checkpoint)
        else:
            args.checkpoint_path = [cfg.checkpoint]

    # val_problem ανάλογα με το pde_case
    test_data_folder = cfg.data.test_data

    if args.pde_case.lower() == "diffusion":
        args.val_problem = DiffusionReaction(test_data_folder, geometry=args.geometry)
    elif args.pde_case.lower() == "poisson":
        args.val_problem = Poisson(test_data_folder, geometry=args.geometry)
    elif args.pde_case.lower() == "stokes":
        vel_comp = getattr(cfg.pde, "velocity_component_name", "x")
        args.velocity_component_name = vel_comp
        args.val_problem = Stokes(test_data_folder, vel_comp)
    else:
        raise ValueError(f"Unknown pde.case: {args.pde_case}")

    tester = Tester(args)

    with torch.no_grad():
        mse, mse_bc = tester.calculate()
        print(f"\n===> Final MSE: {mse:.5e}, MSE_BC: {mse_bc:.5e}")


if __name__ == "__main__":
    main()
