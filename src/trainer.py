import os
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging

sys.path.append("../external-libraries")

from model import Net_Integral, empty_context_manager
from utils import Mesh
from inference import VTK_Dataset
from problem import DiffusionReaction, Poisson, Stokes

log = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, args):
        self.args = args
        self.cuda_index = args.cuda_index
        self.device = args.device
        self.pde_case = args.pde_case
        self.resume = args.resume
        self.problem = args.problem
        self.tol = args.tol
        self.tol_change = args.tol_change
        self.domain = args.domain
        self.blocks_num = args.blocks_num
        self.ngs_boundary = args.ngs_boundary
        self.ngs_interior = args.ngs_interior
        self.train_samples = args.train_samples
        self.test_samples = args.test_samples

        meshfile, boundaryfile = self.get_mesh_path(
            self.args.mesh_path,
            self.args.boundary_mesh_path
        )

        self.mesh = Mesh(
            meshfile,
            boundaryfile,
            self.domain,
            self.blocks_num,
            ngs_boundary=self.ngs_boundary,
            ngs_interior=self.ngs_interior
        )

        self.criterion = nn.MSELoss()

        self.epochs_Adam = self.args.epochs_Adam
        self.lam = self.args.lam
        self.lr = self.args.lr

        self.net_pde = Net_Integral(
            self.args.layers,
            self.args.shape,
            self.ngs_boundary,
            self.ngs_interior,
            self.problem,
            self.args.act
        ).to(self.device)

        # ---------------------------------
        # RESUME TRAINING
        # ---------------------------------
        if self.args.resume:
            log.info(f"Resuming training, loading {args.resume} ...")

            for i in range(self.args.shape[0]):
                for j in range(self.args.shape[1]):

                    ckpt = self.args.resume[j]

                    if ckpt.endswith(".pt"):
                        state_dict = torch.load(ckpt, map_location=self.device)

                    elif ckpt.endswith(".npy"):
                        checkpoint = np.load(ckpt, allow_pickle=True).item()

                        state_dict = {
                            "layers_list.0.weight": torch.tensor(checkpoint["0"]),
                            "layers_list.0.bias": torch.tensor(checkpoint["1"]).view(-1),
                            "layers_list.1.weight": torch.tensor(checkpoint["2"]),
                            "layers_list.1.bias": torch.tensor(checkpoint["3"]).view(-1),
                            "layers_list.2.weight": torch.tensor(checkpoint["4"]),
                            "layers_list.2.bias": torch.tensor(checkpoint["5"]).view(-1),
                            "layers_list.3.weight": torch.tensor(checkpoint["6"]),
                            "layers_list.3.bias": torch.tensor(checkpoint["7"]).view(-1),
                        }

                    else:
                        raise ValueError("Unknown checkpoint format")

                    self.net_pde.G[i][j].load_state_dict(state_dict)
                    self.net_pde.G[i][j].eval()

    # -------------------------------------------------
    def get_mesh_path(self, mesh_path, boundary_mesh_path):
        return mesh_path, boundary_mesh_path

    # -------------------------------------------------
    def train(self):

        # ---------- MESH TENSORS TO DEVICE ----------
        self.mesh.X_boundary["normal"] = torch.tensor(
            self.mesh.X_boundary["normal"],
            dtype=torch.float32,
            device=self.device
        )

        self.mesh.X_boundary["boundary_type"] = torch.tensor(
            self.mesh.X_boundary["boundary_type"],
            dtype=torch.float32,
            device=self.device
        )

        for k in range(self.ngs_interior):
            self.mesh.X_interior["coord"][k] = torch.tensor(
                self.mesh.X_interior["coord"][k],
                dtype=torch.float32,
                device=self.device
            )
            self.mesh.X_interior["wts"][k] = torch.tensor(
                self.mesh.X_interior["wts"][k],
                dtype=torch.float32,
                device=self.device
            )

        for k in range(self.ngs_boundary):
            self.mesh.X_boundary["coord"][k] = torch.tensor(
                self.mesh.X_boundary["coord"][k],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )

            self.mesh.X_boundary["wts"][k] = torch.tensor(
                self.mesh.X_boundary["wts"][k],
                dtype=torch.float32,
                device=self.device,
                requires_grad=True
            )

        # ---------- TRAIN BLOCKS ----------
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K]) > 0:
                self.mesh.z_blocks[K]["coord"] = torch.tensor(
                    self.mesh.z_blocks[K]["coord"],
                    dtype=torch.float32,
                    device=self.device
                )
                self.train_block(K)
            else:
                log.info(f"Block #{K} is empty!")

    # -------------------------------------------------
    def train_block(self, k):

        for i in range(self.args.shape[0]):
            for j in range(self.args.shape[1]):
                self.net_pde.G[i][j].train()

        params = [
            p
            for i in range(self.args.shape[0])
            for j in range(self.args.shape[1])
            for p in self.net_pde.G[i][j].parameters()
            if p.requires_grad
        ]

        self.optimizer_Adam = optim.Adam(
            params,
            lr=self.lr,
            weight_decay=self.args.weight_decay
        )

        if self.args.lr_scheduler == "StepDecay":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer_Adam,
                step_size=self.args.lr_scheduler_step_size,
                gamma=0.9,
            )
        elif self.args.lr_scheduler == "const":
            self.lr_scheduler = None
        else:
            raise ValueError("Unknown lr scheduler type")

        best_train_loss = 1.0
        log.info(f"Start Training (block #{k})")
        tt = time.time()

        train_dataset = VTK_Dataset(
            self.args,
            self.mesh,
            self.problem,
            self.train_samples
        )

        # -------------------------------------------------
        for epoch in range(self.epochs_Adam):

            train_loss = 0
            total_loss = 0

            for data in train_dataset:

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

                coord = z_blocks[k]["coord"]

                self.optimizer_Adam.zero_grad()

                u_pred = self.net_pde(
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

                u_exac = self.problem.u_exact(coord, case_index)
                u_exac = torch.tensor(u_exac, dtype=torch.float32, device=self.device)

                loss = self.criterion(u_pred, u_exac)

                mean_loss = torch.sum((u_pred - u_exac) ** 2) / len(self.mesh.vertices)

                loss.backward()
                self.optimizer_Adam.step()

                train_loss += loss.item()
                total_loss += mean_loss.item()

            # ---------- LR SCHEDULER ----------
            if self.lr_scheduler and epoch >= 1000:
                self.lr_scheduler.step()

            current_lr = self.optimizer_Adam.param_groups[0]["lr"]

            train_loss /= self.train_samples
            total_loss /= self.train_samples

            # ---------- VALIDATION ----------
            if isinstance(self.problem, Stokes):
                self.net_pde.no_grad = torch.no_grad
                val_mse, val_mse_bc = self.args.tester.calculate(self.net_pde)
                self.net_pde.no_grad = empty_context_manager
            else:
                with torch.no_grad():
                    val_mse, val_mse_bc = self.args.tester.calculate(self.net_pde)

            t2 = time.time()

            log.info(
                f"Epoch {epoch}/{self.epochs_Adam} | "
                f"time {t2 - tt:.2f}s | "
                f"lr {current_lr:.2e} | "
                f"train_mse {train_loss:.5f} | "
                f"train_mse_bc {total_loss:.5f} | "
                f"val_mse {val_mse:.5f} | "
                f"val_mse_bc {val_mse_bc:.5f}"
            )

            tt = time.time()

            # ---------- SAVE BEST ----------
            if (epoch + 1) % 5 == 0:

                if train_loss < best_train_loss:
                    best_train_loss = train_loss

                    for i in range(self.args.shape[0]):
                        for j in range(self.args.shape[1]):
                            checkpoint_path = os.path.join(
                                self.args.output_dir,
                                f"G{j}_epoch_{epoch}.pt"
                            )

                            torch.save(
                                self.net_pde.G[i][j].state_dict(),
                                checkpoint_path
                            )
