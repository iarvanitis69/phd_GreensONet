import sys
sys.path.append("../external-libraries")
import paddle
import os
import time
import numpy as np
from model import Net_Integral, empty_context_manager
from utils import Mesh
from Inference import VTK_Dataset
from problem import DiffusionReaction, Poisson, Stokes
import hydra
import logging
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
        meshfile, boundaryfile = self.get_mesh_path(self.args.mesh_path, args.
            boundary_mesh_path)
        self.mesh = Mesh(meshfile, boundaryfile, self.domain, self.
            blocks_num, ngs_boundary=self.ngs_boundary, ngs_interior=self.
            ngs_interior)
        self.criterion = paddle.nn.MSELoss()
        self.epochs_Adam = self.args.epochs_Adam
        self.lam = self.args.lam
        self.lr = self.args.lr
        self.net_pde = Net_Integral(self.args.layers, args.shape, self.ngs_boundary, self.ngs_interior, self.problem, args.act)
        if self.args.resume:
            log.info(f'Resuming training, loading {args.resume} ...')
            for i in range(self.args.shape[0]):
                for j in range(self.args.shape[1]):
                    if self.args.resume[j].endswith("pdparams"):
                        state_dict = paddle.load(path=str(self.args.resume[j]))
                    elif self.args.resume[j].endswith("npy"):
                        checkpoint = np.load(str(self.args.resume[j]), allow_pickle=True).item()
                        # self.net_pde.G[i][j]
                        state_dict = {}
                        state_dict['layers_list.0.weight'] = checkpoint['0']
                        state_dict['layers_list.0.bias'] = checkpoint['1'].reshape(-1)
                        state_dict['layers_list.1.weight'] = checkpoint['2']
                        state_dict['layers_list.1.bias'] = checkpoint['3'].reshape(-1)
                        state_dict['layers_list.2.weight'] = checkpoint['4']
                        state_dict['layers_list.2.bias'] = checkpoint['5'].reshape(-1)
                        state_dict['layers_list.3.weight'] = checkpoint['6']
                        state_dict['layers_list.3.bias'] = checkpoint['7'].reshape(-1)
                        state_dict = {k:paddle.to_tensor(v, dtype="float32") for k, v in state_dict.items()}
                    self.net_pde.G[i][j].set_state_dict(state_dict)
                    self.net_pde.G[i][j].eval()
        else:
            pass

    def get_mesh_path(self, mesh_path, boundary_mesh_path):
        return mesh_path, boundary_mesh_path

    def train(self):
        if not isinstance(self.mesh.X_boundary['normal'], paddle.Tensor):
            self.mesh.X_boundary['normal'] = paddle.to_tensor(data=self.
                mesh.X_boundary['normal']).astype(dtype='float32').to(self.
                device)
            self.mesh.X_boundary['boundary_type'] = paddle.to_tensor(data=
                self.mesh.X_boundary['boundary_type']).astype(dtype='float32'
                ).to(self.device)
            for k in range(self.ngs_interior):
                self.mesh.X_interior['coord'][k] = paddle.to_tensor(data=
                    self.mesh.X_interior['coord'][k]).astype(dtype='float32'
                    ).to(self.device)
                self.mesh.X_interior['wts'][k] = paddle.to_tensor(data=self
                    .mesh.X_interior['wts'][k]).astype(dtype='float32').to(self
                    .device)
            for k in range(self.ngs_boundary):
                out_34 = paddle.to_tensor(data=self.mesh.X_boundary['coord'][k]
                    ).astype(dtype='float32').to(self.device)
                out_34.stop_gradient = not True
                self.mesh.X_boundary['coord'][k] = out_34
                out_35 = paddle.to_tensor(data=self.mesh.X_boundary['wts'][k]
                    ).astype(dtype='float32').to(self.device)
                out_35.stop_gradient = not True
                self.mesh.X_boundary['wts'][k] = out_35
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K]) > 0:
                if not isinstance(self.mesh.z_blocks[K]['coord'], paddle.Tensor):
                    self.mesh.z_blocks[K]['coord'] = paddle.to_tensor(data=
                        self.mesh.z_blocks[K]['coord']).astype(dtype='float32')
                self.train_block(K)
            else:
                log.info(f'Block #{K} is empty!!!')

    def train_block(self, k):
        for i in range(self.args.shape[0]):
            for j in range(self.args.shape[1]):
                self.net_pde.G[i][j].train()
                self.net_pde.G[i][j].to(self.device)
                self.net_pde.G[i][j].clear_gradients(set_to_zero=False)
        params = [param for i in range(self.args.shape[0]) for j in range(self.args.
            shape[1]) for param in self.net_pde.G[i][j].parameters() if (not
            param.stop_gradient) == True]
        self.optimizer_Adam = paddle.optimizer.Adam(parameters=params,
            learning_rate=self.lr, weight_decay=self.args.weight_decay)
        if self.args.lr_scheduler == "StepDecay":
            self.lr_scheduler = paddle.optimizer.lr.StepDecay(
                step_size=self.args.lr_scheduler_step_size,
                gamma=0.9,
                learning_rate=self.optimizer_Adam.get_lr())
            self.optimizer_Adam.set_lr_scheduler(self.lr_scheduler)
        elif self.args.lr_scheduler == "const":
            self.lr_scheduler = None
        else:
            raise ValueError(f"Unknown lr scheduler type: {self.args.lr_scheduler}")
        best_train_loss = 1.0
        log.info(f'Start Trainning (blocks #{k})')
        tt = time.time()
        train_dataset = VTK_Dataset(self.args, self.mesh, self.problem, self.train_samples)

        for epoch in range(self.epochs_Adam):
            train_loss = 0
            total_loss = 0
            for data in train_dataset:
                x_in_coord_list, x_in_wts_list, x_bc_coord_list, x_bc_wts_list, f_interior_list, g_boundary_list, a_boundary_list, z_blocks, case_index, x_bc_normal = data 
                coord = z_blocks[k]['coord']
                self.optimizer_Adam.clear_gradients(set_to_zero=False)
                u_pred = self.net_pde(
                    x_in_coord_list, 
                    x_in_wts_list, 
                    x_bc_wts_list,
                    x_bc_coord_list,
                    coord,
                    f_interior_list,
                    g_boundary_list,
                    a_boundary_list,
                    x_bc_normal
                )
                u_exac = self.problem.u_exact(coord.numpy(), case_index)
                u_exac = paddle.to_tensor(u_exac, dtype='float32')
                loss = self.criterion(u_pred, u_exac)
                mean_loss = paddle.sum(x=(u_pred - u_exac) ** 2) / len(self
                    .mesh.vertices)
                loss.backward()
                self.optimizer_Adam.step()
                train_loss += loss.item()
                total_loss += mean_loss
                # log.info("Case index", case_index, "mean_loss", mean_loss.item())
            if self.lr_scheduler and epoch >= 1000:
                self.lr_scheduler.step()
            current_lr = self.optimizer_Adam.get_lr()
            train_loss = train_loss / self.train_samples
            total_loss = total_loss.item() / self.train_samples
            if isinstance(self.problem, Stokes):
                self.net_pde.no_grad = paddle.no_grad
                val_mse, val_mse_bc = self.args.tester.calculate(self.net_pde)
                self.net_pde.no_grad = empty_context_manager
            else:
                with paddle.no_grad():
                    val_mse, val_mse_bc = self.args.tester.calculate(self.net_pde)

            t2 = time.time()
            log.info(
                f'Epoch: {epoch}/{self.epochs_Adam} ,'
                f'time: {t2 - tt:.2f} ,'
                f'lr: {current_lr:.2e} ,'
                f'train_mse: {train_loss:.5f}  ' + f'train_mse_bc: {total_loss:.5f} ,'
                f'val_mse: {val_mse:.5f}  ' + f'val_mse_bc: {val_mse_bc:.5f}'
                )

            tt = time.time()
            if (epoch + 1) % 5 == 0:
                if train_loss < best_train_loss:
                    best_train_loss = train_loss
                    for i in range(self.args.shape[0]):
                        for j in range(self.args.shape[1]):
                            checkpoint_path = os.path.join(self.args.output_dir, f'G{j}_epoch_{epoch}.pdparams')
                            checkpoint = paddle.save(self.net_pde.G[i][j].state_dict(), path=checkpoint_path)
