import os
import re
import sys
import time
sys.path.append("../external-libraries")
import paddle
import random
import meshio
import numpy as np
from paddle.io import Dataset
from options import Options
from model import Net_Integral, tile
from problem import DiffusionReaction, Poisson, Stokes
from utils import Mesh


def set_random_seed(seed):
    paddle.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
set_random_seed(0)

def extract_integers_from_string(input_string_list):
    pattern = re.compile(r'\d+')
    return [int(pattern.findall(s)[0]) for s in input_string_list]


class VTK_Dataset(Dataset):
    def __init__(self, args, mesh, problem, train_samples=None):
        self.args = args
        self.problem = problem
        self.file_list = os.listdir(problem.data_folder)
        self.case_index_list = extract_integers_from_string(self.file_list)
        self.case_index_list.sort()
        self.case_index_list = self.case_index_list[:train_samples]
        self.mesh = mesh
        self.z_blocks = self.mesh.z_blocks
        self.x_in_wts = paddle.to_tensor(self.mesh.X_interior['wts'], dtype='float32')
        self.x_in_coord = paddle.to_tensor(self.mesh.X_interior['coord'], dtype='float32')
        self.x_bc_wts = paddle.to_tensor(self.mesh.X_boundary['wts'], dtype='float32')
        self.x_bc_coord = paddle.to_tensor(self.mesh.X_boundary['coord'], dtype='float32')
        self.x_bc_normal = self.mesh.X_boundary['normal'].astype("float32")
        self.z_blocks_coord = self.mesh.z_blocks[0]['coord'].astype("float32")
        if "boundary_type" in self.mesh.X_boundary.keys():
            self.input_boundary_type = self.mesh.X_boundary['boundary_type']
            if not isinstance(self.input_boundary_type, paddle.Tensor):
                self.input_boundary_type = paddle.to_tensor(self.input_boundary_type).astype('float32')

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

        if isinstance(self.problem, Stokes):
            input_boundary_type = tile(self.input_boundary_type, self.z_blocks_coord)[:, 0]
            input_boundary_list = []
            for i in range(self.args.ngs_boundary):
                input_boundary_list.append(tile(self.x_bc_coord[i], self.z_blocks_coord)[:, :3])
        else:
            input_boundary_type = case_index
            input_boundary_list = self.x_bc_coord

        for i in range(self.args.ngs_interior):
            x_in_coord_list.append(self.x_in_coord[i])
            x_in_wts_list.append(self.x_in_wts[i])
            f_interior = self.problem.f(self.x_in_coord[i], case_index)
            if isinstance(self.problem, Stokes):
                f_interior = f_interior.reshape([N_interior, -1])
            f_interior_list.append(f_interior)

        for i in range(self.args.ngs_boundary):
            x_bc_wts_list.append(self.x_bc_wts[i])
            x_bc_coord_list.append(self.x_bc_coord[i])
            g_boundary = self.problem.g(input_boundary_list[i], input_boundary_type)
            a_boundary = self.problem.a(input_boundary_list[i])
            if isinstance(self.problem, Stokes):
                g_boundary = g_boundary.reshape([N_boundary, -1])
                a_boundary = a_boundary.reshape([N_boundary, -1])
            g_boundary_list.append(g_boundary)
            a_boundary_list.append(a_boundary)
        return x_in_coord_list, x_in_wts_list, x_bc_coord_list, x_bc_wts_list, f_interior_list, g_boundary_list, a_boundary_list, self.z_blocks, case_index, self.x_bc_normal

    def __len__(self):
        return len(self.case_index_list)


class Tester():
    def __init__(self, args):
        self.args = args
        self.val_problem = args.val_problem
        self.mesh = Mesh(
            args.mesh_path, 
            args.boundary_mesh_path, 
            args.domain, 
            args.blocks_num,
            ngs_boundary=self.args.ngs_boundary, 
            ngs_interior=self.args.ngs_interior
        )
        self.loss_function = paddle.nn.MSELoss()
        self.net_pde = Net_Integral(
            args.layers, args.shape, args.ngs_boundary, args.ngs_interior, args.val_problem, args.act, eval_mode=True)
        # load the pre-trained model from checkpoint
        if hasattr(self.args, "checkpoint_path"):
            for i in range(args.shape[0]):
                for j in range(args.shape[1]):
                    print("Loading the pre-trained model from : ", self.args.checkpoint_path[j])
                    if self.args.checkpoint_path[j].endswith("pdparams"):
                        state_dict = paddle.load(path=str(self.args.checkpoint_path[j]))
                    elif self.args.checkpoint_path[j].endswith("npy"):
                        checkpoint = np.load(str(self.args.checkpoint_path[j]), allow_pickle=True).item()
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
            print("Initialize Evaluator")

    def calculate(self, model_val=None):
        for K in range(len(self.mesh.blocks)):
            if self.mesh.z_blocks[K] is not None and len(self.mesh.z_blocks[K]) > 0:
                self.mesh.z_blocks[K]['coord'] = paddle.to_tensor(self.mesh.z_blocks[K]['coord'], dtype='float32')
                mse, mse_bc = self.calculate_block(K, model_val)
            else:
                raise ValueError(f'Block #{K} is empty!!!')
        return np.mean(mse), np.mean(mse_bc)

    def calculate_block(self, k, model_val=None):
        loss = 0
        loss_list = []
        mse_bc_list = []

        if not model_val:
            model = self.net_pde
        else:
            model = model_val
        test_dataset = VTK_Dataset(self.args, self.mesh, self.val_problem)

        for data in test_dataset:
            x_in_coord_list, x_in_wts_list, x_bc_coord_list, x_bc_wts_list, f_interior_list, g_boundary_list, a_boundary_list, z_blocks, case_index, x_bc_normal = data 
            coord = z_blocks[k]['coord']
            tt = time.time()
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
            
            u_exac = self.val_problem.u_exact(coord.numpy(), case_index)
            u_exac = paddle.to_tensor(u_exac, dtype='float32')
            loss = self.loss_function(u_pred, u_exac)
            
            mse_bc = paddle.sum((u_pred - u_exac)**2) / len(self.mesh.vertices)
            # if not model_val:
                # print(f'Case {case_index}, t = {(time.time() - tt):.2f} s, mse loss: {loss.item():.4f}, mse_bc loss: {mse_bc.item():.6f}')
            
            if self.args.save_vtk is True:
                self.save_to_vtk(case_index, coord.numpy(), u_pred.numpy(), u_exac.numpy())
                
            loss_list.append(loss.item())
            mse_bc_list.append(mse_bc.item())

        if not model_val:
            print(f'MSE loss over [{len(test_dataset)}] test cases {(sum(loss_list) / len(loss_list)):.2e}')
            print(f'MSE with BC loss over [{len(test_dataset)}] test cases {(sum(mse_bc_list) / len(mse_bc_list)):.2e}')
        return loss_list, mse_bc_list

    def save_to_vtk(self, case_index, coords, u_pred, u_exac):
        error = np.abs(u_pred.numpy()-u_exac.numpy())
        sorted_data = []
        for node in self.mesh.vertices:
            if any(np.allclose(node, coord) for coord in coords):
                index = np.where(np.all(np.isclose(coords,node), axis=1))[0][0]
                sorted_data.append([node[0], node[1], node[2], u_pred[index][0], u_exac[index][0], error[index][0]])
            else:
                res = self.val_problem.u_exact(node[np.newaxis, ...], case_index)
                boundary_point_value = res
                sorted_data.append([node[0], node[1], node[2],
                    boundary_point_value[0][0], boundary_point_value[0][0], 0])
        sorted_data = np.array(sorted_data)
        mesh = meshio.Mesh(points=self.mesh.vertices, cells=[('tetra', self
            .mesh.elements)], point_data={'u_pred': sorted_data[:, 3],
            'u_exac': sorted_data[:, 4], 'u_error': sorted_data[:, 5]})
        mesh.write(f'../output/vtk_in_paraview/test_case{case_index + 1}_Greenonet.vtu')


if __name__ == '__main__':
    args = Options().parse()
    WORK_DIR = "../GON_jcp"

    print("\n[Case 1: Heterogeneous reaction-diffusion equations] [name:Flat Plate]")
    args.pde_case = 'Diffusion'
    args.mesh_path = WORK_DIR + '/mesh/diffusion/regular_domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/diffusion/regular_boundary.mphtxt'
    args.checkpoint_path = [
        './output/GreensONet/Diffusion_Plate/train/2025-09-10-03-44-36/G0_epoch_1999.pdparams',
        './output/GreensONet/Diffusion_Plate/train/2025-09-10-03-44-36/G1_epoch_1999.pdparams']
    args.val_problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case1/test_data',
        geometry="plate")

    args.save_vtk = False
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 2]
    args.blocks_num = [1, 1, 1]
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.layers = [[[6, 12, 12, 12, 1], [6, 12, 12, 12, 1]]]
    args.act = "sin"
    tester = Tester(args)
    with paddle.no_grad():
        tester.calculate()

    print("\n[Case 2: Steady heat conduction equations] [name : Poisson Equation]")
    args = Options().parse()
    args.pde_case = 'Poisson'
    args.val_problem = Poisson(
        WORK_DIR + '/data/poisson/case1/test_data',
        geometry="chips")
    args.mesh_path = WORK_DIR + '/mesh/poisson/heat_sink_v2_domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/poisson/heat_sink_v2_boundary.mphtxt'
    args.checkpoint_path =[
        './output/GreensONet/Poisson/train/2025-09-08-07-04-17/G0_epoch_1584.pdparams',
        './output/GreensONet/Poisson/train/2025-09-08-07-04-17/G1_epoch_1584.pdparams']
    args.save_vtk = False
    args.domain = [0, 1, 0, 1, 0, 1]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 2]
    args.ngs_boundary = 1
    args.ngs_interior = 1
    args.test_samples = 21
    args.act = "sin"
    paddle.seed(seed=args.seed)
    args.layers = [[[6, 12, 12, 12, 1], [6, 12, 12, 12, 1]]]
    tester = Tester(args)
    with paddle.no_grad():
        tester.calculate()

    print("\n[Case 3: Stokes equations] [name : 3D lid-driven cavity]")
    args.mesh_path = WORK_DIR + '/mesh/stokes/domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/stokes/boundary.mphtxt'
    args.checkpoint_path =[
        './output/GreensONet/Stokes/train/2025-09-09-06-46-06/G0_epoch_1979.pdparams',
        './output/GreensONet/Stokes/train/2025-09-09-06-46-06/G1_epoch_1979.pdparams',
        './output/GreensONet/Stokes/train/2025-09-09-06-46-06/G2_epoch_1979.pdparams']
    args.velocity_component_name = "x"
    args.val_problem = Stokes(WORK_DIR + '/data/stokes/test_data', args.velocity_component_name)
    args.save_vtk = False
    args.domain = [0, 1, 0, 1, 0, 1]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 3]
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.test_samples = 5
    args.act = "sin"
    args.layers = [[[6, 12, 24, 12, 1], [6, 12, 24, 12, 1], [6, 12, 24, 12, 1]]]
    tester = Tester(args)
    tester.calculate()

    print("\n[Case 4: Diffusion equations] [name : pipe]")
    args.geometry = 'pipe'
    args.mesh_path = WORK_DIR + '/mesh/diffusion/pipe_v2_domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/diffusion/pipe_v2_boundary.mphtxt'
    args.checkpoint_path =[
        './output/GreensONet/Diffusion_Pipe/train/2025-09-10-08-13-36/G0_epoch_9999.pdparams',
        './output/GreensONet/Diffusion_Pipe/train/2025-09-10-08-13-36/G1_epoch_9999.pdparams']
    args.val_problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case2/test_data',
        geometry=args.geometry
        )
    args.save_vtk = False
    args.domain = [-5, 5, -5, 5, -5, 5]
    args.blocks_num = [1, 1, 1]
    args.shape = [1, 2]
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.test_samples = 5
    args.act = "relu"
    args.layers = [[[6, 14, 24, 24, 1], [6, 24, 24, 24, 1]]]
    tester = Tester(args)
    tester.calculate()
 