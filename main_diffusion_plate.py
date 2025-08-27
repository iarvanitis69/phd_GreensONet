import sys
sys.path.append("../external-libraries")
import paddle
from problem import DiffusionReaction
from options import Options
from trainer import Trainer


if __name__ == '__main__':
    args = Options().parse()
    args.pde_case = 'Diffusion'
    args.mesh_path = '/ssd2/wangguan12/GreensONet_torch/mesh/diffusion/regular_domain.mphtxt'
    args.boundary_mesh_path = ('/ssd2/wangguan12/GreensONet_torch/mesh/diffusion/regular_boundary.mphtxt')
    args.problem = DiffusionReaction('/ssd2/wangguan12/GreensONet_torch/data/diffusion/case1/train_data', geometry="plate")

    args.resume = False
    args.save_vtk = False
    
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 1]
    args.blocks_num = [1, 1, 1]
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.layers = [[[6, 12, 12, 12, 1],]]

    args.epochs_Adam = 1000
    args.train_samples = 40
    args.test_samples = 5
    args.lr = 1e-3

    paddle.seed(seed=args.seed)

    trainer = Trainer(args)
    trainer.train()
