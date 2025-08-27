import sys
sys.path.append("../external-libraries")
import paddle
from problem import DiffusionReaction
from options import Options
from trainer import Trainer


if __name__ == '__main__':
    args = Options().parse()
    args.pde_case = 'Diffusion'
    args.mesh_path = '../mesh/diffusion/pipe_v2_domain.mphtxt'
    args.boundary_mesh_path = ('../mesh/diffusion/regular_boundary.mphtxt')
    args.problem = DiffusionReaction('../data/diffusion/case2/train_data')

    args.resume = False
    args.save_vtk = False
    
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 2]
    args.blocks_num = [1, 1, 1]
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.layers = [[[6, 24, 24, 24, 1], [6, 24, 24, 24, 1]]]

    args.epochs_Adam = 7000
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 1e-3

    paddle.seed(seed=args.seed)

    trainer = Trainer(args)
    trainer.train()

