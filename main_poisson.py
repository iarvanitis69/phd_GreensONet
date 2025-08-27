import sys
sys.path.append("../external-libraries")
import paddle
from problem import Poisson
from options import Options
from trainer import Trainer


if __name__ == '__main__':
    args = Options().parse()
    args.pde_case = 'Poisson'
    args.mesh_path = '../mesh/poisson/heat_sink_v2_domain.mphtxt'
    args.boundary_mesh_path = ('../mesh/poisson/heat_sink_v2_boundary.mphtxt')
    args.problem = Poisson('../data/poisson/case1/train_data')

    args.resume = False
    args.save_vtk = False
    
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 2]
    args.blocks_num = [1, 1, 1]
    args.domain = [0, 1, 0, 1, 0, 1]
    args.layers = [[[6, 12, 12, 12, 1], [6, 12, 12, 12, 1]]]

    args.epochs_Adam = 2000
    args.train_samples = 80
    args.test_samples = 5
    args.lr = 1e-3

    paddle.seed(seed=args.seed)

    trainer = Trainer(args)
    trainer.train()

