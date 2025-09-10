import os
import sys
import hydra
import paddle
from options import Options
from trainer import Trainer
from Inference import Tester
from problem import DiffusionReaction

sys.path.append("../external-libraries")


@hydra.main(config_path="./configs", config_name="diffusion_pipe.yaml")
def main(cfg):
    input_dir = cfg.input_dir
    args = Options().parse()
    WORK_DIR = "../GON_jcp"
    args.pde_case = 'Diffusion'
    args.geometry = "pipe"
    args.mesh_path = WORK_DIR + '/mesh/diffusion/regular_domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/diffusion/regular_boundary.mphtxt'
    args.problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case2/train_data',
        geometry=args.geometry
        )
    args.val_problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case2/test_data',
        geometry=args.geometry)

    args.resume = False
    args.save_vtk = False
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 2]
    args.blocks_num = [1, 1, 1]
    args.domain = [-5, 5, -5, 5, -5, 5]
    args.layers = [[[6, 24, 24, 24, 1], [6, 24, 24, 24, 1]]]
    args.epochs_Adam = 10000
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 5e-3
    args.lr_scheduler = 'StepDecay'
    args.output_dir = cfg.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    paddle.seed(seed=args.seed)
    args.lr_scheduler_step_size = 200
    tester = Tester(args)
    args.tester = tester
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
