import sys
import os
import hydra
import paddle
from problem import Stokes
from options import Options
from trainer import Trainer
from Inference import Tester


@hydra.main(config_path="./configs", config_name="stokes.yaml")
def main(cfg):
    args = Options().parse()
    input_dir = cfg.input_dir
    args.pde_case = 'Stokes'
    args.mesh_path = input_dir + '/mesh/stokes/domain.mphtxt'
    args.boundary_mesh_path = input_dir + '/mesh/stokes/boundary.mphtxt'
    args.velocity_component_name = 'x'
    args.problem = Stokes(
        input_dir + '/data/stokes/train_data',
        args.velocity_component_name)
    args.val_problem = Stokes(
        input_dir + '/data/stokes/test_data',
        args.velocity_component_name)
    args.resume = False
    args.save_vtk = False
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.shape = [1, 3]
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.blocks_num = [1, 1, 1]
    args.layers = [[[6, 12, 24, 12, 1], [6, 12, 24, 12, 1], [6, 12, 24, 12, 1]]]
    args.epochs_Adam = 2000
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 1e-3
    args.act = "sin"
    args.lr_scheduler = 'StepDecay'
    args.lr_scheduler_step_size = 100
    args.weight_decay = 0.
    args.output_dir = cfg.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    paddle.seed(seed=args.seed)
    tester = Tester(args)
    args.tester = tester
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()
