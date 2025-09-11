import os
import sys
import hydra
import paddle
from problem import Poisson
from options import Options
from trainer import Trainer
from Inference import Tester
sys.path.append("../external-libraries")


@hydra.main(version_base=None, config_path="./configs", config_name="poisson.yaml")
def main(cfg):
    args = Options().parse()
    input_dir = cfg.input_dir
    args.pde_case = 'Poisson'
    args.mesh_path = input_dir + '/mesh/poisson/heat_sink_v2_domain.mphtxt'
    args.boundary_mesh_path = input_dir + '/mesh/poisson/heat_sink_v2_boundary.mphtxt'
    args.problem = Poisson(
        input_dir + '/data/poisson/case1/train_data',
        'chip')
    args.val_problem = Poisson(
        input_dir + '/data/poisson/case1/test_data',
        'chip')
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
    args.act = "sin"
    args.lr_scheduler = "StepDecay"
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