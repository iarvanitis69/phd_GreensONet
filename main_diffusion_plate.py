from problem import DiffusionReaction
from options import Options
from trainer import Trainer
from Inference import Tester
from datetime import datetime
import sys
import os
import hydra
import paddle

sys.path.append("../external-libraries")

@hydra.main(version_base=None, config_path="./configs", config_name="diffusion_plate.yaml")
def main(cfg):
    WORK_DIR = cfg.input_dir
    args = Options().parse()
    args.pde_case = 'Diffusion'
    args.geometry = "plate"
    args.mesh_path = WORK_DIR + '/mesh/diffusion/regular_domain.mphtxt'
    args.boundary_mesh_path = WORK_DIR + '/mesh/diffusion/regular_boundary.mphtxt'
    args.problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case1/train_data',
        geometry=args.geometry)
    args.val_problem = DiffusionReaction(
        WORK_DIR + '/data/diffusion/case1/test_data',
        geometry=args.geometry)

    args.save_vtk = False
    args.ngs_boundary = 3
    args.ngs_interior = 4
    args.seed = 3407
    args.shape = [1, 2]
    args.blocks_num = [1, 1, 1]
    args.domain = [-1, 1, -1, 1, -1, 1]
    args.layers = [[[6, 12, 12, 12, 1], [6, 12, 12, 12, 1]]]
    args.epochs_Adam = 5000
    args.train_samples = 15
    args.test_samples = 5
    args.lr = 1e-3
    args.lr_scheduler = "const"
    args.output_dir = cfg.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    paddle.seed(seed=args.seed)
    tester = Tester(args)
    args.tester = tester
    trainer = Trainer(args)
    trainer.train()

if __name__ == '__main__':
    main()