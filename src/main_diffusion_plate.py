import os
import sys
import torch
import hydra
from omegaconf import DictConfig

# Αν έχεις εξωτερικές βιβλιοθήκες
sys.path.append("../external-libraries")

from problem import DiffusionReaction          # PyTorch version
from trainer import Trainer                    # PyTorch version
from inference import Tester                  # PyTorch version


# -----------------------------
# SEED
# -----------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------
# MAIN (Hydra)
# -----------------------------
@hydra.main(version_base=None, config_path="../config", config_name="diffusion_plate.yaml")
def main(cfg):

    # ✅ 1. Δημιουργούμε ΤΑ OBECTS ΕΚΤΟΣ cfg
    problem = DiffusionReaction(
        cfg.data.train_data,
        geometry=cfg.pde.geometry
    )

    val_problem = DiffusionReaction(
        cfg.data.test_data,
        geometry=cfg.pde.geometry
    )

    # ✅ 2. Δημιουργούμε ένα "args" object τύπου SimpleNamespace
    from types import SimpleNamespace

    args = SimpleNamespace(
        # --- device ---
        device=cfg.device,
        cuda_index=0,

        # --- PDE ---
        pde_case=cfg.pde.case,
        geometry=cfg.pde.geometry,
        problem=problem,
        val_problem=val_problem,

        # --- Mesh ---
        mesh_path=cfg.mesh.interior_path,
        boundary_mesh_path=cfg.mesh.boundary_path,
        domain=cfg.pde.domain,
        blocks_num = cfg.mesh.blocks_num,
        ngs_boundary=cfg.mesh.ngs_boundary,
        ngs_interior=cfg.mesh.ngs_interior,

        # --- Model ---
        layers=cfg.model.layers,
        shape=cfg.model.shape,
        act=cfg.model.activation,

        # --- Training ---
        epochs_Adam=cfg.training.epochs,
        train_samples=cfg.training.train_samples,
        test_samples=cfg.training.test_samples,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        lr_scheduler=cfg.training.lr_scheduler,
        lr_scheduler_step_size=cfg.training.lr_scheduler_step_size,

        # --- Output ---
        output_dir=cfg.output_dir,
        save_vtk=cfg.vtk.save_vtk,

        # --- Misc ---
        resume=False,
        tol=1e-12,
        tol_change=0.0,
        lam=cfg.training.lam
    )

    # ✅ 3. Δημιουργία Tester & Trainer
    tester = Tester(args)
    args.tester = tester

    trainer = Trainer(args)
    trainer.train()



# -----------------------------
if __name__ == "__main__":
    main()
