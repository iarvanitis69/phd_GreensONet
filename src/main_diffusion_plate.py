import argparse
import torch
import os
from datetime import datetime

# -----------------------------
# ARGUMENTS
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion Plate - PyTorch")

    # === DEVICE: ΤΟ ΒΑΖΕΙΣ ΜΕ ΤΟ ΧΕΡΙ ===
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Επίλεξε χειροκίνητα: cpu ή cuda"
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


# -----------------------------
# SEED
# -----------------------------
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# -----------------------------
# MAIN
# -----------------------------
def main():
    args = parse_args()

    # ✅ ΕΣΥ ΔΙΑΛΕΓΕΙΣ ΣΥΣΚΕΥΗ
    device = torch.device(args.device)

    print("====================================")
    print(f"DEVICE (χειροκίνητο): {device}")
    print(f"SEED: {args.seed}")
    print(f"CONFIG: {args.config}")
    print("====================================")

    # Αν κάποιος βάλει CUDA χωρίς GPU → καθαρό error
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Ζήτησες CUDA αλλά ΔΕΝ υπάρχει διαθέσιμη GPU!")

    set_seed(args.seed)

    # -----------------------------
    # RUN FOLDER (timestamp)
    # -----------------------------
    if args.run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_name = args.run_name

    run_dir = os.path.join("models", run_name)
    os.makedirs(run_dir, exist_ok=True)

    print(f"Run directory: {run_dir}")

    # -----------------------------
    # ΕΔΩ ΘΑ ΜΠΟΥΝ:
    # - φόρτωση YAML
    # - model = FNN(...)
    # - trainer
    # - tester
    # -----------------------------

    # ΠΑΡΑΔΕΙΓΜΑ:
    # model = FNN(...)
    # model.to(device)

    print("Training ξεκινάει...")

    # trainer.train()

    print("Training ολοκληρώθηκε.")


# -----------------------------
if __name__ == "__main__":
    main()
