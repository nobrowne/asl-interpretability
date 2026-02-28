"""Training wrapper for SPOTER on WLASL100.

Trains the SPOTER model (Bohacek & Hruz, WACV 2022) on the WLASL100 dataset
and saves epoch-specific checkpoints at fixed intervals for interpretability
analysis. All checkpoints include both model weights and training metadata.

Checkpoint epochs: [10, 50, 100, 200, 350]
  - Epoch 10:  Early training — model learning basic structure
  - Epoch 50:  Mid-early — representations beginning to stabilize
  - Epoch 100: Mid-training — most rapid accuracy gain should be past
  - Epoch 200: Late training — representations mostly converged
  - Epoch 350: Final — ~63% expected top-1 accuracy on WLASL100

Note on dataset class name: SPOTER uses `CzechSLRDataset` as a general-purpose
skeleton-data CSV loader. Despite the name, when given WLASL100 CSV files it
loads American Sign Language data. The class was originally developed at the
University of West Bohemia for Czech SLR and reused across all SPOTER datasets.

Usage (via SLURM — do not run interactively):
    python scripts/train_spoter.py \\
        --training_set_path /path/to/WLASL100_train.csv \\
        --validation_set_path /path/to/WLASL100_val.csv \\
        --testing_set_path /path/to/WLASL100_test.csv
"""

import os
import sys
import logging
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms


# ---------------------------------------------------------------------------
# Path setup — must happen before any spoter imports.
# SPOTER uses bare `from augmentations import *` and
# `from normalization.xxx import ...` which require its own directory to be
# both on sys.path and to be the CWD.
# ---------------------------------------------------------------------------

# Resolve absolute paths BEFORE any os.chdir calls.
_SCRIPT_DIR = Path(__file__).resolve().parent          # analysis/scripts/
_ANALYSIS_DIR = _SCRIPT_DIR.parent                     # analysis/
_SPOTER_DIR = _ANALYSIS_DIR.parent / "spoter"          # asl-interpretability/spoter/

if not _SPOTER_DIR.exists():
    raise FileNotFoundError(
        f"SPOTER directory not found at {_SPOTER_DIR}. "
        "Make sure it is cloned at ../spoter relative to the analysis repo."
    )

sys.path.insert(0, str(_SPOTER_DIR))
os.chdir(_SPOTER_DIR)  # Required so bare imports (`augmentations`, `normalization`) resolve.

# Spoter imports (after path setup).
from datasets.czech_slr_dataset import CzechSLRDataset  # noqa: E402
from spoter.spoter_model import SPOTER                  # noqa: E402
from spoter.utils import train_epoch, evaluate          # noqa: E402
from spoter.gaussian_noise import GaussianNoise         # noqa: E402


# Epochs at which we always save a checkpoint, regardless of accuracy.
CHECKPOINT_EPOCHS = [10, 50, 100, 200, 350]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments for the training run."""
    parser = argparse.ArgumentParser(
        description="Train SPOTER on WLASL100 with epoch-specific checkpointing."
    )

    # Data paths
    parser.add_argument(
        "--training_set_path", type=str, required=True,
        help="Absolute path to the WLASL100 training CSV file."
    )
    parser.add_argument(
        "--validation_set_path", type=str, default="",
        help="Absolute path to the WLASL100 validation CSV file (optional)."
    )
    parser.add_argument(
        "--testing_set_path", type=str, default="",
        help="Absolute path to the WLASL100 test CSV file (optional)."
    )

    # Model hyperparameters (SPOTER paper defaults for WLASL100)
    parser.add_argument("--num_classes", type=int, default=100,
                        help="Number of sign classes (100 for WLASL100).")
    parser.add_argument("--hidden_dim", type=int, default=108,
                        help="Transformer hidden dimension.")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=350,
                        help="Total number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="SGD learning rate.")
    parser.add_argument("--seed", type=int, default=379,
                        help="Random seed for reproducibility.")
    parser.add_argument("--scheduler_factor", type=float, default=0.1,
                        help="ReduceLROnPlateau reduction factor.")
    parser.add_argument("--scheduler_patience", type=int, default=5,
                        help="ReduceLROnPlateau patience (epochs).")

    # Gaussian noise augmentation (applied to training input)
    parser.add_argument("--gaussian_mean", type=float, default=0.0)
    parser.add_argument("--gaussian_std", type=float, default=0.001)

    # Output
    parser.add_argument(
        "--experiment_name", type=str, default="wlasl100_spoter",
        help="Name prefix for checkpoint files and log file."
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="Directory to save checkpoints. Defaults to analysis/results/checkpoints/."
    )

    return parser.parse_args()


def setup_logging(log_path: Path) -> None:
    """Configure logging to both file and stdout.

    Args:
        log_path: Path to the output log file.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(str(log_path)),
            logging.StreamHandler(sys.stdout),
        ],
    )


def set_seeds(seed: int) -> torch.Generator:
    """Seed all random number generators for reproducibility.

    Args:
        seed: Integer seed value.

    Returns:
        A seeded torch.Generator for use with DataLoader.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)
    return g


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_acc: float,
    val_acc: float,
    args: argparse.Namespace,
) -> None:
    """Save a training checkpoint with full metadata.

    Saves state_dict (not the full model object) for portability — the SPOTER
    class definition just needs to be importable when loading.

    Args:
        path: Full file path for the checkpoint (.pth).
        epoch: Current epoch number.
        model: SPOTER model instance.
        optimizer: Current optimizer state (for potential resume).
        train_acc: Training accuracy at this epoch.
        val_acc: Validation accuracy at this epoch (0.0 if no val set).
        args: All training arguments (saved for reproducibility).
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_acc": train_acc,
            "val_acc": val_acc,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    """Run the full SPOTER training loop with epoch-specific checkpointing."""
    args = get_args()
    g = set_seeds(args.seed)

    # Resolve output directories using the pre-chdir absolute paths.
    checkpoint_dir = (
        Path(args.output_dir) if args.output_dir
        else _ANALYSIS_DIR / "results" / "checkpoints"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = _ANALYSIS_DIR / "results" / "logs" / f"{args.experiment_name}.log"
    setup_logging(log_path)

    # -----------------------------------------------------------------------
    # Device — use CUDA only when SLURM has allocated a GPU. Checking
    # CUDA_VISIBLE_DEVICES avoids accidentally running on a GPU node's
    # default device when no GPU was requested.
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)
    model.train(True)
    model.to(device)
    logging.info(
        f"SPOTER: {sum(p.numel() for p in model.parameters()):,} parameters, "
        f"num_classes={args.num_classes}, hidden_dim={args.hidden_dim}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
    )

    # -----------------------------------------------------------------------
    # Data loaders
    # -----------------------------------------------------------------------
    transform = transforms.Compose([GaussianNoise(args.gaussian_mean, args.gaussian_std)])
    train_set = CzechSLRDataset(args.training_set_path, transform=transform, augmentations=True)
    train_loader = DataLoader(train_set, shuffle=True, generator=g)
    logging.info(f"Training set: {len(train_set)} samples")

    val_loader = None
    if args.validation_set_path:
        val_set = CzechSLRDataset(args.validation_set_path)
        val_loader = DataLoader(val_set, shuffle=False, generator=g)
        logging.info(f"Validation set: {len(val_set)} samples")

    eval_loader = None
    if args.testing_set_path:
        eval_set = CzechSLRDataset(args.testing_set_path)
        eval_loader = DataLoader(eval_set, shuffle=False, generator=g)
        logging.info(f"Test set: {len(eval_set)} samples")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    logging.info(
        f"Starting {args.experiment_name}: {args.epochs} epochs, "
        f"lr={args.lr}, seed={args.seed}"
    )
    logging.info(f"Epoch checkpoints scheduled at: {CHECKPOINT_EPOCHS}")
    logging.info(f"Checkpoints will be saved to: {checkpoint_dir}")

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        # --- Train one epoch ---
        train_loss, _, _, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        avg_loss = train_loss.item() / len(train_loader)

        # --- Validate ---
        val_acc = 0.0
        if val_loader:
            model.train(False)
            _, _, val_acc = evaluate(model, val_loader, device)
            model.train(True)
            # Step scheduler on average training loss (no separate val loss available).
            scheduler.step(avg_loss)

        logging.info(
            f"[{epoch:3d}/{args.epochs}] loss={avg_loss:.4f}  "
            f"train={train_acc:.4f}  val={val_acc:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        # --- Epoch-specific checkpoint (always saved at fixed epochs) ---
        if epoch in CHECKPOINT_EPOCHS:
            ckpt_name = f"{args.experiment_name}_epoch{epoch:03d}.pth"
            ckpt_path = checkpoint_dir / ckpt_name
            save_checkpoint(ckpt_path, epoch, model, optimizer, train_acc, val_acc, args)
            logging.info(f"  -> Saved scheduled checkpoint: {ckpt_name}")

        # --- Best-validation checkpoint ---
        if val_loader and val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = checkpoint_dir / f"{args.experiment_name}_best.pth"
            save_checkpoint(best_path, epoch, model, optimizer, train_acc, val_acc, args)
            logging.info(
                f"  -> New best val_acc={val_acc:.4f} at epoch {epoch}, "
                f"saved {best_path.name}"
            )

    # -----------------------------------------------------------------------
    # Final test evaluation using best checkpoint
    # -----------------------------------------------------------------------
    if eval_loader:
        logging.info("Running final evaluation on test set...")
        best_path = checkpoint_dir / f"{args.experiment_name}_best.pth"
        if best_path.exists():
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
        model.train(False)
        _, _, test_acc = evaluate(model, eval_loader, device, print_stats=True)
        logging.info(f"Test accuracy (best-val checkpoint): {test_acc:.4f}")

    logging.info("Training complete.")
    logging.info(f"All checkpoints in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
