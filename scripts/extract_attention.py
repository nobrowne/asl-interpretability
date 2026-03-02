"""Extract per-head attention weights from a SPOTER checkpoint.

Runs the WLASL100 test set through a trained SPOTER model and captures
attention weights from all 12 MHA modules (6 encoder self-attention +
6 decoder cross-attention) for every sample. Saves results as a single
`.pt` file for downstream interpretability analysis.

Background: PyTorch's TransformerEncoderLayer hardcodes `need_weights=False`,
so we monkey-patch each MHA module's forward method to force
`need_weights=True, average_attn_weights=False` and capture per-head weights
before they're discarded by the surrounding layer logic.

Output format:
    {
        "metadata": {
            "checkpoint": str,
            "epoch": int,
            "num_samples": int,
            "num_encoder_layers": 6,
            "num_decoder_layers": 6,
            "num_heads": 9,
        },
        "samples": [
            {
                "sample_idx": int,
                "label": int,
                "predicted_label": int,
                "correct": bool,
                "seq_len": int,
                "encoder_self_attn": {layer_idx: tensor(9, T, T)},
                "decoder_cross_attn": {layer_idx: tensor(9, 1, T)},
            },
            ...
        ],
    }

Usage (via SLURM — do not run interactively):
    sbatch slurm/analysis.sh extract_attention.py \\
        --checkpoint results/checkpoints/wlasl100_spoter_epoch350.pth \\
        --data_path /path/to/WLASL100_test_25fps.csv
"""

import os
import sys
import logging
import argparse
import functools
import re
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Path setup — mirror train_spoter.py exactly.
# SPOTER uses bare `from augmentations import *` and `from normalization...`
# which require its own directory on sys.path AND as CWD.
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent      # analysis/scripts/
_ANALYSIS_DIR = _SCRIPT_DIR.parent                 # analysis/
_SPOTER_DIR = _ANALYSIS_DIR.parent / "spoter"      # asl-interpretability/spoter/

if not _SPOTER_DIR.exists():
    raise FileNotFoundError(
        f"SPOTER directory not found at {_SPOTER_DIR}. "
        "Make sure it is cloned at ../spoter relative to the analysis repo."
    )

sys.path.insert(0, str(_ANALYSIS_DIR / "stubs"))  # numpy cv2 stub — avoids opencv's bundled OpenSSL (FIPS incompatible)
sys.path.insert(0, str(_SPOTER_DIR))
os.chdir(_SPOTER_DIR)

from datasets.czech_slr_dataset import CzechSLRDataset  # noqa: E402
from spoter.spoter_model import SPOTER                  # noqa: E402


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract per-head attention weights from a SPOTER checkpoint."
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to a .pth checkpoint file (output of train_spoter.py)."
    )
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to the WLASL100 test CSV file."
    )
    parser.add_argument(
        "--output_path", type=str, default="",
        help=(
            "Output .pt file path. Defaults to "
            "results/attention/attention_epoch{N}.pt derived from checkpoint metadata."
        ),
    )
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=108)
    return parser.parse_args()


def setup_logging() -> None:
    """Configure logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def patch_mha(
    mha_module: nn.MultiheadAttention,
    storage: dict[str, torch.Tensor],
    name: str,
) -> None:
    """Monkey-patch one MHA module to capture per-head attention weights.

    PyTorch's TransformerEncoderLayer calls self_attn with `need_weights=False`,
    and SPOTERTransformerDecoderLayer discards the second return value of
    multihead_attn. This patch intercepts the forward call and:
      - Forces `need_weights=True, average_attn_weights=False`
      - Stores the resulting weight tensor in `storage[name]`
      - Returns the full original output tuple unchanged (so surrounding code
        sees the same interface — the decoder layer's [0] slicing still works).

    Args:
        mha_module: The nn.MultiheadAttention instance to patch.
        storage: Shared dict where captured weights are written.
            Key is `name`, value is tensor of shape (num_heads, L, S)
            after squeezing the batch dimension.
        name: Key used to store the captured weights in `storage`.
    """
    original_forward = mha_module.forward

    @functools.wraps(original_forward)
    def patched_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        # Force per-head weight capture regardless of what the caller requested.
        kwargs["need_weights"] = True
        kwargs["average_attn_weights"] = False

        output = original_forward(*args, **kwargs)

        # output[1] shape: (batch_size=1, num_heads, L, S)
        # Squeeze batch dim → (num_heads, L, S)
        storage[name] = output[1].detach().cpu().squeeze(0)

        return output

    mha_module.forward = patched_forward  # type: ignore[method-assign]


def patch_all_attention(
    model: nn.Module,
) -> tuple[dict[str, torch.Tensor], Callable[[], None]]:
    """Monkey-patch all 12 MHA modules in SPOTER to capture per-head weights.

    Patches:
      - model.transformer.encoder.layers[i].self_attn (i = 0..5)
        → keys "enc_{i}"
      - model.transformer.decoder.layers[i].multihead_attn (i = 0..5)
        → keys "dec_{i}"

    Args:
        model: SPOTER model instance (eval mode expected).

    Returns:
        A tuple (storage, unpatch_fn) where:
          - storage: dict populated with weight tensors after each forward pass.
          - unpatch_fn: Callable that restores all original forward methods
            (useful for cleanup / testing).
    """
    storage: dict[str, torch.Tensor] = {}
    originals: list[tuple[nn.MultiheadAttention, Any]] = []

    for i, layer in enumerate(model.transformer.encoder.layers):
        mha = layer.self_attn
        originals.append((mha, mha.forward))
        patch_mha(mha, storage, f"enc_{i}")

    for i, layer in enumerate(model.transformer.decoder.layers):
        mha = layer.multihead_attn
        originals.append((mha, mha.forward))
        patch_mha(mha, storage, f"dec_{i}")

    def unpatch() -> None:
        for mha, original_fwd in originals:
            mha.forward = original_fwd  # type: ignore[method-assign]

    return storage, unpatch


def extract_attention(
    model: nn.Module,
    dataloader: DataLoader,
    storage: dict[str, torch.Tensor],
    device: torch.device,
) -> list[dict[str, Any]]:
    """Run inference on the dataloader and collect per-sample attention dicts.

    Each forward pass populates `storage` (via the monkey-patches installed by
    `patch_all_attention`). After each sample, the captured tensors are copied
    into a per-sample result dict and `storage` is cleared for the next sample.

    Args:
        model: Patched SPOTER model in eval mode.
        dataloader: DataLoader over the test set (batch_size=1 expected).
        storage: Shared dict written to by the MHA patches.
        device: Inference device.

    Returns:
        List of per-sample dicts with keys:
            sample_idx, label, predicted_label, correct, seq_len,
            encoder_self_attn, decoder_cross_attn.
    """
    num_encoder_layers = len(list(model.transformer.encoder.layers))
    num_decoder_layers = len(list(model.transformer.decoder.layers))

    samples = []
    correct = 0

    with torch.no_grad():
        for sample_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            label = int(labels[0].item())

            # Forward pass — triggers patches which write into storage.
            logits = model(inputs)  # (1, 1, num_classes) or (batch, 1, num_classes)
            pred = int(logits.squeeze().argmax().item())
            is_correct = pred == label

            if is_correct:
                correct += 1

            # inputs shape: (1, T, num_landmarks, 2) — T is number of frames.
            # After flatten(start_dim=1) inside SPOTER.forward: (T, 108).
            seq_len = inputs.shape[1]

            # Copy captured tensors out of storage (storage is cleared below).
            enc_attn = {
                i: storage[f"enc_{i}"].clone()
                for i in range(num_encoder_layers)
            }
            dec_attn = {
                i: storage[f"dec_{i}"].clone()
                for i in range(num_decoder_layers)
            }

            samples.append(
                {
                    "sample_idx": sample_idx,
                    "label": label,
                    "predicted_label": pred,
                    "correct": is_correct,
                    "seq_len": seq_len,
                    "encoder_self_attn": enc_attn,
                    "decoder_cross_attn": dec_attn,
                }
            )

            # Clear storage so stale tensors from this sample don't bleed into
            # the next if a patch somehow doesn't fire (shouldn't happen but
            # makes the bug obvious rather than silent).
            storage.clear()

            if (sample_idx + 1) % 50 == 0:
                logging.info(
                    f"  Processed {sample_idx + 1}/{len(dataloader)} samples "
                    f"(running accuracy: {correct / (sample_idx + 1):.3f})"
                )

    return samples


def infer_epoch_from_checkpoint(checkpoint_path: Path) -> int:
    """Extract the epoch number from checkpoint metadata or filename.

    First tries the 'epoch' key in the checkpoint dict (reliable). Falls back
    to a regex parse of the filename (e.g. `wlasl100_spoter_epoch350.pth` → 350).

    Args:
        checkpoint_path: Path to the .pth checkpoint file.

    Returns:
        Epoch number, or 0 if it cannot be determined.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "epoch" in ckpt:
            return int(ckpt["epoch"])
    except Exception:
        pass

    match = re.search(r"epoch(\d+)", checkpoint_path.name)
    return int(match.group(1)) if match else 0


def main() -> None:
    """Arg parsing, model setup, attention extraction, and saving."""
    setup_logging()
    args = get_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_path = Path(args.data_path).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Data CSV not found: {data_path}")

    # -----------------------------------------------------------------------
    # Determine output path. Default: results/attention/attention_epoch{N}.pt
    # -----------------------------------------------------------------------
    epoch = infer_epoch_from_checkpoint(checkpoint_path)
    logging.info(f"Checkpoint: {checkpoint_path} (epoch {epoch})")

    if args.output_path:
        output_path = Path(args.output_path)
    else:
        output_path = _ANALYSIS_DIR / "results" / "attention" / f"attention_epoch{epoch:03d}.pt"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Device — same CUDA_VISIBLE_DEVICES convention as train_spoter.py.
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")
    logging.info(f"Device: {device}")
    if device.type == "cuda":
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------------------------
    # Load model.
    # -----------------------------------------------------------------------
    model = SPOTER(num_classes=args.num_classes, hidden_dim=args.hidden_dim)

    # PyTorch 2.6 compatibility: TransformerDecoder.forward now accesses
    # layer.self_attn.batch_first, but SPOTERTransformerDecoderLayer deletes
    # self_attn in __init__. Restore a minimal stub on each decoder layer.
    import types
    for layer in model.transformer.decoder.layers:
        if not hasattr(layer, "self_attn"):
            layer.self_attn = types.SimpleNamespace(batch_first=False)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.to(device)
    logging.info(
        f"Loaded SPOTER: {sum(p.numel() for p in model.parameters()):,} parameters"
    )

    # -----------------------------------------------------------------------
    # Patch all 12 MHA modules to capture per-head attention.
    # -----------------------------------------------------------------------
    storage, _unpatch = patch_all_attention(model)
    num_encoder_layers = len(list(model.transformer.encoder.layers))
    num_decoder_layers = len(list(model.transformer.decoder.layers))
    num_heads = model.transformer.nhead
    logging.info(
        f"Patched {num_encoder_layers} encoder + {num_decoder_layers} decoder MHA modules "
        f"({num_heads} heads each)"
    )

    # -----------------------------------------------------------------------
    # Load test set (no transform, no augmentation — same as train eval).
    # -----------------------------------------------------------------------
    test_set = CzechSLRDataset(str(data_path))
    # batch_size=1: SPOTER processes variable-length sequences; collation of
    # variable-T sequences would require padding. One-at-a-time is simpler and
    # avoids any attention-mask complications.
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    logging.info(f"Test set: {len(test_set)} samples from {data_path.name}")

    # -----------------------------------------------------------------------
    # Extract attention.
    # -----------------------------------------------------------------------
    logging.info("Starting attention extraction...")
    samples = extract_attention(model, test_loader, storage, device)

    total = len(samples)
    accuracy = sum(s["correct"] for s in samples) / total if total > 0 else 0.0
    logging.info(f"Extraction complete: {total} samples, accuracy = {accuracy:.4f}")

    # -----------------------------------------------------------------------
    # Save.
    # -----------------------------------------------------------------------
    results = {
        "metadata": {
            "checkpoint": str(checkpoint_path),
            "epoch": epoch,
            "num_samples": total,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "num_heads": num_heads,
        },
        "samples": samples,
    }

    torch.save(results, output_path)

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    logging.info(f"Saved to: {output_path} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
