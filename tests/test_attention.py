"""Sanity-check tests for the attention extraction pipeline.

Tests use a randomly-initialized SPOTER model (no checkpoint required) so
they can run on CPU without a GPU allocation. They verify structural properties
that must hold regardless of learned weights:

  - Encoder self-attention weights sum to 1 along the attended (key) dimension.
  - Decoder cross-attention weights sum to 1 along the attended (key) dimension.
  - Output shapes match expectations: encoder (9, T, T), decoder (9, 1, T).
  - Attention matrices differ across layers (a common hook bug causes all layers
    to overwrite the same tensor in the storage dict, making all layers identical).
  - No NaN values anywhere.
  - No all-zero attention matrices (would indicate the patch didn't fire or the
    weights collapsed, both of which are bugs).

Run via SLURM:
    sbatch slurm/tests.sh
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Path setup — mirror extract_attention.py / train_spoter.py exactly.
# ---------------------------------------------------------------------------

_TESTS_DIR = Path(__file__).resolve().parent
_ANALYSIS_DIR = _TESTS_DIR.parent
_SPOTER_DIR = _ANALYSIS_DIR.parent / "spoter"

if not _SPOTER_DIR.exists():
    pytest.skip(
        f"SPOTER directory not found at {_SPOTER_DIR}",
        allow_module_level=True,
    )

sys.path.insert(0, str(_ANALYSIS_DIR / "stubs"))  # numpy cv2 stub — avoids opencv's bundled OpenSSL (FIPS incompatible)
sys.path.insert(0, str(_SPOTER_DIR))
os.chdir(_SPOTER_DIR)

from spoter.spoter_model import SPOTER                          # noqa: E402
from scripts.extract_attention import patch_all_attention       # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_CLASSES = 10   # Smaller than production (100) for speed — logic is identical.
HIDDEN_DIM = 108   # Must match SPOTER default.
NUM_HEADS = 9
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6
SEQ_LEN = 30       # Synthetic sequence length — realistic range for WLASL100.
NUM_LANDMARKS = 54  # SPOTER landmark count.


@pytest.fixture(scope="module")
def model_and_storage() -> tuple[SPOTER, dict[str, torch.Tensor]]:
    """Return a randomly-initialized SPOTER model with all MHA modules patched.

    Uses scope='module' so we pay the setup cost once per test module rather
    than once per test function.
    """
    model = SPOTER(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)
    model.eval()
    storage, _ = patch_all_attention(model)
    return model, storage


@pytest.fixture(scope="module")
def enc_attn_tensors(
    model_and_storage: tuple[SPOTER, dict[str, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Run one synthetic forward pass and return encoder attention tensors.

    Returns:
        Dict mapping layer_idx -> tensor(NUM_HEADS, SEQ_LEN, SEQ_LEN).
    """
    model, storage = model_and_storage
    storage.clear()

    # Shape: (batch=1, T, NUM_LANDMARKS, 2)
    dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, 2)

    with torch.no_grad():
        _ = model(dummy_input)

    return {i: storage[f"enc_{i}"].clone() for i in range(NUM_ENCODER_LAYERS)}


@pytest.fixture(scope="module")
def dec_attn_tensors(
    model_and_storage: tuple[SPOTER, dict[str, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    """Return decoder cross-attention tensors from the same forward pass.

    Returns:
        Dict mapping layer_idx -> tensor(NUM_HEADS, 1, SEQ_LEN).
    """
    model, storage = model_and_storage
    # The 'enc_attn_tensors' fixture already ran the forward pass (same scope).
    # Storage was NOT cleared between fixtures, so decoder tensors are already
    # present. Access them directly.
    return {i: storage[f"dec_{i}"].clone() for i in range(NUM_DECODER_LAYERS)}


# ---------------------------------------------------------------------------
# Tests: encoder self-attention
# ---------------------------------------------------------------------------


def test_encoder_attention_shape(
    enc_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Encoder attention tensors should have shape (NUM_HEADS, T, T)."""
    for layer_idx, attn in enc_attn_tensors.items():
        assert attn.shape == (NUM_HEADS, SEQ_LEN, SEQ_LEN), (
            f"Encoder layer {layer_idx}: expected shape "
            f"({NUM_HEADS}, {SEQ_LEN}, {SEQ_LEN}), got {tuple(attn.shape)}"
        )


def test_encoder_attention_sums_to_one(
    enc_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Each query position's attention distribution should sum to 1.

    Attention weights are a probability distribution over key positions,
    so summing over the last dimension (keys) should give 1 for each
    (head, query) pair.
    """
    for layer_idx, attn in enc_attn_tensors.items():
        row_sums = attn.sum(dim=-1)  # (NUM_HEADS, SEQ_LEN)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Encoder layer {layer_idx}: attention rows do not sum to 1. "
            f"Max deviation: {(row_sums - 1).abs().max():.2e}"
        )


def test_encoder_attention_no_nan(
    enc_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Encoder attention tensors should contain no NaN values."""
    for layer_idx, attn in enc_attn_tensors.items():
        assert not torch.isnan(attn).any(), (
            f"Encoder layer {layer_idx}: attention contains NaN values"
        )


def test_encoder_attention_not_all_zero(
    enc_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Encoder attention tensors should not be all-zero.

    An all-zero matrix indicates the patch didn't fire or weights collapsed.
    """
    for layer_idx, attn in enc_attn_tensors.items():
        assert attn.abs().sum() > 0, (
            f"Encoder layer {layer_idx}: attention is all-zero"
        )


def test_encoder_attention_differs_across_layers(
    enc_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Attention tensors must differ across layers.

    A common hook/patch bug: all layers write into the same storage slot,
    so all layers end up with identical tensors (the last layer's output).
    This test catches that failure mode.
    """
    tensors = list(enc_attn_tensors.values())
    for i in range(len(tensors) - 1):
        assert not torch.allclose(tensors[i], tensors[i + 1]), (
            f"Encoder layers {i} and {i + 1} have identical attention tensors — "
            "possible storage aliasing bug in patch_all_attention()"
        )


# ---------------------------------------------------------------------------
# Tests: decoder cross-attention
# ---------------------------------------------------------------------------


def test_decoder_attention_shape(
    dec_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Decoder cross-attention tensors should have shape (NUM_HEADS, 1, T).

    The decoder query is a single class token (tgt_len=1) attending over
    all T encoder frames (src_len=T).
    """
    for layer_idx, attn in dec_attn_tensors.items():
        assert attn.shape == (NUM_HEADS, 1, SEQ_LEN), (
            f"Decoder layer {layer_idx}: expected shape "
            f"({NUM_HEADS}, 1, {SEQ_LEN}), got {tuple(attn.shape)}"
        )


def test_decoder_attention_sums_to_one(
    dec_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Each decoder query's cross-attention should sum to 1 over encoder frames."""
    for layer_idx, attn in dec_attn_tensors.items():
        row_sums = attn.sum(dim=-1)  # (NUM_HEADS, 1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Decoder layer {layer_idx}: attention rows do not sum to 1. "
            f"Max deviation: {(row_sums - 1).abs().max():.2e}"
        )


def test_decoder_attention_no_nan(
    dec_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Decoder cross-attention tensors should contain no NaN values."""
    for layer_idx, attn in dec_attn_tensors.items():
        assert not torch.isnan(attn).any(), (
            f"Decoder layer {layer_idx}: attention contains NaN values"
        )


def test_decoder_attention_not_all_zero(
    dec_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Decoder cross-attention tensors should not be all-zero."""
    for layer_idx, attn in dec_attn_tensors.items():
        assert attn.abs().sum() > 0, (
            f"Decoder layer {layer_idx}: attention is all-zero"
        )


def test_decoder_attention_differs_across_layers(
    dec_attn_tensors: dict[int, torch.Tensor],
) -> None:
    """Decoder cross-attention tensors must differ across layers."""
    tensors = list(dec_attn_tensors.values())
    for i in range(len(tensors) - 1):
        assert not torch.allclose(tensors[i], tensors[i + 1]), (
            f"Decoder layers {i} and {i + 1} have identical attention tensors — "
            "possible storage aliasing bug in patch_all_attention()"
        )


# ---------------------------------------------------------------------------
# Tests: patch correctness
# ---------------------------------------------------------------------------


def test_patch_does_not_alter_model_output() -> None:
    """Patching MHA modules must not change model predictions.

    The monkey-patch only adds weight capture as a side effect — it must not
    alter the actual output tensor returned by each MHA or the final logits.
    """
    model_unpatched = SPOTER(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)
    model_patched = SPOTER(num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM)

    # Copy weights so both models are identical.
    model_patched.load_state_dict(model_unpatched.state_dict())

    model_unpatched.eval()
    model_patched.eval()

    patch_all_attention(model_patched)

    dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, 2)

    with torch.no_grad():
        out_unpatched = model_unpatched(dummy_input)
        out_patched = model_patched(dummy_input)

    assert torch.allclose(out_unpatched, out_patched, atol=1e-6), (
        "Patched model produces different logits than unpatched model — "
        "patch is incorrectly modifying the forward pass"
    )


def test_all_12_modules_captured(
    model_and_storage: tuple[SPOTER, dict[str, torch.Tensor]],
) -> None:
    """All 12 MHA modules (enc_0..5, dec_0..5) must appear in storage after one pass."""
    model, storage = model_and_storage
    storage.clear()

    dummy_input = torch.randn(1, SEQ_LEN, NUM_LANDMARKS, 2)
    with torch.no_grad():
        _ = model(dummy_input)

    expected_keys = {f"enc_{i}" for i in range(NUM_ENCODER_LAYERS)} | {
        f"dec_{i}" for i in range(NUM_DECODER_LAYERS)
    }
    assert set(storage.keys()) == expected_keys, (
        f"Storage keys mismatch.\n"
        f"  Expected: {sorted(expected_keys)}\n"
        f"  Got:      {sorted(storage.keys())}"
    )
