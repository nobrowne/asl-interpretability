# Technical Decisions

## 2026-02-28 — Training wrapper as separate script, not patch to train.py
**Context:** Needed to add epoch-specific checkpoint saving to SPOTER's training loop.
**Decision:** Write `scripts/train_spoter.py` as a standalone wrapper that imports from the spoter clone rather than modifying `spoter/train.py`.
**Rationale:** Keeps the spoter clone clean and unmodified, making it easy to pull upstream changes or diff against the original. All project deliverables stay in the analysis repo.
**Alternatives considered:** Patching train.py directly (simpler but pollutes the clone); subprocess call to train.py (less control, can't add checkpoint hooks).

## 2026-02-28 — Epoch-specific checkpoints at [10, 50, 100, 200, 350]
**Context:** Interpretability analysis requires checkpoints at multiple training stages to study how representations evolve.
**Decision:** Save checkpoints unconditionally at epochs 10, 50, 100, 200, 350, plus a rolling best-validation checkpoint.
**Rationale:** Epoch 10 = early structure; 50 = stabilizing; 100 = past rapid gain; 200 = mostly converged; 350 = final. Covers the full training arc with log-ish spacing. Best-val checkpoint is used for final test evaluation.
**Alternatives considered:** Every-N-epochs (more disk use, less targeted); only best (loses training dynamics).

## 2026-02-28 — state_dict checkpoints, not full model saves
**Context:** Original SPOTER saves full model objects with `torch.save(model, path)`.
**Decision:** Save `{"epoch": ..., "model_state_dict": ..., "optimizer_state_dict": ..., "args": ...}` instead.
**Rationale:** state_dict is portable across PyTorch versions and doesn't require the exact class to be importable at load time. Metadata (epoch, accuracies, hyperparams) in the same file aids reproducibility.
**Alternatives considered:** Full model save (convenient but fragile across versions).

## 2026-02-28 — CUDA device via CUDA_VISIBLE_DEVICES, not torch.cuda.is_available()
**Context:** Setting device for training.
**Decision:** `torch.device("cuda" if "CUDA_VISIBLE_DEVICES" in os.environ else "cpu")`.
**Rationale:** On SLURM, `CUDA_VISIBLE_DEVICES` is set by the scheduler when a GPU is allocated. This prevents accidentally using a GPU when none was requested, and is more explicit about intent than probing hardware availability.
**Alternatives considered:** `torch.cuda.is_available()` (common default but can grab GPUs opportunistically).

## 2026-02-28 — Attention extraction via MHA monkey-patching, not forward hooks
**Context:** Need per-head attention weights from all 12 MHA modules in SPOTER.
**Decision:** Monkey-patch each `nn.MultiheadAttention.forward` to force `need_weights=True, average_attn_weights=False`, capture the weight tensor in a shared storage dict, and return the original output unchanged.
**Rationale:** PyTorch's `TransformerEncoderLayer` hardcodes `need_weights=False` when calling `self_attn` — standard forward hooks on the MHA module only capture *output* tensors, not the internal weight tensors (which aren't in the return value when `need_weights=False`). Patching at the `forward` level lets us override the kwarg before softmax, so we get the per-head weights. The patch returns the full original tuple, so `SPOTERTransformerDecoderLayer`'s `[0]` indexing still works transparently.
**Alternatives considered:** PyTorch's `register_forward_hook` (can't intercept weights not in the output); modifying SPOTER source (violates "don't modify the clone" policy).

## 2026-02-28 — batch_size=1 for attention extraction DataLoader
**Context:** Loading the WLASL100 test set for attention extraction.
**Decision:** Use `batch_size=1` — process one sample at a time.
**Rationale:** WLASL100 sequences have variable frame length T. Batching variable-length sequences requires padding, which would cause attention to distribute mass over padding positions, corrupting the per-head analysis. One-at-a-time avoids this entirely, and the test set (~250 samples) is small enough that throughput is not a concern.
**Alternatives considered:** Variable-length batching with masking (complex, error-prone, unnecessary for ~250 samples).

## 2026-02-28 — results/ symlinked to ~/nobackup/autodelete/
**Context:** Checkpoints (~250MB), attention weights (~200MB/epoch), and other large outputs must not fill home directory quota.
**Decision:** `analysis/results` is a symlink to `~/nobackup/autodelete/asl-interpretability/results/`. All scripts write to `results/` as before and transparently get autodelete storage. Added `results` to `.gitignore` (the symlink target is not tracked).
**Rationale:** Autodelete storage is fast, not backed up, and expires after ~12 weeks — well past the Apr 22 deadline. Files are not tracked in git anyway (too large, binary). The symlink is transparent to all scripts.

## 2026-02-28 — Python 3.12 venv via uv with PyTorch 2.6.0+cu124
**Context:** Need a Python environment on the BYU supercomputer, which runs in FIPS mode.
**Decision:** Use `uv venv --python /apps/python/3.12.2/gcc-11.4.1/bin/python3 .venv` to create the venv against the system Python (loaded via `module load python/3.12`). Install packages with `uv pip install --python .venv/bin/python`; for torch, pass `--index-url https://download.pytorch.org/whl/cu124`. All SLURM scripts run `module load python/3.12` before activating the venv.
**Rationale:** The cluster enforces FIPS mode for OpenSSL. uv's bundled CPython links against its own OpenSSL which fails the FIPS self-test at runtime (`FATAL FIPS SELFTEST FAILURE`, core dump). The system Python at `/apps/python/3.12.2/gcc-11.4.1/` is built against the FIPS-compliant system OpenSSL. uv is still used for fast package management; only the interpreter is changed.
**Alternatives considered:** uv's bundled Python (fails FIPS); conda (not available).
