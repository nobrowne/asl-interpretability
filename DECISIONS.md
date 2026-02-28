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

## 2026-02-28 — Python 3.12 venv with PyTorch 2.6.0+cu124
**Context:** No `uv` available on the BYU supercomputer; need a Python environment.
**Decision:** `module load python/3.12` + standard `python -m venv .venv` + `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
**Rationale:** Python 3.12 is the system default and well-supported by PyTorch 2.x. CUDA 12.4 PyTorch build is compatible with the cluster's CUDA 12.8.
**Alternatives considered:** Python 3.11 (older but more conservative); conda (not the system package manager here).
