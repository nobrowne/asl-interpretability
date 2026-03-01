# Project Progress Log

## Session 2026-02-28
### What was done
- Explored SPOTER codebase: architecture, training loop, dataset loader, checkpoint logic
- Created full analysis repo directory structure (scripts/, slurm/, tests/, results/, figures/, report/)
- Wrote `scripts/train_spoter.py`: training wrapper with epoch-specific checkpoints at epochs 10/50/100/200/350
- Wrote `slurm/train.sh`, `slurm/analysis.sh`, `slurm/tests.sh`
- Wrote `scripts/download_data.sh` (login-node only — compute nodes have no internet)
- Set up GitHub repo: git@github.com:nobrowne/asl-interpretability.git
- Fixed SSH config: added `IdentityFile ~/.ssh/id_rsa` + `IdentitiesOnly yes` for github.com (FIPS mode blocks Ed25519)
- Discovered correct SPOTER data release URL: github.com/maty-bohacek/spoter (not matyasbohacek)
- Downloaded all three WLASL100 CSVs to spoter/data/ (train: 58MB, val: 14MB, test: 11MB)
- Set up Python 3.12 venv + installed torch 2.6.0+cu124, transformers, scikit-learn, matplotlib, seaborn, pytest, pandas
- Submitted training job: SLURM job 10546305, up to 4 hours, pending GPU allocation

### Key decisions
- See DECISIONS.md

### Results & observations
- `CzechSLRDataset` name is misleading — it's a general skeleton CSV loader used for both Czech SLR and WLASL (ASL) data
- SPOTER transformer: encoder sees sequence of frames (each 108-dim), decoder has single class query token. Encoder self-attention operates over temporal frame sequence — this is the primary target for interpretability analysis
- Checkpoints save `state_dict` (not full model object) + metadata (epoch, train/val acc, args) for portability
- CUDA device detection uses `"CUDA_VISIBLE_DEVICES" in os.environ` rather than `torch.cuda.is_available()` per user preference

### Open questions
- Will job 10546305 reproduce ~63% val accuracy? Watch `slurm/10546305-train.out`
- How many frames does a typical WLASL100 sample have? Determines attention matrix size

### Next steps
1. ~~Write `scripts/extract_attention.py`~~ — DONE
2. ~~Write `tests/test_attention.py`~~ — DONE
3. Write `scripts/extract_mbert.py`
4. Select 5–10 case-study signs from WLASL100 label set
5. Start proposal draft on Overleaf

### Time spent
- This session: ~2 hours
- Running total: ~2 hours / ~50 hours budgeted

---

## Session 2026-02-28 (continued)
### What was done
- Migrated `results/` to fast autodelete storage (`~/nobackup/autodelete/asl-interpretability/results/`)
  - Created subdirs: `checkpoints/`, `logs/`, `attention/`
  - Replaced `analysis/results/` with a symlink → autodelete
  - Updated `.gitignore`: `results` entry replaces per-file patterns (symlink target not tracked)
- Wrote `scripts/extract_attention.py`:
  - Monkey-patches all 12 MHA modules (`enc_0..5`, `dec_0..5`) via `patch_mha()` to force `need_weights=True, average_attn_weights=False`
  - Captures per-head weights (after squeezing batch dim) in shared storage dict
  - Inference loop over test set (batch_size=1), collects per-sample dicts
  - Auto-derives output path from checkpoint epoch: `results/attention/attention_epoch{N:03d}.pt`
  - Saves full results dict with metadata + samples list via `torch.save`
- Wrote `tests/test_attention.py`:
  - Uses random-weight SPOTER (no checkpoint needed, CPU-only)
  - 11 tests: shapes, sums-to-1, no-NaN, not-all-zero, differs-across-layers for both enc+dec; patch doesn't alter output; all 12 storage keys captured

### Key decisions
- `need_weights=True` is NOT returned by default in PyTorch's TransformerEncoderLayer (hardcoded False). Monkey-patching the MHA forward is the correct approach — forward hooks on the module would get the post-softmax weights only if `need_weights=True` is passed through. Patching at the MHA level lets us override the kwarg before the actual computation.
- Decoder layer's `[0]` indexing still works after patching because our patched forward returns the full original output tuple unchanged.
- `batch_size=1` for extraction DataLoader — WLASL100 sequences have variable length T, and padding would alter attention patterns (padded keys would receive attention mass). One-at-a-time avoids this entirely.

### Open questions
- Will job 10546305 reproduce ~63% val accuracy? Watch `slurm/10546305-train.out`

### Next steps
1. Once checkpoints exist: `sbatch slurm/analysis.sh extract_attention.py --checkpoint results/checkpoints/wlasl100_spoter_epoch350.pth --data_path /path/to/WLASL100_test_25fps.csv`
2. Run tests: `sbatch slurm/tests.sh`
3. Write `scripts/extract_mbert.py`
4. Select 5–10 case-study signs
5. Start proposal draft on Overleaf

### Time spent
- This session: ~1 hour
- Running total: ~3 hours / ~50 hours budgeted
