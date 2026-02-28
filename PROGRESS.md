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
- Is the `need_weights=True` flag needed to extract attention from `nn.MultiheadAttention`? Check when writing extract_attention.py
- How many frames does a typical WLASL100 sample have? Determines attention matrix size

### Next steps
1. Write `scripts/extract_attention.py` (next priority — needed for proposal visualization)
2. Write `scripts/extract_mbert.py`
3. Write `tests/test_data_loading.py` and `tests/test_attention.py`
4. Select 5–10 case-study signs from WLASL100 label set
5. Start proposal draft on Overleaf

### Time spent
- This session: ~2 hours
- Running total: ~2 hours / ~50 hours budgeted
