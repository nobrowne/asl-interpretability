# CLAUDE.md — ASL Interpretability Project

## What Is This Project?

A research project for CS 601R-004 (Interpretability and Analysis of Multilingual Language Models) at BYU, Winter 2026. We train a pose-based sign language transformer (SPOTER) on WLASL100 and apply interpretability techniques to understand what the model learns about ASL. No prior work exists at this intersection — this is novel research.

**Research questions:**
1. Do attention heads specialize by body region (hands vs. body vs. head)?
2. Do representations evolve from spatial/geometric (early layers) to semantic (later layers)?
3. What sign linguistic properties (handshape, movement, location) are linearly decodable from intermediate representations?
4. How do SPOTER's sign representations compare to mBERT's English gloss representations? (Interlingua hypothesis across modalities.)

## Project Context Files

Read these when relevant to the task at hand:
- **PLAN.md** — Phased timeline, milestones, and course deadlines
- **ARCHITECTURE.md** — SPOTER details, mBERT usage, linguistic dimensions, analysis approaches
- **PROGRESS.md** — Living session log (read at start of every session, update at end)
- **DECISIONS.md** — Technical decision record

## Directory Structure

```
~/projects/asl-interpretability/
├── analysis/                        # This repo (the deliverable)
│   ├── CLAUDE.md                    # This file
│   ├── PLAN.md
│   ├── ARCHITECTURE.md
│   ├── PROGRESS.md
│   ├── DECISIONS.md
│   ├── README.md
│   ├── scripts/
│   │   ├── download_data.sh
│   │   ├── train_spoter.py
│   │   └── (future: extract_attention.py, extract_representations.py, etc.)
│   ├── slurm/
│   │   ├── train.sh
│   │   ├── analysis.sh
│   │   └── tests.sh
│   ├── tests/
│   ├── analysis/
│   │   └── (future: visualize_attention.py, probing_analysis.py, etc.)
│   ├── results/
│   │   ├── checkpoints/
│   │   └── logs/
│   ├── figures/
│   └── report/
│
├── spoter/                          # Cloned SPOTER repo (modified as needed)
│   ├── spoter/                      # Model code
│   ├── data/                        # Preprocessed WLASL pose data
│   ├── train.py                     # Original training script
│   └── ...
```

## Coding Standards

- **Type hints on all function signatures.** No exceptions.
- **Google-style docstrings on all functions and classes.**
- **Verbose comments** for non-obvious logic, especially ML pipeline steps.
- **Test-supported development:** Write sanity-check tests for objectively verifiable things (shapes, no NaNs, attention sums to 1, layers are distinct). Not full TDD — you can't pre-specify what "good" attention looks like. Run tests before each new experiment phase.

## Hard Constraints

- **NEVER run Python interactively in the terminal.** Always use SLURM job scripts. Interactive execution causes CPU time limit exceeded errors and core dumps on this system. This includes `python -c "..."` one-liners — even importing torch directly on the login node can trigger a core dump due to CUDA initialization hitting the CPU time limit.
- **~50 hours total budget.** Track time in PROGRESS.md. Pivot early if something eats hours without progress.
- **Interpretability is the centerpiece, not model performance.** Don't optimize SPOTER beyond reproducing ~63% on WLASL100.
- **Report:** ACL LaTeX format on Overleaf. Proposal grows into final report (same document).
- **SPOTER's WLASL data is CC BY-NC 4.0.** Non-commercial use only.
- **Save everything.** Checkpoints, attention weights, probe results, figures.

## Environment

- **Python environment:** uv venv
- **Compute:** BYU supercomputer, A100 GPUs, SLURM scheduler
- **Key dependencies:** torch, torchvision, transformers, scikit-learn, matplotlib, seaborn, pytest

### SLURM Templates

**Training:**
```bash
#!/bin/bash
#SBATCH --job-name=spoter-train
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=16G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-train.out

source ~/projects/asl-interpretability/analysis/.venv/bin/activate
python scripts/train_spoter.py "$@"
```

**Analysis:**
```bash
#!/bin/bash
#SBATCH --job-name=spoter-analysis
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-analysis.out

source ~/projects/asl-interpretability/analysis/.venv/bin/activate
python scripts/"$1" "${@:2}"
```

**Tests (CPU only):**
```bash
#!/bin/bash
#SBATCH --job-name=spoter-tests
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --qos=cs
#SBATCH --output=slurm/%j-tests.out

source ~/projects/asl-interpretability/analysis/.venv/bin/activate
pytest tests/ -v
```

## Session Protocol

1. **Start of session:** Read PROGRESS.md to restore context.
2. **During session:** Work on tasks, run tests, save results.
3. **End of session:** Propose an update to PROGRESS.md and DECISIONS.md (if any decisions were made).
