# CLAUDE.md — ASL Interpretability Project

## Project Overview

This is a research project for CS 601R-004 (Interpretability and Analysis of Multilingual Language Models) at BYU, Winter 2026. The goal is to train a pose-based sign language transformer (SPOTER) on the WLASL dataset and apply interpretability techniques to understand what the model learns about American Sign Language (ASL).

**Research questions:**
1. Do attention heads in SPOTER specialize by body region (hands vs. body vs. head)?
2. How do representations evolve across layers — spatial/geometric early, semantic later?
3. What linguistic properties of signs (handshape, movement, location) are linearly decodable from intermediate representations?
4. How do SPOTER's sign representations compare geometrically to mBERT's representations of the corresponding English glosses?

**This project is novel.** No prior work applies interpretability techniques to sign language transformers. The intersection of ASL processing and model interpretability is essentially unexplored.

---

## Key Deadlines

| Date | Milestone |
|---|---|
| Feb 24, 2026 | Optional proposal draft (submit for early feedback) |
| Mar 2, 2026 | Proposal document due (~3 pages, ACL format) |
| Mar 3–5, 2026 | Proposal presentation (~10 min) |
| Mar 19, 2026 | Mid-project check-in (brief verbal update) |
| Mar 26 & Apr 2, 2026 | Peer feedback sessions |
| Mar 31, 2026 | Optional draft final report |
| Apr 7–14, 2026 | Final presentations (~20 min) |
| Apr 22, 2026 | Final written report due (~8 pages, ACL format) |

---

## Phased Plan

### Phase 1: Reproduction & Proposal (Now → Mar 2) ~15 hours
- [ ] Clone SPOTER repo, set up environment, download preprocessed WLASL100 data
- [ ] Train SPOTER on WLASL100, verify ~63% accuracy reproduction
- [ ] Save checkpoints at multiple training stages (e.g., epoch 10, 50, 100, 200, 350)
- [ ] Add hooks to extract attention weights and intermediate layer representations
- [ ] Produce at least one attention visualization showing head behavior
- [ ] Extract mBERT embeddings for the 100 English glosses (basic feasibility check)
- [ ] Select 5–10 case-study signs varying along linguistic dimensions
- [ ] Write and submit proposal (ACL LaTeX on Overleaf)
- [ ] Prepare proposal presentation (~10 min)

### Phase 2: Core Interpretability Analysis (Mar 3 → Mar 26) ~20 hours
- [ ] Systematic attention analysis across all heads and layers
- [ ] Cluster/categorize attention heads by body-region specialization
- [ ] Compare attention patterns across training stages (epoch 10 vs. 350)
- [ ] Compare attention on case-study signs (iconic vs. arbitrary, 1-hand vs. 2-hand)
- [ ] Design and train probing classifiers for linguistic properties at each layer
- [ ] Cross-modal comparison: CKA/RSA between SPOTER representations and mBERT
- [ ] Layer-wise alignment analysis: which layers are most semantically aligned?

### Phase 3: Synthesis & Writing (Mar 26 → Apr 22) ~15 hours
- [ ] Synthesize findings into coherent narrative
- [ ] Prepare final presentation (~20 min)
- [ ] Write final report (~8 pages, ACL format)
- [ ] Clean up code repo for submission
- [ ] Stretch goals if time allows:
  - [ ] Expand to WLASL300
  - [ ] Add XLM-R comparison alongside mBERT
  - [ ] Facial landmark extension (MediaPipe face mesh → retrain SPOTER)

---

## Architecture & Technical Context

### SPOTER (Sign POse-based TransformER)
- **Paper:** Bohacek & Hruz, WACV 2022
- **Repo:** https://github.com/matyasbohacek/spoter (clone, do not fork — modifications go in local copy)
- **Architecture:** Standard transformer encoder-decoder
  - 6 encoder layers, 6 decoder layers, 9 attention heads
  - Hidden dim: 108, feed-forward dim: 2048
  - 5.92M parameters
  - Single "Class Query" decoded into sign classification
- **Input:** Sequences of 54 body landmarks (108 dimensions per frame)
  - 5 head landmarks (eyes, ears, nose)
  - 21 landmarks per hand (4 per finger + wrist)
  - Body joints (shoulders, elbows, wrists, etc.)
- **Data:** Preprocessed pose data for WLASL100 and LSA64 provided in repo
- **Training:** 350 epochs, SGD, lr=1e-3, cross-entropy loss
- **Expected accuracy:** ~63% top-1 on WLASL100

### mBERT (Multilingual BERT)
- **Model:** `bert-base-multilingual-cased` from HuggingFace
- **Purpose:** Extract English gloss representations for cross-modal comparison
- **Usage:** Inference only — no fine-tuning. Extract hidden states at each layer for the 100 WLASL100 glosses.

### Linguistic Dimensions for Analysis
When analyzing attention patterns and probing, focus on these dimensions:
- **Handshape:** The configuration of the hand (e.g., flat, fist, pointing)
- **Location:** Where the sign is produced (e.g., forehead, chest, neutral space)
- **Movement:** The motion pattern (e.g., circular, linear, repeated)
- **One-handed vs. two-handed signs**
- **Iconic vs. arbitrary signs** (iconic = sign visually resembles its meaning)

### Case-Study Sign Selection
Select 5–10 signs from WLASL100 that vary systematically:
- 2–3 iconic signs (e.g., "drink," "book," "eat")
- 2–3 arbitrary signs
- Mix of one-handed and two-handed
- Signs distinguished primarily by handshape vs. location vs. movement
- Document the selection rationale — this goes in the report

---

## Directory Structure

```
~/projects/asl-interpretability/
├── spoter/                          # Cloned SPOTER repo (modified)
│   ├── train.py                     # Training script (add checkpoint saving)
│   ├── spoter/                      # Model code (add attention hooks)
│   └── data/                        # Preprocessed WLASL pose data
│
├── analysis/                        # YOUR repo (deliverable)
│   ├── CLAUDE.md                    # This file
│   ├── PROGRESS.md                  # Living project log (see below)
│   ├── README.md                    # Project overview for submission
│   ├── requirements.txt             # Or pyproject.toml with uv
│   ├── scripts/
│   │   ├── train_spoter.py          # Training wrapper / SLURM script
│   │   ├── extract_attention.py     # Extract and save attention weights
│   │   ├── extract_representations.py  # Extract hidden states for probing
│   │   ├── extract_mbert.py         # Extract mBERT gloss embeddings
│   │   ├── run_probing.py           # Train probing classifiers
│   │   └── compute_alignment.py     # CKA/RSA cross-modal comparison
│   ├── tests/
│   │   ├── test_data_loading.py     # Verify data shapes, no NaNs
│   │   ├── test_attention.py        # Attention weights sum to 1, correct shapes
│   │   ├── test_representations.py  # Layer outputs have expected dimensions
│   │   ├── test_probing.py          # Probe input/output dimensions match
│   │   └── test_mbert.py            # mBERT outputs are layer-specific
│   ├── analysis/
│   │   ├── visualize_attention.py   # Attention heatmaps and head clustering
│   │   ├── probing_analysis.py      # Probing accuracy plots across layers
│   │   └── alignment_analysis.py    # CKA/RSA visualization
│   ├── results/                     # Saved outputs (attention weights, probe results, figures)
│   ├── figures/                     # Publication-quality figures for report
│   ├── slurm/                       # SLURM job scripts
│   └── report/                      # Local copy of report assets (main writing in Overleaf)
```

---

## Development Environment

### Setup
```bash
# On BYU supercomputer
cd ~/projects/asl-interpretability

# Clone SPOTER
git clone https://github.com/matyasbohacek/spoter.git

# Create your analysis repo
mkdir analysis && cd analysis
git init

# Set up Python environment with uv
uv venv .venv
source .venv/bin/activate

# Core dependencies (adjust versions as needed)
uv pip install torch torchvision transformers
uv pip install scikit-learn matplotlib seaborn
uv pip install pytest numpy pandas
```

### Compute
- **GPU:** A100s on BYU supercomputer
- **Workflow:** Python scripts submitted as SLURM jobs
- **NEVER run Python interactively in the terminal** — always use SLURM job scripts or batch submission. Interactive Python causes CPU time limit exceeded errors and core dumps on this system.

### SLURM Templates

**Training (SPOTER on WLASL100 — 350 epochs should comfortably fit in 4 hours on A100):**
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

**Analysis (extraction, probing, alignment — GPU needed for forward passes but not heavy compute):**
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

**Tests (CPU only, quick):**
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

---

## Coding Standards

### Style
- **Type hints on all function signatures.** No exceptions.
- **Docstrings on all functions and classes.** Use Google-style docstrings.
- **Verbose explanations** in comments for non-obvious logic, especially ML pipeline steps.

### Example
```python
def extract_attention_weights(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Extract attention weights from all encoder layers for each input.

    Registers forward hooks on each MultiheadAttention module in the
    SPOTER encoder to capture attention weight matrices during inference.

    Args:
        model: Trained SPOTER model in eval mode.
        data_loader: DataLoader yielding (pose_sequence, label) batches.
        device: Torch device (cuda/cpu).

    Returns:
        Dictionary mapping layer names to tensors of shape
        (num_samples, num_heads, seq_len, seq_len).
    """
    ...
```

### Testing Philosophy: Test-Supported Development
Full TDD doesn't fit ML research — you can't pre-specify what "good" attention patterns look like. Instead, write **sanity-check tests** for everything that is objectively verifiable:

- **Data pipeline tests:** Loaded tensors have expected shapes, no NaNs, values in expected ranges, labels are correct integers.
- **Model output tests:** Attention weights sum to 1 across the attended dimension, hidden states have correct dimensions at each layer, output logits have shape (batch, num_classes).
- **Extraction tests:** Extracted attention matrices are not all zeros or all identical across layers (a common hook bug). mBERT embeddings differ across layers (not accidentally extracting the same layer repeatedly).
- **Probing tests:** Probe input features have correct dimensions, train/test split preserves label distribution, accuracy is above random baseline (1/num_classes for classification probes).

Write tests BEFORE running each new experiment phase. When a test fails, it catches bugs early. When all tests pass, you proceed with confidence.

Run tests with:
```bash
pytest tests/ -v
```

---

## Attention Visualization Approach

Since SPOTER uses PyTorch's `nn.MultiheadAttention` (not HuggingFace), BertViz won't work directly. Instead:

1. **Register forward hooks** on each `MultiheadAttention` module to capture attention weights.
2. **Save weights** as tensors organized by (layer, head, sample).
3. **Visualize with matplotlib/seaborn:**
   - Heatmaps showing attention patterns for individual signs
   - Average attention per head aggregated across the dataset
   - Head specialization analysis: which input dimensions (= body landmarks) each head attends to most
4. **Map attention back to anatomy:** Since input dimensions correspond to known body landmarks, cluster attention patterns into hand-focused, body-focused, and head-focused heads.

Key implementation detail: SPOTER's encoder uses `nn.MultiheadAttention` with `need_weights=True` to get attention weights. You may need to modify the forward pass or use hooks to ensure weights are returned.

---

## Cross-Modal Comparison Approach

### mBERT Embedding Extraction
For each of the 100 WLASL100 glosses (English words):
1. Tokenize with mBERT tokenizer
2. Run forward pass, extract hidden states at all 12 layers
3. For multi-token words, average the token representations
4. Result: 100 embeddings × 12 layers × 768 dimensions

### SPOTER Representation Extraction
For each sign class:
1. Run all test examples through SPOTER
2. Extract hidden states at each of the 6 encoder layers
3. Average across examples of the same sign class
4. Average across sequence positions (or use the class query output)
5. Result: 100 embeddings × 6 layers × 108 dimensions

### Alignment Metrics
- **CKA (Centered Kernel Alignment):** Measures representational similarity between two sets of representations regardless of dimensionality. Compare each SPOTER layer to each mBERT layer → 6×12 similarity matrix.
- **RSA (Representational Similarity Analysis):** Compute pairwise distance matrices within each representation space, then correlate the distance matrices. Tests whether signs that are "close" in SPOTER space have glosses that are "close" in mBERT space.

---

## Context Engineering (Lightweight ACE)

This project uses a lightweight version of Stanford's ACE (Agent Context Engineering) framework to maintain useful context across work sessions.

### PROGRESS.md
A living document that gets updated after each work session. Structure:

```markdown
# Project Progress Log

## Session [DATE]
### What was done
- [Concrete accomplishments]

### Key decisions
- [Decisions made and rationale]

### Results & observations
- [What was observed, even if inconclusive]

### Open questions
- [Questions to investigate next]

### Next steps
- [Prioritized list for next session]

### Time spent
- [Hours this session, running total]
```

**Rules for PROGRESS.md:**
- Claude Code should read PROGRESS.md at the start of every session to restore context.
- Claude Code should propose an update to PROGRESS.md at the end of every session.
- Keep entries concise — this is a working log, not a report.
- Flag any result that was surprising or that contradicts expectations.
- Track cumulative time spent (target: ~50 hours total).

### DECISIONS.md
A separate file tracking key technical decisions and their rationale:

```markdown
# Technical Decisions

## [DATE] - [Decision Title]
**Context:** [Why this decision came up]
**Decision:** [What was decided]
**Rationale:** [Why]
**Alternatives considered:** [What else was on the table]
```

This is especially useful for the final report's methodology section — you'll have a ready-made record of why you made each choice.

---

## Important Constraints

- **Do NOT run Python interactively in the terminal.** Always submit as SLURM jobs or run as batch scripts. Interactive execution causes CPU time limit errors and core dumps.
- **~50 hours total budget.** Track time. If an approach is eating hours without progress, pivot early.
- **Interpretability is the centerpiece, not model performance.** Don't spend time optimizing SPOTER's accuracy beyond reproducing ~63%. The goal is understanding, not SOTA.
- **Report is ACL LaTeX format on Overleaf.** The proposal document grows into the final report — same document, iteratively refined.
- **SPOTER's WLASL data is CC BY-NC 4.0.** Non-commercial use only. Fine for academic work.
- **Save everything.** Checkpoints, attention weights, probe results, figures. Disk space is cheap; re-running experiments is not.

---

## Stretch Goals (in priority order)

1. **WLASL300 expansion:** Retrain on 300 classes, check if attention patterns scale.
2. **XLM-R comparison:** Add alongside mBERT for the cross-modal analysis.
3. **Training dynamics:** Analyze how attention patterns evolve during training using saved checkpoints at multiple epochs.
4. **Facial landmark extension:** Add MediaPipe face mesh landmarks to SPOTER's input, retrain, and investigate whether the model learns to attend to non-manual markers (eyebrows, mouth shape). This connects directly to prior thesis work on eyebrow-based ASL question classification.

---

## References

- Bohacek & Hruz (2022). "Sign Pose-based Transformer for Word-level Sign Language Recognition." WACV 2022.
- Conneau et al. (2018). "What you can cram into a single vector." ACL 2018.
- Clark et al. (2019). "What Does BERT Look At?" BlackboxNLP 2019.
- Tenney et al. (2019). "BERT Rediscovers the Classical NLP Pipeline." ACL 2019.
- Pires et al. (2019). "How multilingual is Multilingual BERT?" ACL 2019.
- Wu & Dredze (2019). "Beto, Bentz, Becas." EMNLP 2019.
- Chi et al. (2020). "Finding Universal Grammatical Relations in Multilingual BERT." ACL 2020.
- Jiang et al. (2024). "SignCLIP: Connecting Text and Sign Language by Contrastive Learning." EMNLP 2024.
