# PLAN.md — Project Timeline & Milestones

## Key Deadlines

| Date | Milestone |
|---|---|
| Feb 24, 2026 | Optional proposal draft (submit for early feedback) |
| Mar 2, 2026 | Proposal document due (~3 pages, ACL format) |
| Mar 3–5, 2026 | Proposal presentation (~10 min) |
| Mar 19, 2026 | Mid-project check-in (brief verbal update) |
| Mar 26 & Apr 2 | Peer feedback sessions |
| Mar 31, 2026 | Optional draft final report |
| Apr 7–14, 2026 | Final presentations (~20 min) |
| Apr 22, 2026 | Final written report due (~8 pages, ACL format) |

---

## Phase 1: Reproduction & Proposal (Now → Mar 2) ~15 hours

- [ ] Clone SPOTER repo, set up environment, download preprocessed WLASL100 data
- [ ] Train SPOTER on WLASL100, verify ~63% accuracy reproduction
- [ ] Save checkpoints at multiple training stages (e.g., epoch 10, 50, 100, 200, 350)
- [ ] Add hooks to extract attention weights and intermediate layer representations
- [ ] Produce at least one attention visualization showing head behavior
- [ ] Extract mBERT embeddings for the 100 English glosses (basic feasibility check)
- [ ] Select 5–10 case-study signs varying along linguistic dimensions
- [ ] Write and submit proposal (ACL LaTeX on Overleaf)
- [ ] Prepare proposal presentation (~10 min)

## Phase 2: Core Interpretability Analysis (Mar 3 → Mar 26) ~20 hours

- [ ] Systematic attention analysis across all heads and layers
- [ ] Cluster/categorize attention heads by body-region specialization
- [ ] Compare attention patterns across training stages (epoch 10 vs. 350)
- [ ] Compare attention on case-study signs (iconic vs. arbitrary, 1-hand vs. 2-hand)
- [ ] Design and train probing classifiers for linguistic properties at each layer
- [ ] Cross-modal comparison: CKA/RSA between SPOTER representations and mBERT
- [ ] Layer-wise alignment analysis: which layers are most semantically aligned?

## Phase 3: Synthesis & Writing (Mar 26 → Apr 22) ~15 hours

- [ ] Synthesize findings into coherent narrative
- [ ] Prepare final presentation (~20 min)
- [ ] Write final report (~8 pages, ACL format)
- [ ] Clean up code repo for submission

## Stretch Goals (in priority order)

1. WLASL300 expansion — retrain on 300 classes, check if attention patterns scale
2. XLM-R comparison alongside mBERT for cross-modal analysis
3. Training dynamics — analyze how attention patterns evolve using multi-epoch checkpoints
4. Facial landmark extension — add MediaPipe face mesh to SPOTER input, retrain, investigate non-manual marker attention (connects to prior thesis work)
