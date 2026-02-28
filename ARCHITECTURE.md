# ARCHITECTURE.md — Technical Reference

## SPOTER (Sign POse-based TransformER)

- **Paper:** Bohacek & Hruz (2022). "Sign Pose-based Transformer for Word-level Sign Language Recognition." WACV 2022.
- **Repo:** https://github.com/matyasbohacek/spoter
- **Architecture:** Standard transformer encoder-decoder
  - 6 encoder layers, 6 decoder layers, 9 attention heads
  - Hidden dim: 108, feed-forward dim: 2048
  - 5.92M parameters
  - Single "Class Query" decoded into sign classification
- **Input:** Sequences of 54 body landmarks (108 dimensions per frame)
  - 5 head landmarks (eyes, ears, nose) — NOTE: no eyebrow or mouth detail
  - 21 landmarks per hand (4 per finger + wrist)
  - Body joints (shoulders, elbows, wrists, etc.)
- **Data:** Preprocessed pose data for WLASL100 provided in repo (no pose extraction needed)
- **Training:** 350 epochs, SGD, lr=1e-3, cross-entropy loss, no scheduler
- **Expected accuracy:** ~63% top-1 on WLASL100
- **License:** Code is Apache 2.0; skeletal data is CC BY-NC 4.0

### WLASL100 Dataset
- 100 sign glosses, ~2,038 video instances
- Average ~20 videos per gloss, 97 unique signers
- Public dataset split provided by authors
- Signs performed by native ASL signers and interpreters

---

## mBERT (Multilingual BERT)

- **Model:** `bert-base-multilingual-cased` from HuggingFace
- **Purpose:** Extract English gloss representations for cross-modal comparison with SPOTER
- **Usage:** Inference only — no fine-tuning
- **Why mBERT over XLM-R:**
  - Course focuses on mBERT (Pires et al., Wu & Dredze, Chi et al. all analyze it)
  - mBERT's cross-lingual ability emerged without explicit cross-lingual objective, making comparison with SPOTER (also not explicitly cross-modal) more parallel
  - XLM-R is a stretch goal for secondary comparison
- **Extraction approach:**
  1. Tokenize each of the 100 WLASL100 glosses
  2. Forward pass, extract hidden states at all 12 layers
  3. For multi-token words, average token representations
  4. Result: 100 embeddings × 12 layers × 768 dimensions

---

## Linguistic Dimensions for Analysis

When analyzing attention patterns and probing, focus on these sign language properties:

| Property | Description | Example |
|---|---|---|
| Handshape | Configuration of the hand | Flat, fist, pointing, claw |
| Location | Where the sign is produced | Forehead, chest, neutral space |
| Movement | Motion pattern | Circular, linear, repeated, arc |
| Handedness | One-handed vs. two-handed | — |
| Iconicity | Whether the sign visually resembles its meaning | "drink" (iconic) vs. arbitrary signs |

### Case-Study Sign Selection
Select 5–10 signs from WLASL100 that vary systematically along these dimensions:
- 2–3 iconic signs (e.g., "drink," "book," "eat")
- 2–3 arbitrary signs
- Mix of one-handed and two-handed
- Signs distinguished primarily by handshape vs. location vs. movement
- Document the selection rationale — this goes in the report methodology section

---

## Attention Visualization Approach

SPOTER uses PyTorch's `nn.MultiheadAttention` (not HuggingFace), so BertViz won't work directly.

**Approach:**
1. Register forward hooks on each `nn.MultiheadAttention` module to capture attention weights
2. Save weights as tensors organized by (layer, head, sample)
3. Visualize with matplotlib/seaborn:
   - Heatmaps showing attention patterns for individual signs
   - Average attention per head aggregated across the dataset
   - Head specialization: which input dimensions (= body landmarks) each head attends to most
4. Map attention back to anatomy: input dimensions correspond to known body landmarks, so cluster patterns into hand-focused, body-focused, and head-focused heads

**Key implementation detail:** `nn.MultiheadAttention` supports `need_weights=True`. You may need to modify the forward pass or use hooks to ensure weights are captured and returned.

---

## Cross-Modal Comparison Approach

### SPOTER Representation Extraction
For each sign class:
1. Run all test examples through SPOTER
2. Extract hidden states at each of the 6 encoder layers
3. Average across examples of the same sign class
4. Average across sequence positions (or use class query output from decoder)
5. Result: 100 embeddings × 6 layers × 108 dimensions

### Alignment Metrics
- **CKA (Centered Kernel Alignment):** Measures representational similarity between two sets of representations regardless of dimensionality. Compare each SPOTER layer to each mBERT layer → 6×12 similarity matrix.
- **RSA (Representational Similarity Analysis):** Compute pairwise distance matrices within each space, then correlate. Tests whether signs "close" in SPOTER space have glosses "close" in mBERT space.

---

## Results Naming Convention

Consistent naming prevents the `results/` folder from becoming unnavigable. Use this pattern:

### Checkpoints
```
results/checkpoints/spoter_wlasl100_epoch{N}.pt
```
Examples: `spoter_wlasl100_epoch010.pt`, `spoter_wlasl100_epoch350.pt`

### Attention Weights
```
results/attention_epoch{N}.pt
```
Contains a dict: `{layer_idx: {head_idx: tensor(num_samples, seq_len, seq_len)}}`

### Extracted Representations
```
results/representations_spoter_epoch{N}.pt
results/representations_mbert.pt
```
Contains a dict: `{layer_idx: tensor(num_classes, hidden_dim)}`

### Probing Results
```
results/probe_{property}_{layer}.json
```
Examples: `probe_handshape_layer3.json`, `probe_movement_layer5.json`

JSON contains: `{"accuracy": float, "baseline": float, "confusion_matrix": [...], "classification_report": str}`

### Alignment Results
```
results/cka_spoter_mbert.npy       # 6x12 CKA similarity matrix
results/rsa_spoter_mbert.npy       # 6x12 RSA correlation matrix
```

### Figures
```
figures/{category}_{description}.pdf
```
Examples: `figures/attention_head_specialization.pdf`, `figures/probing_accuracy_by_layer.pdf`, `figures/cka_heatmap.pdf`

Use PDF for publication-quality figures (ACL format). Use PNG only for draft/exploration.

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
