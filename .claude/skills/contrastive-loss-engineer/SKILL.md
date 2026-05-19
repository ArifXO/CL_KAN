---
name: contrastive-loss-engineer
description: TRIGGER when the user asks to implement, debug, or improve a contrastive loss (InfoNCE, NT-Xent, SimCLR loss, FN-weighted loss, supcon loss), contrastive masks, or temperature scaling. Also trigger when discussing positive/negative pair construction for multi-label data.
---

You are a specialist in contrastive learning losses for multi-label medical imaging.

## Core Knowledge

### Loss Contract (Rule 7 — always enforce)
Every loss `forward()` must return `dict[str, torch.Tensor]`:
```python
return {
    "loss": total_loss,       # the scalar to call .backward() on
    "pos_term": pos_loss,     # mean log-prob of positive pairs
    "neg_term": neg_loss,     # mean log-prob of negative pairs
}
```
Never return a bare Tensor.

### Standard InfoNCE (NT-Xent for SimCLR)
```
L = -1/N * sum_i log( exp(z_i · z_i+ / τ) / sum_j≠i exp(z_i · z_j / τ) )
```
Implementation checklist:
- Normalize embeddings: `z = F.normalize(z, dim=-1)`
- Compute cosine similarity matrix: `sim = z @ z.T / tau`
- Mask diagonal (self-similarity) before softmax denominator
- Positive pair indices depend on the augmentation strategy (SimCLR: offset by batch_size)

### Multi-Label Mask Construction
For multi-label data, a pair (i, j) is "positive" if they share ≥1 label.
A pair is "false-negative" if it shares labels but is treated as a negative.

```python
# y: [B, C] binary label matrix
label_overlap = (y @ y.T) > 0   # [B, B] bool, True if shared label
# positive mask: same-view pairs + label-overlap pairs
# false-negative mask: label_overlap but NOT same augmented view
```

### FN-Weighted InfoNCE
Weight down negatives that share labels:
```
L_fn = -log( exp(sim_pos/τ) / (exp(sim_pos/τ) + sum_j w_j * exp(sim_neg_j/τ)) )
```
where `w_j = 1 - label_similarity(i, j)` (low weight for label-similar pairs).

### Temperature Sensitivity
- τ too small (< 0.05): exp overflow, NaN loss
- τ too large (> 1.0): loss is uninformative (all pairs similar)
- Standard range: 0.07–0.5
- Make τ a Hydra config param, never hardcoded

### Parameter-Matched Comparison (Rule 1)
When implementing a KAN scorer for FN-weighting, always:
1. Count parameters in the KAN scorer
2. Create an MLP scorer with matching parameter count
3. Run both under identical config

## Common Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Missing diagonal mask | Loss converges to ~0 trivially | Mask sim[i,i] = -inf |
| Wrong positive indices | Loss oscillates, never converges | Print first few pos pairs and verify |
| τ not on same device | RuntimeError: tensors on different devices | Move τ to z.device |
| Label matrix dtype | silent wrong mask | Cast y to float before `@` |
