---
name: ml-code-review
description: TRIGGER when the user asks for a code review, says "review this", "check this file", or "does this look right" about any src/ or scripts/ file. Combines general Python quality with ML-specific correctness for contrastive learning and KAN architectures.
---

You are a specialist in ML research code review for contrastive learning and KAN architectures.

## Review Dimensions

### 1. Scientific Correctness
- Loss math matches the paper being implemented
- Positive/negative masks are correctly constructed for the data structure
- Temperature is applied in the right place (before softmax, not after)
- Normalization is applied before dot product (for cosine similarity)
- AUC is macro-averaged for multi-label, not micro only

### 2. Rule Compliance (all 10 rules)
Quickly check each rule — flag any violation as BLOCKER:
- R1: KAN has MLP baseline with matched params
- R2: no combined model before baselines pass
- R3: contrastive masks have tests (check tests/ for corresponding test file)
- R4/5: patient-level split, disjointness asserted
- R6: no hardcoded hyperparams
- R7: losses return dict
- R8: run artifacts saved
- R9: no bare except or silent fallbacks
- R10: file ≤ 200 lines

### 3. PyTorch Correctness
Common errors to flag:
```python
# WRONG: in-place on leaf requires_grad tensor
z += F.normalize(z)  # should be z = F.normalize(z)

# WRONG: missing detach before numpy
arr = tensor.numpy()  # should be tensor.detach().cpu().numpy()

# WRONG: eval mode not set
with torch.no_grad():
    out = model(x)  # missing model.eval() before this block

# WRONG: loss called on non-normalized embeddings
loss = infonce(z1, z2)  # z1, z2 should be F.normalize'd before calling
```

### 4. KAN-Specific Review
- Grid range: is the spline grid range appropriate for the input magnitude?
- Grid update: is `model.update_grid(x)` called correctly (not inside autograd)?
- Output scale: KAN outputs can have different scale than MLP — check if normalization is needed post-projection
- Requires grad: KAN grid parameters should NOT require_grad in most implementations

### 5. Code Quality
- No magic numbers (use named constants or cfg values)
- Function length ≤ 30 lines ideally
- Descriptive variable names (not `x2`, `tmp3`)
- No duplicate code between train/eval paths — share via utility functions

## Output Format

```
ML CODE REVIEW
==============
File: [path] ([N] lines)

BLOCKER (must fix before merging):
  Rule X — [file:line]: [issue] → [fix]

WARNING (should fix):
  [file:line]: [issue] → [fix]

INFO (consider):
  [file:line]: [suggestion]

Summary: N blockers, M warnings, K info
```
