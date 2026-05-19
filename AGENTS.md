# AGENTS.md — CXR-KAN-Contrastive

This file is for AI coding agents (Codex, Claude Code subagents, etc.).
Human contributors: see CLAUDE.md.

---

## Project Summary

Thesis: **Edge-Function-Guided Contrastive Learning with KAN Projection Heads
for False-Negative-Aware Multi-Label Chest X-ray Representation Learning**

Package: `cxr_kan` (src-layout, `src/` directory)  
Config system: Hydra (`configs/`)  
Test runner: `pytest tests/`

---

## The 10 Scientific Rules — enforce before writing any code

1. Every KAN result **must** pair with a parameter-matched MLP baseline.
2. Do **not** implement the combined model until baseline losses + tests pass.
3. All contrastive masks **must** have unit tests (positive/negative/FN cases).
4. Dataset splits **must** be patient-level where patient IDs exist.
5. No data leakage — train/val/test patient sets must be disjoint.
6. Experiments **must** be config-driven (Hydra). No hardcoded hyperparams.
7. Every loss **must** return `dict[str, Tensor]` with named components.
8. Every training run saves: config YAML, git hash, metrics CSV/JSON, param count, runtime.
9. No silent fallbacks. Raise descriptive errors.
10. Modules ≤ ~200 lines. Split if larger.

See `CLAUDE.md` for the full table with rationale.

---

## Repository Layout

```
src/cxr_kan/          ← not present yet; src/ root acts as package during dev
  data/               ← dataset, splits, augmentations
  models/             ← encoders, projection heads
    kan/              ← KAN projector variants
  losses/             ← InfoNCE and FN-weighted variants
  metrics/            ← linear probe, kNN, geometry
  utils/              ← logging, checkpoints, reproducibility
configs/              ← Hydra config tree
tests/                ← pytest
scripts/              ← entry-points (train.py, probe.py, ablate.py)
reports/              ← generated artifacts only, never committed raw data
```

---

## Subagent Roles

| Agent file | Owns rules | When to invoke |
|------------|------------|---------------|
| `loss-auditor.md` | 1, 2, 7 | After writing or modifying any loss module |
| `dataset-leakage-checker.md` | 4, 5 | After writing or modifying any dataset/split code |
| `experiment-auditor.md` | 6, 8 | Before marking any training stage complete |
| `pytorch-debugger.md` | — | When hitting tensor shape or NaN errors |
| `code-reviewer.md` | 9, 10 | General correctness review |

---

## Implementation Sequence (Rules 2 + 6)

Never skip stages. Each stage must have passing tests before the next begins.

```
Stage 0:  repo setup ✅
Stage 1:  ChestMNIST smoke data pipeline ✅
Stage 2:  MLP encoder + InfoNCE baseline (parameter-matched) ✅
Stage 3:  linear probe + kNN evaluation harness 🔲  ← current focus
Stage 4:  FastKAN projector (swap in, compare vs Stage 2) ✅
Stage 5:  residual FastKAN warp ✅
Stage 6:  FN-weighted InfoNCE with MLP scorer ✅
          (src/losses/fn_weighted_infonce.py, src/models/pair_scorer.MLPPairScorer)
Stage 7:  FN-weighted InfoNCE with KAN scorer ✅
          (src/models/pair_scorer.KANPairScorer, param-matched within 15 %)
Stage 8:  geometry metrics (alignment, uniformity, intra-class spread) 🔲
Stage 9:  CheXpert full pipeline 🔲
Stage 10: final ablation runner + paper table generation 🔲
```

---

## Critical Constraints for Any Code Written Here

### Loss modules (`src/losses/`)
```python
# Required return signature:
def forward(self, ...) -> dict[str, torch.Tensor]:
    ...
    return {"loss": total, "pos_term": pos, "neg_term": neg}
```
Any loss that returns a plain Tensor (not dict) violates Rule #7.

### Dataset splits (`src/data/`)
- Must accept a `patient_id_col` argument or detect it automatically.
- Must assert train/val/test patient sets are disjoint before returning.
- Raise `ValueError` (not warning) if IDs overlap.

### Config access (scripts and train loop)
- All hyperparameters via `cfg.xxx` (Hydra OmegaConf).
- No `argparse` in training scripts. CLI overrides via Hydra compose API.

### Experiment artifact saving (`src/utils/`)
- Must save within the Hydra output directory (`hydra.runtime.output_dir`).
- Artifacts: `config.yaml`, `git_hash.txt`, `metrics.json`, `param_count.txt`, `runtime_sec.txt`.

---

## Forbidden Patterns

```python
# FORBIDDEN — silent fallback (Rule 9)
try:
    patient_ids = dataset.patient_ids
except:
    patient_ids = None  # silently proceeds

# FORBIDDEN — hardcoded hyperparameter (Rule 6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# FORBIDDEN — loss returns tensor directly (Rule 7)
def forward(self, z1, z2):
    return F.cross_entropy(logits, labels)
```

---

## Testing Requirements

- Every new public function in `src/` must have at least one test in `tests/`.
- Contrastive mask tests must cover: all-positive, all-negative, mixed FN.
- Run `pytest tests/ -v` before committing.

---

## Running the Smoke Test

```bash
pip install -e .
pytest tests/test_smoke.py -v
```
