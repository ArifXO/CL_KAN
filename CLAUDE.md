# CLAUDE.md — CXR-KAN-Contrastive

## Thesis

**Edge-Function-Guided Contrastive Learning with KAN Projection Heads for
False-Negative-Aware Multi-Label Chest X-ray Representation Learning**

This repository implements and ablates a contrastive learning framework for
chest X-ray representation learning. The core claims are:

1. KAN projection heads learn smoother, more disentangled latent geometries
   than parameter-matched MLPs on multi-label medical images.
2. False-negative-aware contrastive masking improves recall-sensitive metrics.
3. Edge-function guidance regularizes the KAN's learned activation splines.

---

## Scientific Non-Negotiables (enforce every session)

These rules are load-bearing. Violating them invalidates experimental claims.

| # | Rule |
|---|------|
| 1 | Every KAN result **must** be compared against a **parameter-matched MLP baseline** run under identical conditions. |
| 2 | Do **not** implement the combined model before baseline losses and their tests pass. |
| 3 | All contrastive masks **must** have unit tests covering positive, negative, and false-negative cases. |
| 4 | Dataset splits **must** be patient-level where patient IDs exist (ChestMNIST via `patient_id`; CheXpert via subject-id prefix). |
| 5 | **No data leakage** — verify train/val/test patient disjointness before any training run. |
| 6 | Experiments **must** be config-driven (Hydra). No hard-coded hyperparameters in Python source. |
| 7 | Every loss function **must** return a `dict` of named scalar components (e.g. `{"loss": ..., "pos_term": ..., "neg_term": ...}`). |
| 8 | Every training run **must** save: config snapshot (YAML), git commit hash, metrics CSV/JSON, parameter count, and wall-clock runtime. |
| 9 | **No silent fallbacks.** Raise clear, descriptive errors. No bare `except: pass`. |
| 10 | Keep modules small and independently testable. Max ~200 lines per file; split otherwise. |

---

## Repository Layout

```
cxr-kan-contrastive/          ← repo root (this directory)
├── configs/                  ← Hydra config tree (data/model/loss/experiment)
├── src/                      ← importable Python package `cxr_kan`
│   ├── data/                 ← dataset classes, splits, augmentations
│   ├── models/               ← encoders, projection heads
│   │   └── kan/              ← KAN projector variants
│   ├── losses/               ← InfoNCE, FN-weighted variants, geometry losses
│   ├── metrics/              ← linear probe, kNN, AUC, geometry metrics
│   └── utils/                ← logging, checkpointing, reproducibility
├── tests/                    ← pytest test suite
├── scripts/                  ← training entry-points (train.py, probe.py, ablate.py)
├── reports/                  ← generated figures and tables (never committed raw data)
│   ├── figures/
│   └── tables/
└── .claude/                  ← Claude Code configuration
    ├── settings.json
    ├── agents/
    ├── commands/
    └── skills/
```

---

## Development Stages (see TODO.md)

Stage 0 → repo setup  
Stage 1 → ChestMNIST smoke pipeline  
Stage 2 → MLP + InfoNCE baseline  
Stage 3 → linear probe + kNN eval  
Stage 4 → FastKAN projector  
Stage 5 → residual FastKAN warp  
Stage 6 → FN-weighted InfoNCE + MLP scorer  
Stage 7 → FN-weighted InfoNCE + KAN scorer  
Stage 8 → geometry metrics  
Stage 9 → CheXpert pipeline  
Stage 10 → final ablation runner  

**Rule #2:** Do not start Stage N+1 until Stage N has passing tests.

---

## Coding Conventions

- **Package name:** `cxr_kan` (maps to `src/`)
- **Config system:** Hydra + OmegaConf. All hyperparameters in `configs/`.
- **Loss returns:** always `dict` — see Rule #7.
- **Errors:** use `raise ValueError(f"...")` with a human-readable message.
- **Comments:** only when the *why* is non-obvious. No docstring novels.
- **Tests:** `pytest tests/` must pass before any merge.
- **Type hints:** use them on function signatures; not required in bodies.

---

## Subagents

Use `.claude/agents/` subagents for specialized review:

- `loss-auditor` — verify loss dict contracts and baseline parity (Rules 1, 2, 7)
- `dataset-leakage-checker` — verify patient-level splits, no leakage (Rules 4, 5)
- `experiment-auditor` — verify config-driven runs, artifact saving (Rules 6, 8)
- `pytorch-debugger` — diagnose tensor shape errors and training instability
- `code-reviewer` — general correctness and Rule 10 (module size)

---

## Key External Dependencies

| Library | Purpose |
|---------|---------|
| `torch` / `torchvision` | core DL |
| `hydra-core` / `omegaconf` | config management |
| `medmnist` | ChestMNIST dataset |
| `einops` | readable tensor ops |
| `scikit-learn` | linear probe, kNN |
| `pandas` | metrics CSV |
| KAN library | TBD at Stage 4 (likely `efficient-kan`) |

---
## Preferred implementation order

1. ChestMNIST smoke loader
2. MLP projector + InfoNCE
3. linear probe and kNN evaluation
4. FastKAN projector
5. residual FastKAN warp
6. FN-weighted InfoNCE with MLP pair scorer
7. FN-weighted InfoNCE with KAN pair scorer
8. geometry metrics
9. CheXpert loader
10. full ablation runner

## Code style

- PyTorch-first.
- Type hints where useful.
- Small modules.
- No notebooks as source of truth.
- Tests before large experiments.
- Return dictionaries from training/evaluation functions.
- Use clear error messages.

## Dangerous areas

Be extra skeptical with:
- contrastive positive/negative masks
- false-negative weights
- label-overlap computation
- patient-level split
- AUROC/mAP calculation
- parameter count matching
- KAN output scaling
- all negatives being downweighted
- embedding collapse

## Git Hygiene

- Every commit message must reference the active Stage (e.g., `[Stage2]`).
- Tag milestone commits: `git tag stage-N-complete`.
- Do not commit model checkpoints or raw data. See `.gitignore`.
