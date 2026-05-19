# CXR-KAN-Contrastive

**Edge-Function-Guided Contrastive Learning with KAN Projection Heads for
False-Negative-Aware Multi-Label Chest X-ray Representation Learning**

MSc/BSc thesis research codebase.

---

## Overview

This project investigates whether replacing the standard MLP projection head in
contrastive self-supervised learning with a **Kolmogorov-Arnold Network (KAN)**
projection head improves representation quality on multi-label chest X-ray data,
particularly when the contrastive objective is augmented with a
**false-negative-aware masking** mechanism.

### Core Hypotheses

| # | Hypothesis |
|---|-----------|
| H1 | KAN projectors learn smoother latent geometries than parameter-matched MLPs. |
| H2 | False-negative-aware masking improves recall-sensitive downstream metrics. |
| H3 | Edge-function guidance on KAN activations acts as a regularizer for medical images. |

---

## Quick Start

```bash
# 1. Install (editable, with dev extras)
pip install -e ".[dev]"

# 2. Run smoke test
pytest tests/test_smoke.py -v

# 3. Run a Stage 1 smoke data pipeline (once implemented)
python scripts/train.py experiment=smoke_chestmnist
```

---

## Repository Structure

```
configs/        Hydra config tree (data/model/loss/experiment)
src/            Python package `cxr_kan`
  data/           Dataset classes, patient-level splits
  models/         Encoders + projection heads (MLP & KAN)
    kan/            KAN projector variants
  losses/         InfoNCE, FN-weighted InfoNCE, geometry losses
  metrics/        Downstream evaluation (linear probe, kNN, AUC)
  utils/          Logging, checkpointing, reproducibility helpers
tests/          pytest test suite
scripts/        Training/evaluation entry-points
reports/        Generated figures and tables (gitignored raw data)
.claude/        Claude Code subagents, commands, skills
```

---

## Implementation Stages

| Stage | Description | Status |
|-------|-------------|--------|
| 0 | Repo setup | ✅ |
| 1 | ChestMNIST smoke data pipeline | 🔲 |
| 2 | MLP encoder + InfoNCE baseline | 🔲 |
| 3 | Linear probe + kNN evaluation | 🔲 |
| 4 | FastKAN projector | 🔲 |
| 5 | Residual FastKAN warp | 🔲 |
| 6 | FN-weighted InfoNCE + MLP scorer | 🔲 |
| 7 | FN-weighted InfoNCE + KAN scorer | 🔲 |
| 8 | Geometry metrics | 🔲 |
| 9 | CheXpert full pipeline | 🔲 |
| 10 | Final ablation runner | 🔲 |

See `TODO.md` for per-stage task lists.

---

## Datasets

| Dataset | Source | Split strategy |
|---------|--------|---------------|
| ChestMNIST | `medmnist` package | Patient-level (field: `patient_id`) |
| CheXpert | CheXpert paper/download | Patient-level (subject-id prefix) |

Both datasets require patient-level splits (no patient appears in > 1 split).
See Rule #4 in `CLAUDE.md`.

---

## Scientific Rules Summary

All experiments must satisfy the 10 rules in `CLAUDE.md`. The three most
critical for result validity:

- **Rule 1:** Every KAN number has a parameter-matched MLP number beside it.
- **Rule 4/5:** Patient-level splits, no leakage.
- **Rule 7/8:** Every loss is a dict; every run saves its full provenance.

---

## Dependencies

Core: `torch`, `torchvision`, `hydra-core`, `omegaconf`, `medmnist`, `einops`,
`scikit-learn`, `pandas`, `numpy`, `tqdm`

KAN library: determined at Stage 4 (candidate: `efficient-kan`)

Dev: `pytest`, `pytest-cov`

---

## Citing

*(Add thesis citation once submitted.)*
