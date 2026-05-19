# TODO.md — Implementation Milestones

Each stage must have **passing tests** before the next stage begins (Rule #2).
Check boxes are updated manually as work completes.

---

## Stage 0 — Repo Setup ✅

- [x] Directory structure created
- [x] CLAUDE.md, AGENTS.md, README.md written
- [x] pyproject.toml, requirements.txt, .gitignore written
- [x] .claude/ config (agents, commands, skills, settings.json) written
- [x] src/ package stubs created
- [x] configs/ example YAMLs created
- [x] tests/test_smoke.py passes
- [x] git repository initialized

---

## Stage 1 — ChestMNIST Smoke Data Pipeline ✅

- [x] `src/data/chestmnist.py` — ChestMNIST dataset wrapper using `medmnist`
- [x] `src/data/splits.py` — patient-level train/val/test split logic
- [x] `src/data/augmentations.py` — augmentation pipeline (SimCLR-style)
- [x] `src/data/__init__.py` — expose `get_dataloader(cfg)`
- [x] `configs/data/chestmnist.yaml` — full config
- [x] `tests/test_data_splits.py` — assert patient disjointness
- [x] `tests/test_data_pipeline.py` — smoke: load batch, check shapes
- [x] All Stage 1 tests pass

**Acceptance:** `pytest tests/test_data_*.py -v` all green.

---

## Stage 2 — MLP Encoder + InfoNCE Baseline ✅

- [x] `src/models/encoder.py` — ResNet-18 encoder wrapper
- [x] `src/models/mlp_head.py` — 2-layer MLP projection head
- [x] `src/losses/infonce.py` — standard InfoNCE, returns dict (Rule #7)
- [x] `src/utils/param_count.py` — count and log parameter totals
- [x] `configs/model/mlp_baseline.yaml`
- [x] `configs/loss/infonce.yaml`
- [x] `configs/experiment/smoke_mlp.yaml`
- [x] `scripts/train.py` — Hydra entry-point
- [x] `tests/test_infonce.py` — loss dict contract, gradient flow, numerical stability
- [x] `tests/test_mlp_head.py` — output shape, param count
- [x] Baseline training run completes on CPU with smoke config
- [x] Run artifacts saved (Rule #8): config, git hash, metrics, param count, runtime

**Acceptance:** `pytest tests/test_infonce.py tests/test_mlp_head.py -v` all green.

---

## Stage 3 — Linear Probe + kNN Evaluation 🔲

- [ ] `src/metrics/linear_probe.py` — fit sklearn LogisticRegression on frozen encoder
- [ ] `src/metrics/knn.py` — kNN classifier on embeddings
- [ ] `src/metrics/auc.py` — macro/micro AUC for multi-label
- [ ] `scripts/probe.py` — evaluation entry-point
- [ ] `configs/experiment/probe.yaml`
- [ ] `tests/test_metrics.py` — probe + kNN on synthetic embeddings
- [ ] Evaluation harness produces `reports/tables/` CSV

**Acceptance:** `pytest tests/test_metrics.py -v` all green.

---

## Stage 4 — FastKAN Projector ✅

- [x] Decide on KAN library (`efficient-kan` vs `fastkan`) — document choice
- [x] `src/models/kan/kan_head.py` — KAN projection head
- [x] `src/models/kan/__init__.py`
- [x] Verify parameter-matched config vs MLP head (Rule #1)
- [x] `configs/model/kan_head.yaml`
- [x] `configs/experiment/smoke_kan.yaml`
- [x] `tests/test_kan_head.py` — shape, param parity with MLP baseline
- [x] Side-by-side comparison run: KAN vs MLP under identical config

**Acceptance:** `pytest tests/test_kan_head.py -v` all green + param counts logged.

---

## Stage 5 — Residual FastKAN Warp ✅

- [x] `src/models/kan/residual_kan_head.py` — residual warp variant
- [x] `configs/model/residual_kan_head.yaml`
- [x] `tests/test_residual_kan.py`
- [x] Ablation: KAN vs residual KAN vs MLP

**Acceptance:** `pytest tests/test_residual_kan.py -v` all green.

---

## Stage 6 — FN-Weighted InfoNCE + MLP Scorer ✅

- [x] `src/losses/fn_weighted_infonce.py` — FNWeightedInfoNCELoss (dict return, Rule #7)
- [x] `src/models/pair_scorer.py` — MLPPairScorer (false-negative scorer)
- [x] `tests/test_pair_scorer.py` — shape, bounds, gradient, validation errors
- [x] `tests/test_fn_weighted_loss.py` — Rule #3 positive/negative/FN cases, monotonicity, gradient flow
- [x] `configs/loss/fn_weighted_mlp.yaml`

**Acceptance:** `pytest tests/test_pair_scorer.py tests/test_fn_weighted_loss.py -v` all green (40 tests).

---

## Stage 7 — FN-Weighted InfoNCE + KAN Scorer ✅

- [x] `KANPairScorer` added to `src/models/pair_scorer.py` (FastKAN-based, same API as MLPPairScorer)
- [x] Parameter-matched vs MLP scorer within 15 % (Rule #1): MLP H=32 ≈ 1 089 params; KAN H=4,K=8 ≈ 1 193 params
- [x] `configs/loss/fn_weighted_kan.yaml`
- [x] `tests/test_kan_pair_scorer.py` — same contract tests + param parity assertion + interchangeability test
- [x] Interchangeability verified: KAN scorer plugged into FNWeightedInfoNCELoss, finite grads on all params

**Acceptance:** `pytest tests/test_kan_pair_scorer.py -v` all green (17 tests).

---

## Stage 8 — Geometry Metrics 🔲

- [ ] `src/metrics/geometry.py` — alignment loss, uniformity loss, intra-class spread
- [ ] `src/metrics/embedding_viz.py` — UMAP/t-SNE for reports
- [ ] `tests/test_geometry.py` — known-case checks for alignment/uniformity
- [ ] `scripts/analyze_geometry.py` — run on saved checkpoints

**Acceptance:** `pytest tests/test_geometry.py -v` all green.

---

## Stage 9 — CheXpert Pipeline 🔲

- [ ] `src/data/chexpert.py` — CheXpert dataset wrapper
- [ ] Update `src/data/splits.py` for subject-id prefix splitting
- [ ] Verify patient-level disjointness (Rule #4/5)
- [ ] `configs/data/chexpert.yaml`
- [ ] `tests/test_chexpert_splits.py`
- [ ] Full training run on CheXpert with best Stage 7 config

**Acceptance:** `pytest tests/test_chexpert_splits.py -v` all green.

---

## Stage 10 — Final Ablation Runner 🔲

- [ ] `scripts/ablate.py` — multi-run ablation launcher (Hydra multirun)
- [ ] `scripts/export_results.py` — generate paper-ready tables and figures
- [ ] `configs/experiment/ablation_full.yaml`
- [ ] `reports/tables/` — final CSV results
- [ ] `reports/figures/` — final plots

**Acceptance:** all ablation runs reproduce, tables exportable in one command.

---

## Cross-Cutting Items (any stage)

- [ ] CI setup (GitHub Actions) — `pytest` on push
- [ ] `src/utils/reproducibility.py` — seed everything, log env info
- [ ] `src/utils/checkpoint.py` — save/load with full artifact bundle (Rule #8)
- [ ] `src/utils/logging.py` — structured experiment logger
