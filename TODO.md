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

## Stage 1 — ChestMNIST Smoke Data Pipeline 🔲

- [ ] `src/data/chestmnist.py` — ChestMNIST dataset wrapper using `medmnist`
- [ ] `src/data/splits.py` — patient-level train/val/test split logic
- [ ] `src/data/augmentations.py` — augmentation pipeline (SimCLR-style)
- [ ] `src/data/__init__.py` — expose `get_dataloader(cfg)`
- [ ] `configs/data/chestmnist.yaml` — full config
- [ ] `tests/test_data_splits.py` — assert patient disjointness
- [ ] `tests/test_data_pipeline.py` — smoke: load batch, check shapes
- [ ] All Stage 1 tests pass

**Acceptance:** `pytest tests/test_data_*.py -v` all green.

---

## Stage 2 — MLP Encoder + InfoNCE Baseline 🔲

- [ ] `src/models/encoder.py` — ResNet-18 encoder wrapper
- [ ] `src/models/mlp_head.py` — 2-layer MLP projection head
- [ ] `src/losses/infonce.py` — standard InfoNCE, returns dict (Rule #7)
- [ ] `src/utils/param_count.py` — count and log parameter totals
- [ ] `configs/model/mlp_baseline.yaml`
- [ ] `configs/loss/infonce.yaml`
- [ ] `configs/experiment/smoke_mlp.yaml`
- [ ] `scripts/train.py` — Hydra entry-point
- [ ] `tests/test_infonce.py` — loss dict contract, gradient flow, numerical stability
- [ ] `tests/test_mlp_head.py` — output shape, param count
- [ ] Baseline training run completes on CPU with smoke config
- [ ] Run artifacts saved (Rule #8): config, git hash, metrics, param count, runtime

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

## Stage 4 — FastKAN Projector 🔲

- [ ] Decide on KAN library (`efficient-kan` vs `fastkan`) — document choice
- [ ] `src/models/kan/kan_head.py` — KAN projection head
- [ ] `src/models/kan/__init__.py`
- [ ] Verify parameter-matched config vs MLP head (Rule #1)
- [ ] `configs/model/kan_head.yaml`
- [ ] `configs/experiment/smoke_kan.yaml`
- [ ] `tests/test_kan_head.py` — shape, param parity with MLP baseline
- [ ] Side-by-side comparison run: KAN vs MLP under identical config

**Acceptance:** `pytest tests/test_kan_head.py -v` all green + param counts logged.

---

## Stage 5 — Residual FastKAN Warp 🔲

- [ ] `src/models/kan/residual_kan_head.py` — residual warp variant
- [ ] `configs/model/residual_kan_head.yaml`
- [ ] `tests/test_residual_kan.py`
- [ ] Ablation: KAN vs residual KAN vs MLP

**Acceptance:** `pytest tests/test_residual_kan.py -v` all green.

---

## Stage 6 — FN-Weighted InfoNCE + MLP Scorer 🔲

- [ ] `src/losses/fn_infonce.py` — FN-weighted InfoNCE loss (dict return)
- [ ] `src/models/fn_scorer_mlp.py` — MLP false-negative scorer
- [ ] `src/data/label_graph.py` — co-occurrence / label similarity matrix
- [ ] `tests/test_fn_mask.py` — mask unit tests (Rule #3): all-positive, all-negative, mixed FN
- [ ] `tests/test_fn_infonce.py` — loss dict contract, gradient flow
- [ ] `configs/loss/fn_infonce_mlp.yaml`

**Acceptance:** `pytest tests/test_fn_*.py -v` all green.

---

## Stage 7 — FN-Weighted InfoNCE + KAN Scorer 🔲

- [ ] `src/models/fn_scorer_kan.py` — KAN false-negative scorer
- [ ] Verify parameter-matched vs MLP scorer (Rule #1)
- [ ] `configs/loss/fn_infonce_kan.yaml`
- [ ] `tests/test_fn_scorer_kan.py`
- [ ] Side-by-side: KAN scorer vs MLP scorer

**Acceptance:** `pytest tests/test_fn_scorer_kan.py -v` all green.

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
