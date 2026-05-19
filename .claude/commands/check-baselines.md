---
description: Verify that every KAN component has a parameter-matched MLP baseline config and test. Critical for Rule 1 — run before starting any KAN-specific stage (4+).
allowed-tools: Read, Grep, Glob
---

Check that every KAN model variant has a parameter-matched MLP baseline.

For each file found in `src/models/kan/`:
1. Find the corresponding MLP baseline (same role, similar parameter count).
2. Check that `configs/model/` has both a KAN config and an MLP config with matching architecture dims.
3. Check that `tests/` has parameter count tests verifying parity.

Report:
- Which KAN variants have a verified MLP baseline
- Which are missing a baseline (BLOCKER — Rule 1)
- Parameter count comparison (if tests exist)

This check must pass before any KAN results are reported.
