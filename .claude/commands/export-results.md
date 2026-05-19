---
description: Export paper-ready tables and figures from completed experiment runs. Collects metrics CSVs from Hydra output dirs and generates reports/tables/ and reports/figures/.
argument-hint: <experiment_dir or 'all'>
allowed-tools: Bash(python:*), Read, Glob
---

Export results from completed experiment runs into paper-ready format.

Steps:
1. If $ARGUMENTS is a specific directory, collect metrics from that run.
   If $ARGUMENTS is 'all' or empty, scan all Hydra output directories.
2. Run: `python scripts/export_results.py $ARGUMENTS`
3. Output goes to `reports/tables/` (CSV/LaTeX) and `reports/figures/` (PNG/PDF).

Before running, verify:
- The experiment directory contains `metrics.json` and `config.yaml` (Rule 8).
- A baseline (MLP) run exists alongside any KAN run (Rule 1).
- Git hash is recorded (Rule 8).

Report what was exported and any missing artifacts.
