---
description: Invoke the experiment-auditor subagent to verify a training stage is config-driven and saves full provenance. Run before marking any stage complete.
argument-hint: <stage number or experiment name>
allowed-tools: Read, Grep, Glob
---

Invoke the `experiment-auditor` subagent to verify Rule 6 and Rule 8 compliance for the experiment at Stage $ARGUMENTS (or the most recently modified experiment if no argument given).

Checks:
- Rule 6: no hardcoded hyperparameters; all values from Hydra config
- Rule 8: training run saves config.yaml, git_hash.txt, metrics.json, param_count.txt, runtime_sec.txt

The auditor will scan `scripts/` and `src/utils/` for compliance.

If any FAIL items are returned, fix them before running any training.
