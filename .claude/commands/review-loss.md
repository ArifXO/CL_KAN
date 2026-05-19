---
description: Invoke the loss-auditor subagent to check all loss modules for Rule 1/2/7 compliance. Use after writing or modifying src/losses/ or any scorer.
argument-hint: [optional: path to specific loss file]
allowed-tools: Read, Grep, Glob
---

Invoke the `loss-auditor` subagent to audit loss modules.

If $ARGUMENTS is provided, audit only that file. Otherwise audit all files in `src/losses/` and any scorer files in `src/models/`.

The auditor checks:
- Rule 1: KAN results have parameter-matched MLP baselines
- Rule 2: combined model not implemented before baselines pass
- Rule 7: all loss `forward()` methods return `dict[str, Tensor]`

Report the full audit output including any FAIL items that must be fixed before proceeding.
