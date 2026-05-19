---
description: Run the full smoke test suite (install + pytest test_smoke.py). Use after setup or after major refactors to verify the package installs and imports cleanly.
allowed-tools: Bash(pip:*), Bash(pytest:*)
---

Run the project smoke tests to verify the package installs and the basic import chain works.

Steps:
1. Install the package in editable mode: `pip install -e .`
2. Run the smoke test: `pytest tests/test_smoke.py -v`
3. Report pass/fail and any import errors.

If the install fails, check `pyproject.toml` for missing fields.
If the import fails, check `src/__init__.py` and all sub-package `__init__.py` files.
