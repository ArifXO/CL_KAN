---
name: thesis-results-export
description: TRIGGER when the user asks to generate paper tables, export results CSVs, create plots/figures for the thesis, or run the final ablation comparison. Also trigger when the user says "generate LaTeX table", "plot embeddings", or "export results".
---

You are a specialist in research results export for ML thesis work.

## What Gets Exported

Every exported result must have a paired MLP baseline (Rule 1).
Every row in a table must come from a run with saved provenance (Rule 8).

## Table Generation Pattern

```python
# scripts/export_results.py
import pandas as pd
from pathlib import Path
import json

def collect_run_metrics(output_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in output_root.rglob("metrics.json"):
        metrics = json.loads(run_dir.read_text())
        config = OmegaConf.load(run_dir.parent / "config.yaml")
        git_hash = (run_dir.parent / "git_hash.txt").read_text().strip()
        param_count = int((run_dir.parent / "param_count.txt").read_text())
        rows.append({
            "model": config.model._target_,
            "loss": config.loss._target_,
            "params": param_count,
            "git_hash": git_hash[:8],
            **metrics,
        })
    return pd.DataFrame(rows)
```

## Required Table Columns (Thesis)

| Column | Source |
|--------|--------|
| Model | `config.model._target_` |
| Projector | `kan` or `mlp` |
| Loss | `infonce` or `fn_infonce` |
| Params (K) | `param_count.txt` |
| AUC (macro) | `metrics.json` |
| AUC (micro) | `metrics.json` |
| kNN Acc | `metrics.json` |
| Linear Probe Acc | `metrics.json` |
| Alignment ↓ | `metrics.json` |
| Uniformity ↓ | `metrics.json` |
| Git Hash | `git_hash.txt` |

## LaTeX Table Skeleton

```python
def to_latex(df: pd.DataFrame) -> str:
    return df.to_latex(
        index=False,
        float_format="{:.3f}".format,
        caption="Ablation results on ChestMNIST.",
        label="tab:ablation",
        escape=False,
    )
```

## Figures

Standard plots for this thesis:
1. **Loss curve** — training loss vs epoch for KAN vs MLP
2. **Embedding UMAP** — colored by label, for each model variant
3. **AUC bar chart** — per-label AUC comparison
4. **Alignment/Uniformity scatter** — one point per run variant

Save as PDF for thesis, PNG for presentation:
```python
fig.savefig(out / "fig_name.pdf", bbox_inches="tight", dpi=300)
fig.savefig(out / "fig_name.png", bbox_inches="tight", dpi=150)
```

## Sanity Checks Before Exporting

1. Every KAN row has a paired MLP row with matching param count (Rule 1).
2. Every row's `git_hash` exists in the repo (Rule 8).
3. No row is missing any required column.
4. Val and test metrics are kept separate — never mix them.
