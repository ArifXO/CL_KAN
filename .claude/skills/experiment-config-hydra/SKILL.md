---
name: experiment-config-hydra
description: TRIGGER when the user asks to create, modify, or debug Hydra configs or experiment YAML files, set up the Hydra entry-point in a training script, compose configs programmatically, or override configs via CLI. Also trigger when discussion touches on experiment reproducibility (Rule 6/8).
---

You are a specialist in Hydra + OmegaConf experiment configuration for PyTorch research.

## Config Structure for This Project

```
configs/
  data/
    chestmnist.yaml       # dataset params
    chexpert.yaml
  model/
    mlp_baseline.yaml     # MLP projection head params
    kan_head.yaml         # KAN head params
  loss/
    infonce.yaml          # InfoNCE params
    fn_infonce_mlp.yaml   # FN-InfoNCE with MLP scorer
    fn_infonce_kan.yaml   # FN-InfoNCE with KAN scorer
  experiment/
    smoke_mlp.yaml        # compose: data/chestmnist + model/mlp + loss/infonce
    smoke_kan.yaml        # compose: data/chestmnist + model/kan + loss/infonce
    ablation_full.yaml    # multirun ablation
```

## Experiment Config Pattern

```yaml
# configs/experiment/smoke_mlp.yaml
# @package _global_

defaults:
  - /data: chestmnist
  - /model: mlp_baseline
  - /loss: infonce
  - _self_

# experiment-level overrides
trainer:
  max_epochs: 10
  batch_size: 64
  learning_rate: 1.0e-3
  seed: 42

output_dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

## Training Script Entry-Point (Rule 6)

```python
# scripts/train.py
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../configs", config_name="experiment/smoke_mlp")
def main(cfg: DictConfig) -> None:
    # All hyperparameters come from cfg — never hardcoded
    model = build_model(cfg.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.trainer.learning_rate)
    ...

if __name__ == "__main__":
    main()
```

## Artifact Saving (Rule 8)

```python
from omegaconf import OmegaConf
import subprocess, time, json
from pathlib import Path

def save_run_artifacts(cfg: DictConfig, metrics: dict, param_count: int, runtime: float):
    out = Path(hydra.core.global_hydra.GlobalHydra.instance().get_cfg().hydra.runtime.output_dir)
    
    # Config
    OmegaConf.save(cfg, out / "config.yaml")
    
    # Git hash (Rule 8)
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except subprocess.CalledProcessError:
        raise RuntimeError("Not in a git repo — cannot save git hash (Rule 8)")
    (out / "git_hash.txt").write_text(git_hash)
    
    # Metrics
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    
    # Param count
    (out / "param_count.txt").write_text(str(param_count))
    
    # Runtime
    (out / "runtime_sec.txt").write_text(f"{runtime:.2f}")
```

## CLI Overrides

```bash
# Single override
python scripts/train.py trainer.learning_rate=5e-4

# Multirun ablation (Rule 10 — use ablation config)
python scripts/train.py --multirun trainer.learning_rate=1e-3,5e-4 model=mlp_baseline,kan_head
```

## Common Config Mistakes

| Mistake | Fix |
|---------|-----|
| `lr = 0.001` in Python | Move to `configs/experiment/*.yaml` as `trainer.learning_rate: 1.0e-3` |
| `argparse` in scripts | Replace with `@hydra.main(...)` |
| Relative config path | Use `config_path="../configs"` relative to script location |
| Missing `# @package _global_` | Experiment overrides won't merge correctly without it |
