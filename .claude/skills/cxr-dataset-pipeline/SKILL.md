---
name: cxr-dataset-pipeline
description: TRIGGER when the user asks to implement, debug, or configure chest X-ray dataset loading, augmentation pipelines, or train/val/test splits. Covers ChestMNIST (medmnist) and CheXpert. Also trigger when discussing patient-level splitting or label imbalance handling.
---

You are a specialist in chest X-ray dataset pipelines for contrastive learning research.

## Datasets in This Project

### ChestMNIST (Stage 1)
- Source: `medmnist` package (`medmnist.ChestMNIST`)
- Resolution: 28×28 grayscale (can upsample to 224×224)
- Labels: 14 pathologies (multi-label binary)
- Patient IDs: available as metadata field `patient_id`
- Train/val/test: use `medmnist` official splits BUT re-group by `patient_id`

```python
from medmnist import ChestMNIST
dataset = ChestMNIST(split="train", download=True, size=224)
# Access patient_id via dataset.info or metadata CSV
```

### CheXpert (Stage 9)
- Download: requires user agreement; provide path via config
- Labels: 14 labels, uncertain (`-1`) requires handling policy
- Patient IDs: extracted from image path prefix (e.g., `patient12345/study1/...`)
- Uncertainty policy: options are `ignore`, `positive`, `negative`, `LSR`

## Patient-Level Splitting (Rules 4 & 5)

```python
def patient_level_split(
    patient_ids: list[str],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    unique_patients = sorted(set(patient_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_patients)
    n = len(unique_patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = set(unique_patients[:n_train])
    val = set(unique_patients[n_train:n_train + n_val])
    test = set(unique_patients[n_train + n_val:])
    # Verify disjointness (Rule 5)
    assert len(train & val) == 0
    assert len(train & test) == 0
    assert len(val & test) == 0
    return train, val, test
```

## SimCLR Augmentation Pipeline (Stage 1)

```python
import torchvision.transforms as T

def get_contrastive_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),  # grayscale CXR
    ])
```

Note: For chest X-rays, be conservative with color jitter (grayscale images).
RandomHorizontalFlip is standard; avoid vertical flip (anatomical meaning).

## Label Handling

ChestMNIST: binary, no uncertainty. Straightforward.

CheXpert uncertainty policy (set in config, not code):
```yaml
# configs/data/chexpert.yaml
uncertainty_policy: ignore  # or: positive, negative, LSR
```

The dataset class raises `ValueError` if policy is not one of the valid options
(Rule #9 — no silent fallbacks).

## DataLoader Pattern

```python
def get_dataloader(cfg: DictConfig) -> dict[str, DataLoader]:
    # Returns dict with keys: "train", "val", "test"
    ...
    return {"train": train_loader, "val": val_loader, "test": test_loader}
```

Always return a dict, not a tuple — callers shouldn't depend on positional order.
