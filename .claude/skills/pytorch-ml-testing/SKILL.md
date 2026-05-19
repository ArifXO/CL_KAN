---
name: pytorch-ml-testing
description: TRIGGER when the user asks to write, fix, or improve pytest tests for PyTorch modules — loss functions, dataset classes, model forward passes, masks, or evaluation metrics. Also trigger when a test is failing with tensor shape or NaN issues.
---

You are a specialist in pytest-based testing for PyTorch ML research code.

## Testing Philosophy for This Project

- Tests prove *contracts*, not implementations. A loss test verifies the dict
  contract and gradient flow — not the exact loss value.
- Contrastive mask tests must cover three cases: all-positive, all-negative, mixed.
- Shape tests use `assert out.shape == (B, D)` — always check both dims.
- Use small synthetic tensors (`B=4, D=8`) for speed.

## Canonical Test Patterns

### Loss dict contract test
```python
def test_infonce_returns_dict():
    loss_fn = InfoNCELoss(temperature=0.1)
    z = F.normalize(torch.randn(8, 64), dim=-1)
    out = loss_fn(z[:4], z[4:])
    assert isinstance(out, dict), "Loss must return dict (Rule 7)"
    assert "loss" in out
    assert out["loss"].shape == ()       # scalar
    assert out["loss"].requires_grad     # differentiable
```

### Gradient flow test
```python
def test_loss_gradient_flows():
    z = F.normalize(torch.randn(8, 64, requires_grad=True), dim=-1)
    out = loss_fn(z[:4], z[4:])
    out["loss"].backward()
    assert z.grad is not None
    assert not torch.isnan(z.grad).any()
```

### Contrastive mask test (Rule 3)
```python
@pytest.mark.parametrize("case", ["all_positive", "all_negative", "mixed"])
def test_mask_construction(case):
    if case == "all_positive":
        y = torch.ones(4, 3)      # all samples share all labels
    elif case == "all_negative":
        y = torch.eye(4, 3)       # each sample has unique labels
    else:
        y = torch.randint(0, 2, (4, 3)).float()
    mask = build_contrastive_mask(y)
    assert mask.shape == (4, 4)
    assert mask.dtype == torch.bool
    if case == "all_positive":
        assert mask.all()
    if case == "all_negative":
        assert not mask.fill_diagonal_(False).any()
```

### Patient-level split test (Rules 4, 5)
```python
def test_patient_level_split_no_leakage():
    train_ids, val_ids, test_ids = get_patient_splits(dataset)
    assert len(set(train_ids) & set(val_ids)) == 0, "Train/val leakage"
    assert len(set(train_ids) & set(test_ids)) == 0, "Train/test leakage"
    assert len(set(val_ids) & set(test_ids)) == 0, "Val/test leakage"
```

### Parameter count parity (Rule 1)
```python
def test_kan_mlp_param_parity():
    kan = KANHead(in_dim=128, out_dim=64, **kan_cfg)
    mlp = MLPHead(in_dim=128, out_dim=64, **mlp_cfg)
    kan_n = sum(p.numel() for p in kan.parameters())
    mlp_n = sum(p.numel() for p in mlp.parameters())
    # allow ≤5% difference
    ratio = abs(kan_n - mlp_n) / max(kan_n, mlp_n)
    assert ratio <= 0.05, f"Param mismatch: KAN={kan_n}, MLP={mlp_n}"
```

## pytest Fixtures for This Project

```python
# tests/conftest.py
import pytest, torch

@pytest.fixture
def batch_embeddings():
    torch.manual_seed(42)
    return torch.nn.functional.normalize(torch.randn(16, 128), dim=-1)

@pytest.fixture
def multi_label_targets():
    torch.manual_seed(42)
    return torch.randint(0, 2, (16, 14)).float()  # 14 CheXpert labels
```

## Running Tests

```bash
pytest tests/ -v                    # all tests
pytest tests/test_losses.py -v      # specific file
pytest -k "mask" -v                 # tests matching "mask"
pytest --tb=short -q                # quiet, short traceback
```
