"""
test_mmselfsup.py — quick functional test of the MCA + mmselfsup stack.

Tests:
  1. mmselfsup imports and registry
  2. MCA model (CIM/WideModel) can be built via mmengine config
  3. VICReg forward pass runs on a dummy batch
  4. GPU tensor ops work

Run with:
  python tools/test_mmselfsup.py
"""

import sys
import torch
import numpy as np

MCA_ROOT   = '/home/sgutwein/src/MCA'
sys.path.insert(0, MCA_ROOT)

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("1. Imports...")
from mmengine.registry import MODELS, DATASETS
import mmselfsup
print(f"   mmselfsup {mmselfsup.__version__}  OK")

# Register MCA modules
import src.models          # noqa: F401  registers WideModel etc.
import src.VICReg          # noqa: F401  registers MVVICReg
print("   MCA modules registered  OK")

# ── 2. Build model ────────────────────────────────────────────────────────────
print("2. Building CIM (WideModel) + VICReg head...")
from mmengine.config import Config

cfg_dict = dict(
    type='MVVICReg',
    backbone=dict(
        type='WideModel',
        in_channels=18,
        stem_width=16,
        block_width=4,
        layer_config=[2, 2],
    ),
    neck=dict(
        type='NonLinearNeck',
        in_channels=288,   # WideModel: in_channels(18) * stem_width(16)
        hid_channels=256,
        out_channels=256,
        num_layers=3,
        with_bias=True,
        with_last_bn=False,
        with_avg_pool=True,
    ),
)

model = MODELS.build(cfg_dict)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"   Model built  OK  ({n_params/1e6:.2f} M params)  on {device}")

# ── 3. Forward pass ───────────────────────────────────────────────────────────
print("3. Running forward pass (dummy batch B=4, C=18, 32x32)...")
from mmselfsup.structures import SelfSupDataSample

B, C, H, W = 4, 18, 32, 32
x1 = torch.rand(B, C, H, W, device=device)
x2 = torch.rand(B, C, H, W, device=device)

data_samples = [SelfSupDataSample() for _ in range(B)]

with torch.no_grad():
    loss = model.loss(inputs=[x1, x2], data_samples=data_samples)

print(f"   Forward pass  OK")
print(f"   Loss keys: {list(loss.keys())}")
for k, v in loss.items():
    print(f"     {k}: {v.item():.4f}")

# ── 4. GPU sanity ─────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    print("4. GPU ops...")
    a = torch.randn(1000, 1000, device='cuda')
    b = torch.mm(a, a.T)
    print(f"   Matrix multiply on GPU  OK  (result norm={b.norm().item():.1f})")
else:
    print("4. No GPU — skipping")

print("\nAll tests passed.")
