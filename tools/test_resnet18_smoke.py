"""
Smoke test for ResNet18 — checks forward + backward pass, output shape,
and peak GPU/CPU memory for all 4 dataset configurations.

Run locally (CPU):
    python tools/test_resnet18_smoke.py

Run on cluster (GPU):
    python tools/test_resnet18_smoke.py --device cuda
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn

# Register models
from models_early_fusion import ResNet18  # noqa: F401 — side-effect: registers with MODELS

CONFIGS = [
    # name,        n_markers, patch_size, batch_size
    ('CODEX_cHL',          41, 32, 128),
    ('CODEX_cHL_KRONOS18', 18, 32, 128),
    ('MIBI_TNBC',          37, 32, 128),
    ('IMC_NB_TumorSub',    31, 24, 128),
]


def fmt_mem(bytes_val: int) -> str:
    return f'{bytes_val / 1024**2:.1f} MB'


def run_config(name, n_markers, patch_size, batch_size, device):
    print(f'\n{"─"*60}')
    print(f'  {name}  |  C={n_markers}  H=W={patch_size}  B={batch_size}')
    print(f'{"─"*60}')

    model = ResNet18(in_channels=n_markers, drop_prob=0.05).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters : {n_params:,}')

    x = torch.randn(batch_size, n_markers, patch_size, patch_size, device=device)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Forward
    out = model(x)
    assert len(out) == 1, f'Expected 1 output, got {len(out)}'
    feat = out[0]
    assert feat.shape == (batch_size, 512, 1, 1), \
        f'Expected ({batch_size}, 512, 1, 1), got {feat.shape}'
    print(f'  Output shape: {tuple(feat.shape)}  ✓')

    # Backward (simulate VICReg loss = mean of features)
    loss = feat.mean()
    loss.backward()
    print(f'  Backward    : ok  ✓')

    if device == 'cuda':
        peak = torch.cuda.max_memory_allocated()
        print(f'  Peak GPU mem: {fmt_mem(peak)}')
    else:
        import tracemalloc
        tracemalloc.stop()

    del model, x, feat
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()

    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, falling back to CPU')
        device = 'cpu'

    print(f'Device: {device}')
    print(f'PyTorch: {torch.__version__}')

    for cfg in CONFIGS:
        run_config(*cfg, device=device)

    print(f'\n{"="*60}')
    print('  All smoke tests passed!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
