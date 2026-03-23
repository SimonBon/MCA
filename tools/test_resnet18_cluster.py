"""
Cluster smoke test for ResNet18 — mirrors test_mmselfsup.py pattern.

Run on cluster:
    python tools/test_resnet18_cluster.py
"""
import sys
import torch

MCA_ROOT = '/home/sgutwein/src/MCA'
sys.path.insert(0, MCA_ROOT)

from src.models_early_fusion import ResNet18

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device : {device}')
print(f'PyTorch: {torch.__version__}')

CONFIGS = [
    ('CODEX_cHL',          41, 32),
    ('CODEX_cHL_KRONOS18', 18, 32),
    ('MIBI_TNBC',          37, 32),
    ('IMC_NB_TumorSub',    31, 24),
]

all_ok = True

for name, n_markers, patch_size in CONFIGS:
    print(f'\n{"─"*60}')
    print(f'  {name}  |  C={n_markers}  H=W={patch_size}')
    print(f'{"─"*60}')

    try:
        model = ResNet18(in_channels=n_markers, drop_prob=0.05).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f'  Model build : ok  ({n_params:,} params)')
    except Exception as e:
        print(f'  Model build : FAILED — {e}')
        all_ok = False
        continue

    try:
        B = 128
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        x = torch.randn(B, n_markers, patch_size, patch_size, device=device)
        out = model(x)
        assert out[0].shape == (B, 512, 1, 1), f'Bad shape: {out[0].shape}'
        out[0].mean().backward()
        if device == 'cuda':
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f'  Forward/bwd : ok  (peak GPU: {peak_mb:.0f} MB)')
        else:
            print(f'  Forward/bwd : ok')
        del model, x, out
        if device == 'cuda':
            torch.cuda.empty_cache()
    except Exception as e:
        print(f'  Forward/bwd : FAILED — {e}')
        all_ok = False

print(f'\n{"="*60}')
if all_ok:
    print('  ALL CHECKS PASSED — safe to submit training jobs')
else:
    print('  SOME CHECKS FAILED — fix before submitting')
print(f'{"="*60}')
sys.exit(0 if all_ok else 1)
