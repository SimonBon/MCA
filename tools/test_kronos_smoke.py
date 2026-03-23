"""
Smoke test for KRONOS integration.

Downloads KRONOS vits16 from HuggingFace (first run only, cached afterwards)
and verifies it runs correctly on cell-patch-shaped inputs.

Input requirements:
  - [B, C, H, W] float32, H = W and both divisible by 16 (ViT internal patch_size)
  - 32×32 → 2×2 = 4 spatial tokens per marker (minimum sensible size)
  - 64×64 → 4×4 = 16 tokens per marker (as used in KRONOS cell tutorials)
  - Positional embeddings are bicubic-interpolated from training 14×14 grid → any size works

Run locally (CPU, small batch):
    python tools/test_kronos_smoke.py --kronos_src /Users/simon.gutwein/src/KRONOS

Run on cluster (GPU):
    python tools/test_kronos_smoke.py \\
        --kronos_src /home/sgutwein/src/KRONOS \\
        --cache_dir  /nobackup/lab_taschner-mandl/simongutwein/model_assets/kronos
"""

import argparse
import sys
import torch
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--kronos_src', default='/Users/simon.gutwein/src/KRONOS',
                   help='Path to KRONOS repo root')
    p.add_argument('--checkpoint', default='hf_hub:MahmoodLab/kronos',
                   help='Local .pt path or hf_hub:MahmoodLab/kronos')
    p.add_argument('--cache_dir',  default='./model_assets',
                   help='Cache dir for HF download')
    p.add_argument('--hf_token',   default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.kronos_src not in sys.path:
        sys.path.insert(0, args.kronos_src)

    from kronos import create_model_from_pretrained

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load model ────────────────────────────────────────────────────────────
    print(f'Loading KRONOS from {args.checkpoint} ...')
    model, precision, embed_dim = create_model_from_pretrained(
        checkpoint_path=args.checkpoint,
        cfg={'model_type': 'vits16', 'token_overlap': False},
        hf_auth_token=args.hf_token,
        cache_dir=args.cache_dir,
    )
    model.eval().to(device)
    print(f'  Model loaded. embed_dim={embed_dim}, precision={precision}')

    # ── Input size tests ──────────────────────────────────────────────────────
    # Requirements: H = W, divisible by 16
    # 32×32 → 2×2 = 4 spatial tokens/marker  (our cell patches)
    # 64×64 → 4×4 = 16 spatial tokens/marker (KRONOS tutorial)
    test_cases = [
        # (batch, n_markers, H, W, description)
        (4,  41, 32, 32, 'CODEX_cHL full panel (32px, our patches)'),
        (4,  18, 32, 32, 'KRONOS18 panel (32px)'),
        (2,  41, 64, 64, 'CODEX_cHL full panel (64px, KRONOS tutorial size)'),
        (2,   1, 32, 32, 'Single marker'),
    ]

    print()
    all_ok = True
    for B, C, H, W, desc in test_cases:
        try:
            # Simulate KRONOS tutorial pipeline:
            # 1. Raw uint16 intensities → /65535 → (x - mean) / std
            # 2. Multiply by binary cell mask
            # 3. Pass with marker_ids as [B, C] int64 tensor
            x = torch.randint(0, 65535, (B, C, H, W)).to(dtype=precision) / 65535.0
            mean = torch.rand(C, dtype=precision) * 0.5
            std  = torch.rand(C, dtype=precision) * 0.3 + 0.1
            x = (x - mean[None, :, None, None]) / std[None, :, None, None]

            # Binary cell mask: random ellipse-ish
            cell_mask = torch.ones(B, H, W, dtype=precision)
            cell_mask[:, :H//4, :] = 0  # zero out some border pixels
            x = x * cell_mask.unsqueeze(1)

            x = x.to(device)

            # marker_ids as [B, C] int64 tensor (as per KRONOS tutorial DataLoader)
            marker_ids = torch.tensor(
                list(range(4, 4 + C)), dtype=torch.int64
            ).unsqueeze(0).expand(B, -1).to(device)

            with torch.no_grad():
                patch_feats, marker_feats, token_feats = model(x, marker_ids=marker_ids)

            n_tokens_per_marker = (H // 16) * (W // 16)
            flat_marker = marker_feats.reshape(B, -1)

            assert patch_feats.shape  == (B, embed_dim),                  f'patch_feats: {patch_feats.shape}'
            assert marker_feats.shape == (B, C, embed_dim),               f'marker_feats: {marker_feats.shape}'
            assert token_feats.shape  == (B, C, H//16, W//16, embed_dim), f'token_feats: {token_feats.shape}'
            assert flat_marker.shape  == (B, C * embed_dim),              f'flat_marker: {flat_marker.shape}'

            print(f'  OK  [{B},{C},{H},{W}]'
                  f'  patch_feats={tuple(patch_feats.shape)}'
                  f'  marker_feats={tuple(marker_feats.shape)} → flat={tuple(flat_marker.shape)}'
                  f'  ({n_tokens_per_marker} tokens/marker)'
                  f'  | {desc}')
        except Exception as e:
            print(f'  FAIL [{B},{C},{H},{W}] — {e}  | {desc}')
            all_ok = False

    # ── Check non-square / non-divisible (should fail) ────────────────────────
    print()
    fail_cases = [
        (2, 5, 30, 30, '30×30 — not divisible by 16'),
        (2, 5, 32, 16, '32×16 — non-square (interpolation assumes square)'),
    ]
    for B, C, H, W, desc in fail_cases:
        try:
            x = torch.randn(B, C, H, W, dtype=precision).to(device)
            mids = [torch.tensor(list(range(4, 4+C)), device=device) for _ in range(B)]
            with torch.no_grad():
                model(x, marker_ids=mids)
            print(f'  UNEXPECTED OK [{B},{C},{H},{W}]  | {desc}')
        except Exception as e:
            print(f'  Expected fail [{B},{C},{H},{W}]: {type(e).__name__}  | {desc}')

    print()
    print('Smoke test', 'PASSED' if all_ok else 'FAILED')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
