"""
test_external_models.py — functional test of external model baselines.

Tests DINOv2 and OpenPhenom with dummy input.
UNI is tested with random weights (no gated HF access needed).

Run on a GPU node:
  python ~/test_external_models.py
"""

import sys
import torch
import torch.nn.functional as F

MCA_ROOT = '/home/sgutwein/src/MCA'
sys.path.insert(0, MCA_ROOT)

# Point torch hub to our local clones
import torch
torch.hub.set_dir('/home/sgutwein/src')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}\n")

B, C, H, W = 4, 18, 32, 32
dummy = torch.rand(B, C, H, W)

PASS = "OK  "
FAIL = "FAIL"

def check(name, fn):
    try:
        result = fn()
        print(f"  [{PASS}] {name:<40} {result}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {name:<40} {e}")
        return False

# ── DINOv2 ────────────────────────────────────────────────────────────────────
print("=" * 60)
print("DINOv2")
print("=" * 60)

def test_dinov2():
    from src.models_external import DINOv2Backbone
    # DINOv2 was trained at 518×518 (patch 14 → 37×37=1369 patches/channel)
    model = DINOv2Backbone(variant='dinov2_vitb14', img_size=518, freeze=True).to(DEVICE).eval()
    with torch.no_grad():
        out = model(dummy.to(DEVICE))[0]   # [B, 768, 1, 1]
    return f"input={tuple(dummy.shape)}  output={tuple(out.shape)}  dim={model.out_channels}"

check("DINOv2-ViT-B/14 forward pass", test_dinov2)

# ── OpenPhenom ────────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("OpenPhenom")
print("=" * 60)

def test_ophenphenom_load():
    from src.models_external import OpenPhenomBackbone
    model = OpenPhenomBackbone(img_size=256, freeze=True).to(DEVICE).eval()
    return f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M  dim={model.out_channels}"

def test_ophenphenom_forward():
    from src.models_external import OpenPhenomBackbone
    model = OpenPhenomBackbone(img_size=256, freeze=True).to(DEVICE).eval()
    # Full 18-channel input — chunked internally into groups of <=11
    with torch.no_grad():
        out = model(dummy.to(DEVICE))[0]   # [B, 384, 1, 1]
    return f"input={tuple(dummy.shape)}  output={tuple(out.shape)}  (per-channel, averaged over 18)"

check("OpenPhenom load", test_ophenphenom_load)
check("OpenPhenom forward pass", test_ophenphenom_forward)

# ── UNI (mock weights — no gated HF access needed) ───────────────────────────
print()
print("=" * 60)
print("UNI (random weights — architecture test only)")
print("=" * 60)

def test_uni_arch():
    import timm
    from src.models_external import _adapt_patch_embed_to_1channel, _channel_agnostic_vit_forward, _per_channel_instance_norm
    model = timm.create_model(
        'vit_large_patch16_224', img_size=224, patch_size=16,
        init_values=1e-5, num_classes=0, dynamic_img_size=True,
    ).eval()
    model.patch_embed.proj = _adapt_patch_embed_to_1channel(model.patch_embed.proj)
    for p in model.parameters():
        p.requires_grad_(False)
    x = F.interpolate(dummy, size=(224, 224), mode='bilinear', align_corners=False)
    x = _per_channel_instance_norm(x)
    with torch.no_grad():
        cls = _channel_agnostic_vit_forward(model, x, model_type='timm')
    return f"input={tuple(dummy.shape)}  output={tuple(cls.shape)}  (random weights)"

check("UNI ViT-L/16 channel-agnostic forward", test_uni_arch)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print("Done.")
