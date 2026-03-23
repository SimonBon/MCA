"""External pretrained feature extractors for multi-channel multiplexed imaging.

Three models, three adaptation strategies:

  OpenPhenom (Recursion)
    Natively channel-agnostic MAE trained on 6-channel microscopy (Cell Painting).
    Works out of the box — just resize patches to 256×256 and call predict().
    Weights: recursionpharma/OpenPhenom on HuggingFace (free, no gating).

  DINOv2 (Meta)  /  UNI (Mahmood Lab)
    Per the KRONOS paper (arXiv:2506.03373):
      Each marker channel is replicated to 3×RGB and passed independently
      through the frozen ViT. The CLS token is extracted per marker and
      concatenated → feature vector of size D × M, where D is the ViT embed
      dim and M is the number of markers.
    This gives NO cross-marker interaction — each marker is processed in
    isolation and the representations are concatenated post-hoc.
    Output dimension scales with M: 1024×M for DINOv2-L / UNI.

All backbones follow the MCA backbone interface:
  Input:  [B, C, H, W]  float32 (any number of channels C)
  Output: ([B, D*C, 1, 1],)  tuple (compatible with VICReg neck / val_hook_rich)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ─────────────────────────────────────────────────────────────────────────────

def _per_channel_instance_norm(x: torch.Tensor) -> torch.Tensor:
    """Self-standardise each channel independently (zero-mean, unit-std per cell).

    Removes absolute intensity variation across samples — critical for
    multiplexed imaging where per-patient staining intensity varies.
    Input/output: [B, C, H, W] float32.
    """
    B, C, H, W = x.shape
    x_flat = x.view(B * C, 1, H, W)
    x_norm = F.instance_norm(x_flat)
    return x_norm.view(B, C, H, W)


def _interp_pos_embed(spatial_pos: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
    """Bicubic-interpolate spatial pos_embed [1, N_orig, D] → [1, h_p*w_p, D]."""
    N_orig = spatial_pos.shape[1]
    if N_orig == h_p * w_p:
        return spatial_pos
    D = spatial_pos.shape[2]
    h_orig = w_orig = int(N_orig ** 0.5)
    pos = spatial_pos.reshape(1, h_orig, w_orig, D).permute(0, 3, 1, 2)
    pos = F.interpolate(pos.float(), size=(h_p, w_p), mode='bicubic', align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, h_p * w_p, D).to(spatial_pos.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Per-marker CLS extraction (KRONOS paper approach for DINOv2 / UNI)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _single_channel_cls_dinov2(model, x_rgb: torch.Tensor) -> torch.Tensor:
    """Standard DINOv2 forward on [B, 3, H, W] → CLS token [B, D].

    Uses the model's own patch_embed (Conv2d 3→D) unchanged, with bicubic
    pos-embed interpolation for non-native resolutions.
    """
    B, _, H, W = x_rgb.shape
    ps = model.patch_embed.patch_size
    ps = ps[0] if isinstance(ps, (tuple, list)) else ps
    h_p, w_p = H // ps, W // ps

    patch_tokens = model.patch_embed(x_rgb)               # [B, N, D]
    cls_pos      = model.pos_embed[:, :1]                  # [1, 1, D]
    spatial_pos  = _interp_pos_embed(model.pos_embed[:, 1:], h_p, w_p)  # [1, N, D]
    cls_token    = model.cls_token.expand(B, -1, -1)       # [B, 1, D]

    x_seq = torch.cat([cls_token, patch_tokens], dim=1)
    x_seq = x_seq + torch.cat([cls_pos, spatial_pos], dim=1)

    if getattr(model, 'register_tokens', None) is not None:
        reg = model.register_tokens.expand(B, -1, -1)
        x_seq = torch.cat([x_seq[:, :1], reg, x_seq[:, 1:]], dim=1)

    for blk in model.blocks:
        x_seq = blk(x_seq)
    x_seq = model.norm(x_seq)
    return x_seq[:, 0]   # CLS token [B, D]


@torch.no_grad()
def _single_channel_cls_timm(model, x_rgb: torch.Tensor) -> torch.Tensor:
    """Standard timm ViT forward on [B, 3, H, W] → CLS token [B, D].

    Works for UNI (ViT-L/16, num_classes=0). timm's forward() returns the
    class token directly when num_classes=0 and global_pool='token'.
    """
    return model(x_rgb)   # [B, D]


@torch.no_grad()
def _per_marker_cls_concat(model, x: torch.Tensor, model_type: str = 'dinov2') -> torch.Tensor:
    """Per-marker CLS token extraction + concatenation (KRONOS paper, Fig. S2).

    Each marker channel is replicated to 3×RGB and passed independently through
    the frozen ViT. CLS tokens are extracted per marker and concatenated.

    Args:
        model:      frozen ViT with original 3-channel patch_embed
        x:          [B, C, H, W] float32, normalised per-channel
        model_type: 'dinov2' or 'timm'

    Returns:
        [B, C*D] — concatenated per-marker CLS tokens
    """
    B, C, H, W = x.shape
    cls_tokens = []
    for c in range(C):
        x_c = x[:, c:c+1].repeat(1, 3, 1, 1)  # [B, 3, H, W]  — replicate to RGB
        if model_type == 'dinov2':
            cls_c = _single_channel_cls_dinov2(model, x_c)
        else:
            cls_c = _single_channel_cls_timm(model, x_c)
        cls_tokens.append(cls_c)                # [B, D]
    return torch.cat(cls_tokens, dim=1)         # [B, C*D]


# ─────────────────────────────────────────────────────────────────────────────
# OpenPhenom backbone
# ─────────────────────────────────────────────────────────────────────────────

class OpenPhenomBackbone(nn.Module):
    """Recursion OpenPhenom — channel-agnostic MAE trained on microscopy.

    Native multi-channel support: the pretrained ViT-S/16 uses a single
    Conv2d(1→384) patch projection applied independently to each channel,
    then attends over all C×256 tokens jointly (for 256×256 input).

    Output dimension: 384 (global avg-pool over all tokens).

    Weights downloaded automatically from HuggingFace on first use:
        huggingface-cli download recursionpharma/OpenPhenom --local-dir <dir>
    or set hf_model_path to a local directory.

    Args:
        hf_model_path: local path or HF repo id (default: 'recursionpharma/OpenPhenom')
        img_size:      resize input to this square size (default: 256)
        freeze:        if True, freeze all weights (default: True)
    """

    def __init__(self, hf_model_path: str = 'recursionpharma/OpenPhenom',
                 img_size: int = 256, freeze: bool = True):
        super().__init__()
        self.img_size = img_size

        maes_dir = os.path.join(os.path.dirname(__file__), '../../maes_microscopy')
        maes_dir = os.path.abspath(maes_dir)
        if maes_dir not in sys.path:
            sys.path.insert(0, maes_dir)

        from huggingface_mae import MAEModel

        if not os.path.isdir(hf_model_path):
            from huggingface_hub import snapshot_download
            hf_model_path = snapshot_download(repo_id=hf_model_path)

        self.model = MAEModel.from_pretrained(hf_model_path)
        self.model.eval()
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        self.out_channels = 384  # ViT-S embed_dim

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, C, H, W] float32, values in [0, 1] or any scale.
               Supports any number of channels C by embedding each channel
               independently and averaging the resulting embeddings.
        Returns:
            ([B, 384, 1, 1],)
        """
        B, C = x.shape[:2]

        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        if x.max() <= 1.0:
            x = (x * 255).clamp(0, 255)
        x = x.to(torch.uint8)

        per_channel = [self.model.predict(x[:, c:c+1]) for c in range(C)]
        feat = torch.stack(per_channel, dim=0).mean(dim=0)
        return (feat.view(B, self.out_channels, 1, 1),)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 backbone
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Backbone(nn.Module):
    """DINOv2 ViT for multiplexed imaging, following the KRONOS paper protocol.

    Each marker channel is replicated to 3×RGB and passed independently through
    the frozen, unmodified ViT. The CLS token is extracted per marker and all
    tokens are concatenated → [B, D*C, 1, 1].

    This matches the DINOv2 baseline in the KRONOS paper (arXiv:2506.03373):
      "Each marker channel is individually replicated to 3×RGB and passed
       through the model. The CLS token is extracted per marker and
       concatenated → feature vector of size 1024×M."

    The pretrained patch_embed weights (Conv2d 3→D) are kept unchanged.
    No cross-marker interaction occurs — markers are processed independently.

    Args:
        variant:  'dinov2_vits14' | 'dinov2_vitb14' | 'dinov2_vitl14' | 'dinov2_vitg14'
        img_size: input size (must be multiple of 14; center-cropped per paper)
        freeze:   freeze all weights (default: True)
    """

    EMBED_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }

    def __init__(self, variant: str = 'dinov2_vitl14',
                 img_size: int = 56, freeze: bool = True,
                 repo_path: str = None):
        super().__init__()
        assert img_size % 14 == 0, f"img_size must be a multiple of 14, got {img_size}"
        self.img_size   = img_size
        self.embed_dim  = self.EMBED_DIMS[variant]
        # out_channels reports the per-marker dim; actual output is embed_dim * C
        self.out_channels = self.embed_dim

        if repo_path is None:
            candidate = os.path.expanduser('~/src/dinov2')
            if os.path.isdir(candidate):
                repo_path = candidate

        if repo_path is not None:
            self.vit = torch.hub.load(
                repo_path, variant, pretrained=True,
                source='local', trust_repo=True,
            )
        else:
            self.vit = torch.hub.load(
                'facebookresearch/dinov2', variant, pretrained=True,
            )

        # patch_embed is kept unchanged (Conv2d 3→D)
        self.vit.eval()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, C, H, W] float32 — any number of markers C
        Returns:
            ([B, embed_dim*C, 1, 1],)
        """
        B, C = x.shape[0], x.shape[1]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        x = _per_channel_instance_norm(x)
        feat = _per_marker_cls_concat(self.vit, x, model_type='dinov2')  # [B, C*D]
        return (feat.view(B, C * self.embed_dim, 1, 1),)


# ─────────────────────────────────────────────────────────────────────────────
# UNI backbone
# ─────────────────────────────────────────────────────────────────────────────

class UNIBackbone(nn.Module):
    """UNI pathology ViT for multiplexed imaging, following the KRONOS paper protocol.

    Each marker channel is replicated to 3×RGB and passed independently through
    the frozen ViT-L/16. The CLS token per marker is concatenated → [B, 1024*C, 1, 1].

    This matches the UNI baseline in the KRONOS paper (arXiv:2506.03373):
      "Each marker channel is individually replicated to 3×RGB and passed
       through the model. The CLS token is extracted per marker and
       concatenated → feature vector of size 1024×M."

    The pretrained patch_embed weights (Conv2d 3→1024) are kept unchanged.
    No cross-marker interaction occurs — markers are processed independently.

    Requires HuggingFace access — request at https://huggingface.co/MahmoodLab/UNI
    then set HF_TOKEN env variable or call `huggingface_hub.login()` once.

    Args:
        ckpt_path: path to downloaded pytorch_model.bin from MahmoodLab/UNI
        img_size:  input size (must be multiple of 16; center-cropped per paper)
        freeze:    freeze all weights (default: True)
    """

    def __init__(self, ckpt_path: str, img_size: int = 64, freeze: bool = True):
        super().__init__()
        assert img_size % 16 == 0, f"img_size must be a multiple of 16, got {img_size}"
        self.img_size     = img_size
        self.embed_dim    = 1024  # ViT-L
        self.out_channels = 1024  # per-marker dim; actual output is 1024*C

        import timm
        self.vit = timm.create_model(
            'vit_large_patch16_224',
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.vit.load_state_dict(state_dict, strict=True)

        # patch_embed is kept unchanged (Conv2d 3→1024)
        self.vit.eval()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, C, H, W] float32 — any number of markers C
        Returns:
            ([B, 1024*C, 1, 1],)
        """
        B, C = x.shape[0], x.shape[1]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        x = _per_channel_instance_norm(x)
        feat = _per_marker_cls_concat(self.vit, x, model_type='timm')   # [B, C*1024]
        return (feat.view(B, C * self.embed_dim, 1, 1),)
