"""External pretrained feature extractors for multi-channel multiplexed imaging.

Three models, three adaptation strategies:

  OpenPhenom (Recursion)
    Natively channel-agnostic MAE trained on 6-channel microscopy (Cell Painting).
    Works out of the box — just resize patches to 256×256 and call predict().
    Weights: recursionpharma/OpenPhenom on HuggingFace (free, no gating).

  DINOv2 (Meta)
    Self-supervised ViT (vitb14 / vitl14) trained on natural images.
    Adapted via channel-agnostic patch embedding: a single Conv2d(1→D)
    is applied to each marker independently, producing C×N spatial tokens
    that are fed jointly through the frozen ViT transformer.
    Pretrained patch weights (3-channel) are averaged to initialise the
    1-channel projection — all transformer weights are kept frozen.
    Weights: free, downloaded automatically via torch.hub.

  UNI (Mahmood Lab)
    Pathology ViT (ViT-L/16) trained on 200M H&E/IHC images.
    Same channel-agnostic adaptation as DINOv2.
    Weights: gated on HuggingFace — request access at MahmoodLab/UNI first,
    then set HF_TOKEN in your environment.

All backbones follow the MCA backbone interface:
  Input:  [B, C, H, W]  float32 (any number of channels C)
  Output: ([B, D, 1, 1],)  tuple (compatible with VICReg neck / val_hook_rich)
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

    This removes absolute intensity variation across samples — the same
    normalisation Recursion uses in OpenPhenom and that is critical for
    multiplexed imaging where per-patient staining intensity varies.
    Input/output: [B, C, H, W] float32.
    """
    B, C, H, W = x.shape
    # Flatten spatial dims, compute mean/std per channel per batch item
    x_flat = x.view(B * C, 1, H, W)
    x_norm = F.instance_norm(x_flat)          # zero-mean, unit-std per channel
    return x_norm.view(B, C, H, W)


def _adapt_patch_embed_to_1channel(proj: nn.Conv2d) -> nn.Conv2d:
    """Replace a k-channel patch projection with a 1-channel version.

    The new Conv2d(1, out, kH, kW) is initialised by averaging the
    pretrained k-channel weights along the input channel dimension.
    This preserves the pretrained spatial filter structure.

    Args:
        proj: pretrained Conv2d(k, out, kH, kW) — e.g. Conv2d(3, 768, 14, 14)
    Returns:
        new Conv2d(1, out, kH, kW) with averaged weights, no grad (frozen).
    """
    out_channels, in_channels, kH, kW = proj.weight.shape
    new_proj = nn.Conv2d(1, out_channels, kernel_size=(kH, kW),
                         stride=proj.stride, padding=proj.padding, bias=proj.bias is not None)
    with torch.no_grad():
        new_proj.weight.copy_(proj.weight.mean(dim=1, keepdim=True))
        if proj.bias is not None:
            new_proj.bias.copy_(proj.bias)
    return new_proj


# ─────────────────────────────────────────────────────────────────────────────
# Channel-agnostic ViT forward (shared by DINOv2 and UNI)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _channel_agnostic_vit_forward(model, x: torch.Tensor,
                                  model_type: str = 'dinov2') -> torch.Tensor:
    """Run a ViT in channel-agnostic mode on multi-channel input.

    Applies the (adapted) single-channel patch_embed.proj to each marker
    independently, then concatenates C×N spatial tokens and runs the full
    ViT. Returns the CLS token [B, D].

    Works for both DINOv2 (DinoVisionTransformer) and UNI/timm ViTs.

    Args:
        model:      ViT with patch_embed.proj already replaced by Conv2d(1→D).
        x:          [B, C, H, W] float32, normalised per-channel.
        model_type: 'dinov2' or 'timm'
    """
    B, C, H, W = x.shape

    # 1. Apply 1-channel proj to each marker → collect spatial tokens
    if model_type == 'dinov2':
        proj = model.patch_embed.proj
    else:  # timm ViT (UNI)
        proj = model.patch_embed.proj

    patch_tokens_list = []
    for c in range(C):
        p = proj(x[:, c:c+1])             # [B, D, h_p, w_p]
        patch_tokens_list.append(
            p.flatten(2).transpose(1, 2)  # [B, N, D]
        )
    patch_tokens = torch.cat(patch_tokens_list, dim=1)  # [B, C*N, D]

    # 2. Positional embeddings: tile the spatial pos embed C times
    #    (channels share the same spatial grid — they are differentiated
    #     purely by their content, not by position)
    if model_type == 'dinov2':
        cls_pos    = model.pos_embed[:, :1]        # [1, 1, D]
        spatial_pos = model.pos_embed[:, 1:]       # [1, N, D]
        spatial_pos = spatial_pos.repeat(1, C, 1)  # [1, C*N, D]
        cls_token  = model.cls_token.expand(B, -1, -1)

        x_seq = torch.cat([cls_token, patch_tokens], dim=1)           # [B, 1+C*N, D]
        x_seq = x_seq + torch.cat([cls_pos, spatial_pos], dim=1)      # add pos

        # Register tokens (DINOv2 registers)
        if getattr(model, 'register_tokens', None) is not None:
            reg = model.register_tokens.expand(B, -1, -1)
            x_seq = torch.cat([x_seq[:, :1], reg, x_seq[:, 1:]], dim=1)

        for blk in model.blocks:
            x_seq = blk(x_seq)
        x_seq = model.norm(x_seq)
        return x_seq[:, 0]  # CLS token [B, D]

    else:  # timm ViT (UNI)
        cls_pos     = model.pos_embed[:, :1]       # [1, 1, D]
        spatial_pos = model.pos_embed[:, 1:]       # [1, N, D]
        spatial_pos = spatial_pos.repeat(1, C, 1)  # [1, C*N, D]
        cls_token   = model.cls_token.expand(B, -1, -1)

        x_seq = torch.cat([cls_token, patch_tokens], dim=1)
        x_seq = x_seq + torch.cat([cls_pos, spatial_pos], dim=1)

        x_seq = model.norm_pre(x_seq) if hasattr(model, 'norm_pre') else x_seq
        x_seq = model.blocks(x_seq)
        x_seq = model.norm(x_seq)
        return x_seq[:, 0]  # CLS token [B, D]


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

        # Add maes_microscopy to path so we can import MAEModel
        maes_dir = os.path.join(os.path.dirname(__file__), '../../maes_microscopy')
        maes_dir = os.path.abspath(maes_dir)
        if maes_dir not in sys.path:
            sys.path.insert(0, maes_dir)

        from huggingface_mae import MAEModel

        # If hf_model_path looks like a HF repo id (no path separator),
        # download it to a local cache first so from_pretrained works
        # regardless of huggingface_hub version.
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
               Supports any number of channels C by chunking into groups
               of <= 11 and averaging the resulting embeddings.
        Returns:
            ([B, 384, 1, 1],)
        """
        B, C = x.shape[:2]

        # Resize to model's expected input size
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)

        # OpenPhenom expects uint8 [0,255]
        if x.max() <= 1.0:
            x = (x * 255).clamp(0, 255)
        x = x.to(torch.uint8)

        # Embed each channel independently and average.
        # More principled than chunking: symmetric treatment of all markers,
        # no grouping artifact, uses model in its intended single-channel regime.
        per_channel = [self.model.predict(x[:, c:c+1]) for c in range(C)]  # C × [B, 384]
        feat = torch.stack(per_channel, dim=0).mean(dim=0)                 # [B, 384]
        return (feat.view(B, self.out_channels, 1, 1),)


# ─────────────────────────────────────────────────────────────────────────────
# DINOv2 backbone
# ─────────────────────────────────────────────────────────────────────────────

class DINOv2Backbone(nn.Module):
    """DINOv2 ViT with channel-agnostic patch embedding for multiplexed imaging.

    The pretrained 3-channel patch projection (Conv2d(3, D, 14, 14)) is
    replaced with a 1-channel version (Conv2d(1, D, 14, 14)) initialised
    from the average of the 3 RGB filter weights. All transformer weights
    (attention, FFN, norms) are kept frozen at their pretrained values.

    The ViT then processes C*N spatial tokens (one spatial grid per marker)
    jointly through its self-attention layers, enabling cross-marker context.

    Args:
        variant:  'dinov2_vits14' | 'dinov2_vitb14' | 'dinov2_vitl14' | 'dinov2_vitg14'
        img_size: resize input to this square size (must be multiple of 14)
        freeze:   freeze all weights after adaptation (default: True)
    """

    EMBED_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }

    def __init__(self, variant: str = 'dinov2_vitb14',
                 img_size: int = 224, freeze: bool = True,
                 repo_path: str = None):
        super().__init__()
        assert img_size % 14 == 0, f"img_size must be a multiple of 14, got {img_size}"
        self.img_size = img_size
        self.out_channels = self.EMBED_DIMS[variant]

        # repo_path: absolute path to a local dinov2 clone.
        # Falls back to ~/src/dinov2, then torch hub (online download).
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

        # Adapt patch embed: Conv2d(3→D) → Conv2d(1→D)
        self.vit.patch_embed.proj = _adapt_patch_embed_to_1channel(
            self.vit.patch_embed.proj
        )

        self.vit.eval()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, C, H, W] float32
        Returns:
            ([B, D, 1, 1],)
        """
        B = x.shape[0]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        x = _per_channel_instance_norm(x)
        feat = _channel_agnostic_vit_forward(self.vit, x, model_type='dinov2')
        return (feat.view(B, self.out_channels, 1, 1),)


# ─────────────────────────────────────────────────────────────────────────────
# UNI backbone
# ─────────────────────────────────────────────────────────────────────────────

class UNIBackbone(nn.Module):
    """UNI pathology ViT with channel-agnostic patch embedding.

    Same channel adaptation strategy as DINOv2Backbone: the 3-channel
    patch projection is replaced with a 1-channel version, all transformer
    weights remain frozen. UNI is ViT-L/16 trained on 200M pathology images.

    Requires HuggingFace access — request at https://huggingface.co/MahmoodLab/UNI
    then set HF_TOKEN env variable or call `huggingface_hub.login()` once.

    Args:
        ckpt_path: path to downloaded pytorch_model.bin from MahmoodLab/UNI
        img_size:  resize input (must be multiple of 16, default 224)
        freeze:    freeze all weights after adaptation (default: True)
    """

    def __init__(self, ckpt_path: str, img_size: int = 224, freeze: bool = True):
        super().__init__()
        assert img_size % 16 == 0, f"img_size must be a multiple of 16, got {img_size}"
        self.img_size = img_size
        self.out_channels = 1024  # ViT-L embed_dim

        import timm
        self.vit = timm.create_model(
            'vit_large_patch16_224',
            img_size=img_size,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        state_dict = torch.load(ckpt_path, map_location='cpu')
        self.vit.load_state_dict(state_dict, strict=True)

        # Adapt patch embed: Conv2d(3→1024) → Conv2d(1→1024)
        self.vit.patch_embed.proj = _adapt_patch_embed_to_1channel(
            self.vit.patch_embed.proj
        )

        self.vit.eval()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: [B, C, H, W] float32
        Returns:
            ([B, 1024, 1, 1],)
        """
        B = x.shape[0]
        if x.shape[-1] != self.img_size or x.shape[-2] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        x = _per_channel_instance_norm(x)
        feat = _channel_agnostic_vit_forward(self.vit, x, model_type='timm')
        return (feat.view(B, self.out_channels, 1, 1),)
