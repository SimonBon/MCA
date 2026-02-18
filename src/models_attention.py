import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.registry import MODELS


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Residual block: depthwise spatial conv → GroupNorm → pointwise expansion
    → GELU → pointwise projection → optional dropout → residual add.

    Uses GroupNorm (not BatchNorm) for stability on small patches / small
    batches, which is common in cell-level datasets.
    """
    def __init__(self, channels: int, expansion: int = 2, drop_prob: float = 0.0):
        super().__init__()
        hidden = channels * expansion
        self.dw   = nn.Conv2d(channels, channels, 3, padding=1,
                              groups=channels, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.pw1  = nn.Conv2d(channels, hidden, 1, bias=False)
        self.act  = nn.GELU()
        self.pw2  = nn.Conv2d(hidden, channels, 1, bias=False)
        self.drop = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dw(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.drop(x)
        return x + identity


def make_downsample(channels: int) -> nn.Sequential:
    """
    Stride-2 depthwise conv (spatial ↓2) + pointwise mix + norm + activation.
    """
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=3, stride=2,
                  padding=1, groups=channels, bias=False),
        nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        nn.GroupNorm(min(8, channels), channels),
        nn.GELU(),
    )


# ---------------------------------------------------------------------------
# Spatial Focus Map
# ---------------------------------------------------------------------------

class SpatialFocusMap(nn.Module):
    """
    Learns a soft [0, 1] spatial attention mask per biological channel.

    Applied after the first spatial downsampling so local spatial context has
    already been extracted before we decide which pixels are cell body vs.
    neighbour contamination.

    Gaussian logit initialisation
    ------------------------------
    The final logit map is initialised so that sigmoid(logits) ≈ gaussian(y,x)
    at the very start of training.  Concretely:

        spatial_bias[y, x] = logit(gaussian(y, x))
                           = log(g / (1 - g))   where g = exp(-r² / 2σ²)

    This is a learnable parameter that is ADDED to the data-driven conv output.
    The data-driven pathway (context_conv → logit_conv) is zero-initialised so
    it starts completely silent — the Gaussian prior dominates early training
    and the data-driven path only activates as gradients accumulate.

    At step 0 the mask already looks like a centred Gaussian blob.  The model
    deviates from this prior only when the task loss demands it.

    Collapse / saturation detection
    --------------------------------
    If mean(mask) < warn_threshold  → mask collapsed toward zero  (signal loss)
    If mean(mask) > 1 - warn_threshold → mask saturated toward one (no focus)
    Warnings fire at most once per warn_every forward calls.

    Parameters
    ----------
    stem_dim       : feature channels fed into this module
    spatial_size   : H (= W) of the feature map (e.g. 12 for 24x24 input
                     after a stride-2 stem conv)
    sigma_fraction : Gaussian σ as a fraction of spatial_size.  Default 0.35
                     → σ ≈ 4.2 px on 12x12, falling to ~0.14 at the corners.
    warn_threshold : emit warning if mean mask outside [t, 1-t]
    warn_every     : minimum forward calls between consecutive warnings
    """

    def __init__(
        self,
        stem_dim       : int   = 32,
        spatial_size   : int   = 12,
        sigma_fraction : float = 0.35,
        warn_threshold : float = 0.05,
        warn_every     : int   = 200,
        init_mode      : str   = 'gaussian',  # 'gaussian' or 'ones'
    ):
        super().__init__()

        self.warn_threshold = warn_threshold
        self.warn_every     = warn_every
        self._call_count    = 0
        self._last_warned   = -warn_every

        self.context_conv = nn.Conv2d(
            stem_dim, stem_dim // 2, kernel_size=3, padding=1, bias=False
        )
        self.act = nn.GELU()

        # bias=True so we can set it spatially via the reshape trick below
        self.logit_conv = nn.Conv2d(stem_dim // 2, 1, kernel_size=1, bias=False)

        # Zero out all weights — data-driven path starts completely silent
        self.context_conv.weight = nn.init.zeros_(self.context_conv.weight)
        self.logit_conv.weight = nn.init.zeros_(self.logit_conv.weight)
        
        setattr(self.context_conv, '_skip_init', True)
        setattr(self.logit_conv, '_skip_init', True)

        # ── Spatial bias: the only thing speaking at init ────────────────────
        if init_mode == 'gaussian':
            # sigmoid(spatial_bias[y,x]) == gaussian(y,x) at step 0.
            # We compute the Gaussian on the grid, clamp to avoid ±inf,
            # then invert through logit: logit(g) = log(g / (1-g))
            sigma = sigma_fraction * spatial_size
            cy = (spatial_size - 1) / 2.0
            cx = (spatial_size - 1) / 2.0
            ys = torch.arange(spatial_size, dtype=torch.float32) - cy
            xs = torch.arange(spatial_size, dtype=torch.float32) - cx
            gy, gx = torch.meshgrid(ys, xs)
            gauss = torch.exp(-(gy**2 + gx**2) / (2 * sigma**2))  # [H, W]
            gauss = gauss.clamp(0.01, 0.99)
            logit_init = torch.log(gauss / (1.0 - gauss))          # [H, W]

        elif init_mode == 'ones':
            # sigmoid(large positive) ≈ 1 → mask starts at ~1 everywhere.
            # logit(0.99) ≈ 4.6, a comfortable positive constant.
            logit_init = torch.full(
                (spatial_size, spatial_size), fill_value=4.6
            )

        else:
            raise ValueError(f"init_mode must be 'gaussian' or 'ones', got '{init_mode}'")

        # Registered as a learnable parameter: [1, 1, H, W]
        # Broadcasts over batch and feature dims in forward.
        self.register_parameter(
            'spatial_bias',
            nn.Parameter(logit_init.unsqueeze(0).unsqueeze(0))
        )

    def forward(self, x: torch.Tensor):
        """
        x    : [B, stem_dim, H, W]
        mask : [B, 1,        H, W]  ∈ [0, 1]
        out  : [B, stem_dim, H, W]  — x re-weighted by mask
        """
        self._call_count += 1

        # Data-driven logits (zero at init, grows during training)
        ctx    = self.act(self.context_conv(x))   # [B, stem_dim//2, H, W]
        logits = self.logit_conv(ctx)             # [B, 1, H, W]

        # Add Gaussian prior (broadcasts over batch dimension)
        logits = logits + self.spatial_bias       # [B, 1, H, W]

        mask = torch.sigmoid(logits)              # [B, 1, H, W]  ∈ [0, 1]

        # ── Collapse / saturation detection (training only) ──────────────────
        if self.training:
            mean_val    = mask.detach().mean().item()
            since_last  = self._call_count - self._last_warned

            if since_last >= self.warn_every:
                if mean_val < self.warn_threshold:
                    self._last_warned = self._call_count
                    warnings.warn(
                        f"[SpatialFocusMap] Mask COLLAPSE detected — "
                        f"mean activation = {mean_val:.4f} < {self.warn_threshold}. "
                        f"Most of the spatial feature map is suppressed, which will "
                        f"starve downstream layers of signal. "
                        f"Fixes: reduce lambda_c in compactness loss, increase "
                        f"sigma_fraction, or check for exploding gradients.",
                        stacklevel=2,
                    )
                elif mean_val > 1.0 - self.warn_threshold:
                    self._last_warned = self._call_count
                    warnings.warn(
                        f"[SpatialFocusMap] Mask SATURATION detected — "
                        f"mean activation = {mean_val:.4f} > {1-self.warn_threshold:.2f}. "
                        f"The mask is uniformly ~1 (no spatial selectivity). "
                        f"The module may not be contributing any focus learning.",
                        stacklevel=2,
                    )

        return x * mask, mask   # mask broadcasts over stem_dim


# ---------------------------------------------------------------------------
# Cross-Channel Attention
# ---------------------------------------------------------------------------

class ChannelCrossAttention(nn.Module):
    def __init__(
        self,
        n_bio_channels : int,
        embed_dim      : int,
        n_heads        : int   = 4,
        drop_prob      : float = 0.0,
    ):
        super().__init__()
        self.n_bio_channels = n_bio_channels
        self.embed_dim      = embed_dim

        self.norm_in  = nn.LayerNorm(embed_dim)
        self.attn     = nn.MultiheadAttention(embed_dim, n_heads, dropout=drop_prob)
        self.norm_out = nn.LayerNorm(embed_dim)
        self.gate_proj = nn.Linear(embed_dim, embed_dim)

        self.last_attn = None  # store attention weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, CD, H, W = x.shape
        C, D = self.n_bio_channels, self.embed_dim
        assert CD == C * D, f"Expected {C}*{D}={C*D} channels, got {CD}"

        x5 = x.view(B, C, D, H, W)

        # Pool spatially → one token per channel
        tokens = x5.mean(dim=(-2, -1))       # [B, C, D]
        tokens = self.norm_in(tokens)

        # Transpose for old PyTorch: [B, C, D] → [C, B, D]
        tokens_seq = tokens.transpose(0, 1)  # [seq_len=C, batch=B, embed=D]

        attn_out, attn_weights = self.attn(tokens_seq, tokens_seq, tokens_seq)
        # attn_weights shape: [seq_len, seq_len, batch] → permute to [B, C, C]
        #attn_weights = attn_weights.permute(2, 0, 1)
        self.last_attn = attn_weights.detach()

        attn_out = attn_out.transpose(0, 1)  # back to [B, C, D]
        attn_out = self.norm_out(attn_out)

        # Gate projection → broadcast onto spatial maps + residual
        gates = torch.sigmoid(self.gate_proj(attn_out))  # [B, C, D]
        gates = gates.unsqueeze(-1).unsqueeze(-1)        # [B, C, D, 1, 1]
        out = (x5 * gates + x5).view(B, CD, H, W)

        return out

# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

@MODELS.register_module()
class MCIANet(nn.Module):
    """
    Multi-Channel Image Analysis Network — Option B architecture.

    Spatial downsampling BEFORE cross-channel attention so that conv layers
    can spatially clean each channel (suppressing border contamination via
    SpatialFocusMap) before attention reasons about co-expression.

    Forward pass with default params (40 ch, 24x24 input):

      [B, 40, 24, 24]
            ↓  shared stem, stride-2  →  12x12  (per-channel, independent)
            ↓  SpatialFocusMap        →  12x12  (Gaussian-init, learns cell zones)
            ↓  cat + stage1           →  12x12  (joint cross-channel convs)
            ↓  down1                  →   6x6
            ↓  attn1                  →   6x6   (1st co-expression attention)
            ↓  stage2                 →   6x6
            ↓  down2                  →   3x3
            ↓  attn2                  →   3x3   (2nd co-expression attention)
            ↓  avgpool + flatten      →  [B, 40*stem_dim]   ← returned as (out,)

    Parameters
    ----------
    in_channels    : number of biological imaging channels
    stem_dim       : features per channel after stem
    n_heads        : attention heads
    stem_blocks    : ConvBlocks in per-channel stem
    stage1_blocks  : ConvBlocks at 12x12 (joint)
    stage2_blocks  : ConvBlocks at 6x6 (joint)
    expansion      : FFN expansion ratio in ConvBlocks
    drop_prob      : dropout probability
    use_gaussian   : multiply input by a fixed Gaussian distance map before
                     the stem as a geometric contamination prior
    input_size     : H (= W) of input patches
    sigma_fraction : Gaussian σ for SpatialFocusMap as fraction of post-stem size
    """

    def __init__(
        self,
        in_channels:int,
        input_size:int,
        stem_dim       : int   = 32,
        n_heads        : int   = 4,
        stem_blocks    : int   = 2,
        stage1_blocks  : int   = 2,
        stage2_blocks  : int   = 2,
        expansion      : int   = 2,
        drop_prob      : float = 0.05,
        # use_gaussian   : bool  = True,
        sigma_fraction : float = 0.35,
        spatial_init_mode: str = 'ones'
    ):
        super().__init__()

        assert input_size%2==0, f'input_size needs to be even currently {input_size}'
        self.in_channels  = in_channels
        self.stem_dim     = stem_dim
        # self.use_gaussian = use_gaussian
        self.input_size   = input_size

        fused_dim    = in_channels * stem_dim
        stem_spatial = input_size // 2          # spatial size after stride-2 stem

        self._gaussian_cache: dict = {}

        # ── Per-channel shared stem (stride-2, 24→12) ────────────────────────
        self.shared_stem = nn.Sequential(
            nn.Conv2d(1, stem_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, stem_dim), stem_dim),
            nn.GELU(),
            *[ConvBlock(stem_dim, expansion=expansion, drop_prob=drop_prob)
              for _ in range(stem_blocks)],
        )

        # ── SpatialFocusMap at 12x12, Gaussian-initialised ───────────────────
        self.spatial_focus = SpatialFocusMap(
            stem_dim       = stem_dim,
            spatial_size   = stem_spatial,
            sigma_fraction = sigma_fraction,
            init_mode      = spatial_init_mode
        )

        # ── Joint stage 1: 12x12 → 6x6 ──────────────────────────────────────
        self.stage1 = nn.Sequential(
            *[ConvBlock(fused_dim, expansion=expansion, drop_prob=drop_prob)
              for _ in range(stage1_blocks)]
        )
        self.down1 = make_downsample(fused_dim)

        # ── 1st attention at 6x6 ─────────────────────────────────────────────
        self.attn1 = ChannelCrossAttention(in_channels, stem_dim, n_heads, drop_prob)

        # ── Joint stage 2: 6x6 → 3x3 ────────────────────────────────────────
        self.stage2 = nn.Sequential(
            *[ConvBlock(fused_dim, expansion=expansion, drop_prob=drop_prob)
              for _ in range(stage2_blocks)]
        )
        self.down2 = make_downsample(fused_dim)

        # ── 2nd attention at 3x3 ─────────────────────────────────────────────
        # At 3x3: center pixel ≈ nucleus, ring of 8 ≈ cytoplasm/membrane.
        self.attn2 = ChannelCrossAttention(in_channels, stem_dim, n_heads, drop_prob)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight init (skips SpatialFocusMap which manages its own init)
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, SpatialFocusMap):
                continue   # has its own Gaussian init — do not overwrite
            if hasattr(m, "_skip_init"):  # optional extra flag
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                if m.weight is not None: nn.init.ones_(m.weight)
                if m.bias   is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Fixed Gaussian input prior (lazy, cached per spatial size)
    # ------------------------------------------------------------------

    # def _get_gaussian(self, H: int, W: int,
    #                   device: torch.device) -> torch.Tensor:
    #     key = (H, W)
    #     if key not in self._gaussian_cache:
    #         cy = (H - 1) / 2.0
    #         cx = (W - 1) / 2.0
    #         ys = torch.arange(H, dtype=torch.float32) - cy
    #         xs = torch.arange(W, dtype=torch.float32) - cx
    #         gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    #         g = torch.exp(-(gy**2 + gx**2) / (2 * (H / 4.0)**2))
    #         self._gaussian_cache[key] = g.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    #     return self._gaussian_cache[key].to(device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        x : [B, in_channels, H, W]
        returns (embedding,) — [B, in_channels * stem_dim]
        """
        B, C, H, W = x.shape
        assert C == self.in_channels, \
            f"Expected {self.in_channels} input channels, got {C}"

        # Fixed Gaussian input prior — soft geometric down-weighting of borders
        # if self.use_gaussian:
        #     x = x * self._get_gaussian(H, W, x.device)   # [B, C, H, W]

        # ── Per-channel stem + SpatialFocusMap ───────────────────────────────
        channel_feats = []
        spatial_masks = []

        for c in range(C):
            xc = x[:, c:c+1]                        # [B, 1, H, W]
            xc = self.shared_stem(xc)                # [B, stem_dim, H/2, W/2]
            xc, mask = self.spatial_focus(xc)              # [B, stem_dim, H/2, W/2]
            channel_feats.append(xc)
            spatial_masks.append(mask)                     # [B, 1, H/2, W/2]

        # Stored for inspection and optional compactness loss
        # Shape: [B, C, 1, H/2, W/2]
        self.last_spatial_masks = torch.stack(spatial_masks, dim=1)

        fused = torch.cat(channel_feats, dim=1)            # [B, C*stem_dim, 12, 12]

        # ── Stage 1 → down → attn1 ───────────────────────────────────────────
        fused = self.stage1(fused)
        fused = self.down1(fused)                          # [B, C*stem_dim, 6, 6]
        fused = self.attn1(fused)

        # ── Stage 2 → down → attn2 ───────────────────────────────────────────
        fused = self.stage2(fused)
        fused = self.down2(fused)                          # [B, C*stem_dim, 3, 3]
        fused = self.attn2(fused)

        # ── Global pool → flat embedding ─────────────────────────────────────
        out = self.avgpool(fused).flatten(1)               # [B, C*stem_dim]

        return (out,)


# ---------------------------------------------------------------------------
# Compactness regularisation (optional, use in your training loop)
# ---------------------------------------------------------------------------

def spatial_compactness_loss(model: MCIANet) -> torch.Tensor:
    """
    Penalises large activated regions in the SpatialFocusMap masks.

    Usage in training loop:
        loss = task_loss + lambda_c * spatial_compactness_loss(model)
        # anneal lambda_c → 0 over training epochs
    """
    return model.last_spatial_masks.mean()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

# if __name__ == "__main__":

#     model = MCIANet(
#         in_channels    = 40,
#         stem_dim       = 32,
#         n_heads        = 4,
#         stem_blocks    = 2,
#         stage1_blocks  = 2,
#         stage2_blocks  = 2,
#         expansion      = 2,
#         drop_prob      = 0.05,
#         # use_gaussian   = True,
#         input_size     = 24,
#         sigma_fraction = 0.35,
#     )

#     x = torch.randn(4, 40, 24, 24)
#     (out,) = model(x)

#     total     = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     print(f"Input            : {x.shape}")
#     print(f"Output embedding : {out.shape}")
#     print(f"Params           : {total:,} total | {trainable:,} trainable")
#     print(f"Spatial masks    : {model.last_spatial_masks.shape}  [B, C, 1, H/2, W/2]")

#     # Verify Gaussian init: center pixel should be >> corner pixel at step 0
#     model.eval()
#     with torch.no_grad():
#         xc       = torch.ones(1, 1, 24, 24)
#         xc       = model.shared_stem(xc)
#         _, m     = model.spatial_focus(xc)
#         print(f"\nSpatialFocusMap Gaussian init check (uniform input, step 0):")
#         print(f"  Center mask value : {m[0, 0, 6, 6].item():.4f}  ← should be high")
#         print(f"  Corner mask value : {m[0, 0, 0, 0].item():.4f}  ← should be low")

#     print("\nSpatial flow summary:")
#     print("  [B,40,24,24] → stem(stride-2) → [B,40,32,12,12]")
#     print("               → SpatialFocusMap → [B,40,32,12,12]")
#     print("               → cat+stage1      → [B,1280,12,12]")
#     print("               → down1           → [B,1280, 6, 6]")
#     print("               → attn1           → [B,1280, 6, 6]")
#     print("               → stage2+down2    → [B,1280, 3, 3]")
#     print("               → attn2+avgpool   → [B,1280]")