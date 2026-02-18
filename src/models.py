import torch
from torch import nn
from mmengine.registry import MODELS

@MODELS.register_module()
class WideModel(nn.Module):
    
    def __init__(self, in_channels, stem_width=16, block_width=4, layer_config=[2, 2], drop_prob=0.05, late_fusion=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.stem_width = stem_width
        self.block_width = block_width
        self.stem_out_channels = in_channels * self.stem_width
        self.layer_config = layer_config
        self.drop_prob=drop_prob
        
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                self.stem_out_channels, 
                kernel_size=3, 
                padding=1, 
                groups=in_channels, 
                bias=False
            ),
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layers = []
        for config in self.layer_config:
            self.layers.append(self._make_layer(config))
            self.layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        
        if late_fusion:
            
            self.late_fusion = nn.Conv2d(
                self.stem_out_channels,
                self.stem_out_channels,
                kernel_size=1)
            
        else:
            self.late_fusion = None
            
        self.layers = nn.Sequential(*self.layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, n_blocks):
        layers = []
        
        for _ in range(n_blocks):
            layers.append(ConvBlock(
                in_channels=self.stem_out_channels, 
                groups=self.in_channels,
                block_width=self.block_width,
                drop_prob=self.drop_prob
            ))
            
        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        
        try:
        
            x = self.stem(x)
            x = self.layers(x)
            x = self.avgpool(x)
            
        except Exception as e:
            
            if x.shape[-1]//2**len(self.layer_config) < 1:
                print('!!! ERROR MOST LIKELY DUE TO TOO MANY DOWNSAMPLINGS REDUCE len(layer_configs) !!!')
                
            raise e

        if self.late_fusion is not None:
            x = self.late_fusion(x)

        
        return (x,)


@MODELS.register_module()
class SharedStemModel(nn.Module):
    
    def __init__(self, in_channels, stem_width=16, block_width=4, n_layers=2, late_fusion=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.stem_width = stem_width
        self.block_width = block_width
        self.stem_out_channels = in_channels * stem_width  # Total after concatenation
        
        # Shared stem: 1 -> stem_width for each channel
        self.shared_stem = nn.Sequential(
            nn.Conv2d(1, stem_width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True)
        )
        
        # Shared layers: process each channel's stem output
        layers = []
        for _ in range(n_layers):
            layers.append(ConvBlock(
                in_channels=stem_width,  # Input is stem_width per channel
                groups=1,  # Process each channel's features together within the channel
                block_width=block_width
            ))
            
        self.shared_layers = nn.Sequential(*layers)
        
        if late_fusion:
            self.late_fusion = nn.Conv2d(
                self.stem_out_channels,
                self.stem_out_channels,
                kernel_size=1
            )
        else:
            self.late_fusion = None
            
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, *args, **kwargs):
        # x: [B, in_channels, H, W]
        B, C, H, W = x.shape
        
        # Apply shared stem and layers to each channel independently
        feats = []
        for c in range(C):
            # Extract single channel: [B, 1, H, W]
            x_c = x[:, c:c+1, :, :]
            
            # Shared stem: [B, 1, H, W] -> [B, stem_width, H, W]
            x_c = self.shared_stem(x_c)
            
            # Shared layers: [B, stem_width, H, W] -> [B, stem_width, H, W]
            x_c = self.shared_layers(x_c)
            
            feats.append(x_c)
        
        # Stack all channel features: [B, stem_width*C, H, W]
        x = torch.cat(feats, dim=1)
        
        # Late fusion across channels
        if self.late_fusion is not None:
            x = self.late_fusion(x)
        
        # Global pooling: [B, stem_width*C, H, W] -> [B, stem_width*C, 1, 1]
        x = self.avgpool(x)
        return (x,)
    
from typing import List, Optional
import torch
import torch.nn as nn


@MODELS.register_module()
class WideModelAttention(nn.Module):
    """
    Per-channel wide expansion backbone with self-attention across channel tokens.

    Each input channel is expanded to `stem_width` sub-channels via a depthwise
    conv stem. The sub-channels act as per-channel tokens that are mixed via
    multi-head self-attention, then used to gate the spatial feature map before
    global pooling.

    Args:
        in_channels:  Number of input channels (e.g. 1 for grayscale, 3 for RGB).
        stem_width:   Expansion factor per input channel in the stem.
        block_width:  FFN expansion ratio inside each ConvBlock.
        layer_config: Number of ConvBlocks per stage. A pooling layer is inserted
                      between stages. E.g. [2, 2] → 2 blocks, pool, 2 blocks, pool.
        drop_prob:    Dropout probability used in ConvBlocks and attention.
        n_heads:      Number of attention heads (must divide stem_width evenly).
    """

    def __init__(
        self,
        in_channels: int,
        stem_width: int = 16,
        block_width: int = 4,
        layer_config: Optional[List[int]] = None,
        drop_prob: float = 0.05,
        n_heads: int = 4,
    ):
        super().__init__()

        if layer_config is None:
            layer_config = [2, 2]

        assert stem_width % n_heads == 0, (
            f"stem_width ({stem_width}) must be divisible by n_heads ({n_heads})"
        )

        self.in_channels = in_channels
        self.stem_width = stem_width
        self.stem_out_channels = in_channels * stem_width  # C * D

        # Depthwise stem: each input channel independently expanded to stem_width channels.
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                self.stem_out_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True),
        )

        # Convolutional stages with downsampling between them.
        layers = []
        for n_blocks in layer_config:
            for _ in range(n_blocks):
                layers.append(
                    ConvBlock(
                        in_channels=self.stem_out_channels,
                        groups=in_channels,
                        block_width=block_width,
                        drop_prob=drop_prob,
                    )
                )
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.layers = nn.Sequential(*layers)

        # Channel-wise self-attention over [C, B, D] token sequences.
        # embed_dim = stem_width = D (sub-channels per input channel).
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=stem_width,
            num_heads=n_heads,
            dropout=drop_prob,
            # batch_first=False (default) → expects [S, B, D]
        )
        self.attn_norm = nn.LayerNorm(stem_width)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        """
        Args:
            x: Input tensor of shape [B, in_channels, H, W].
        Returns:
            Tuple containing a single tensor of shape [B, in_channels * stem_width, 1, 1].
        """
        x = self.stem(x)       # [B, C*D, H, W]
        x = self.layers(x)     # [B, C*D, H', W']

        B, CD, H, W = x.shape
        C = self.in_channels
        D = self.stem_width

        assert CD == C * D, (
            f"Channel dim mismatch: expected {C * D}, got {CD}. "
            "Check that stem_width hasn't changed between __init__ and forward."
        )

        # Reshape to [B, C, D, H', W'] and compute per-channel spatial tokens.
        tokens = x.view(B, C, D, H, W).mean(dim=(-2, -1))   # [B, C, D]
        tokens = self.attn_norm(tokens)                       # pre-norm

        # MHA expects [S, B, D] (S = C here, sequence over channels).
        tokens_t = tokens.transpose(0, 1)                     # [C, B, D]
        attn_out, _ = self.channel_attn(tokens_t, tokens_t, tokens_t)
        tokens = tokens + attn_out.transpose(0, 1)            # [B, C, D], residual

        # Gate the *spatial* feature map with attention-derived weights,
        # then pool globally. This is the key fix vs. the original: we gate
        # x (still spatial) before pooling, rather than gating an already-pooled tensor.
        gates = torch.sigmoid(tokens)                             # [B, C, D]
        gates_spatial = gates.view(B, CD, 1, 1)                  # broadcast over H', W'
        x_gated = x * gates_spatial                              # [B, C*D, H', W']

        out = x_gated.mean(dim=(-2, -1)).view(B, CD, 1, 1)       # global avg pool

        return (out,)


class ConvBlock(nn.Module):
    """
    Depthwise-separable ConvNeXt-style block with an FFN and skip connection.

    Layout (pre-activation residual):
        identity = x
        x → depthwise conv → BN → pointwise expand → GELU → pointwise project → Dropout
        out = x + identity

    Args:
        in_channels:  Total number of channels (C * stem_width).
        groups:       Number of depthwise groups (should equal the original in_channels,
                      so that each input channel's sub-channels stay independent).
        block_width:  FFN hidden dimension multiplier.
        drop_prob:    Probability for DropPath / channel dropout.
    """

    def __init__(
        self,
        in_channels: int,
        groups: int,
        block_width: int,
        drop_prob: float = 0.0,
    ):
        super().__init__()

        hidden = in_channels * block_width

        self.dw_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1,
            groups=groups, bias=False,
        )
        self.norm = nn.BatchNorm2d(in_channels)

        # Pointwise FFN (also grouped to keep channels independent across input channels).
        self.pw_expand = nn.Conv2d(in_channels, hidden, kernel_size=1, groups=groups, bias=False)
        self.act = nn.GELU()
        self.pw_project = nn.Conv2d(hidden, in_channels, kernel_size=1, groups=groups, bias=False)

        # Dropout2d drops entire feature maps (channels), acting as a structured regulariser.
        self.drop = nn.Dropout2d(drop_prob) if drop_prob > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        x = self.drop(x)

        return x + identity