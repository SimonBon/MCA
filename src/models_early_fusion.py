import torch
import torch.nn as nn
from mmengine.registry import MODELS


class EarlyConvBlock(nn.Module):
    """
    ConvNeXt-style residual block using standard (non-depthwise) convolutions.
    All channels are fully mixed at every operation — no grouping.
    Mirrors the ConvBlock in models.py but with groups=1 throughout.
    """
    def __init__(self, in_channels: int, block_width: int = 4, drop_prob: float = 0.0):
        super().__init__()
        hidden = in_channels * block_width
        self.conv     = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        self.norm     = nn.BatchNorm2d(in_channels)
        self.pw_expand  = nn.Conv2d(in_channels, hidden, 1, bias=False)
        self.act      = nn.GELU()
        self.pw_project = nn.Conv2d(hidden, in_channels, 1, bias=False)
        self.drop     = nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.pw_expand(x)
        x = self.act(x)
        x = self.pw_project(x)
        x = self.drop(x)
        return x + identity


@MODELS.register_module()
class EarlyFusionModel(nn.Module):
    """
    Early Fusion — direct ablation of WideModel (CIM).

    Identical architecture to WideModel in every respect (same stem_width,
    block_width, layer_config, late_fusion option) EXCEPT that all Conv2d
    operations use groups=1 (standard convolutions) instead of
    groups=in_channels (depthwise).

    All input channels are mixed together in the very first stem layer.
    This is the null hypothesis for the channel-separability experiments:
    if EarlyFusionModel >= WideModel, channel separability provides no benefit.

    Output shape: [B, in_channels * stem_width, 1, 1]
    """

    def __init__(
        self,
        in_channels: int,
        stem_width: int = 16,
        block_width: int = 4,
        layer_config = [1, 1],
        drop_prob: float = 0.05,
        late_fusion: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_out_channels = in_channels * stem_width

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.stem_out_channels, kernel_size=3,
                      padding=1, groups=1, bias=False),   # <-- groups=1: early fusion
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True),
        )

        layers = []
        for n_blocks in layer_config:
            for _ in range(n_blocks):
                layers.append(EarlyConvBlock(self.stem_out_channels, block_width, drop_prob))
            layers.append(nn.AvgPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

        self.late_fusion = (
            nn.Conv2d(self.stem_out_channels, self.stem_out_channels, 1)
            if late_fusion else None
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        if self.late_fusion is not None:
            x = self.late_fusion(x)
        return (x,)


@MODELS.register_module()
class ProjectionFusionModel(nn.Module):
    """
    Projection Fusion — channels mixed at pixel level before any spatial processing.

    A 1x1 convolution (no spatial context) collapses all input channels into the
    joint feature space first. Only after this pixel-level fusion does any spatial
    processing occur. This is the most aggressive early fusion strategy: the model
    has no access to per-channel spatial structure at all.

    Contrast with EarlyFusionModel which at least applies a 3x3 conv (giving
    spatial context) to all channels simultaneously. Here the first operation is
    explicitly non-spatial (kernel_size=1).

    Output shape: [B, in_channels * stem_width, 1, 1]
    """

    def __init__(
        self,
        in_channels: int,
        stem_width: int = 16,
        block_width: int = 4,
        layer_config = [1, 1],
        drop_prob: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_out_channels = in_channels * stem_width

        # 1x1 conv: channel mixing at pixel level, no spatial context
        self.pixel_fusion = nn.Sequential(
            nn.Conv2d(in_channels, self.stem_out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True),
        )

        layers = []
        for n_blocks in layer_config:
            for _ in range(n_blocks):
                layers.append(EarlyConvBlock(self.stem_out_channels, block_width, drop_prob))
            layers.append(nn.AvgPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.pixel_fusion(x)
        x = self.layers(x)
        x = self.avgpool(x)
        return (x,)


@MODELS.register_module()
class MidFusionModel(nn.Module):
    """
    Mid Fusion — channel-separable stem, then early fusion in all subsequent layers.

    One depthwise convolutional layer extracts per-channel spatial features
    independently (identical to WideModel's stem), then all subsequent ConvBlocks
    use standard convolutions that mix channels freely.

    This tests whether a single layer of per-channel spatial feature extraction
    is sufficient, or whether maintaining channel separability throughout the
    entire network is necessary. It isolates the contribution of the stem vs.
    the processing layers to the overall channel-separability advantage.

    Output shape: [B, in_channels * stem_width, 1, 1]
    """

    def __init__(
        self,
        in_channels: int,
        stem_width: int = 16,
        block_width: int = 4,
        layer_config = [1, 1],
        drop_prob: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.stem_out_channels = in_channels * stem_width

        # Depthwise stem: channel-separable (same as WideModel)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, self.stem_out_channels, kernel_size=3,
                      padding=1, groups=in_channels, bias=False),  # depthwise
            nn.BatchNorm2d(self.stem_out_channels),
            nn.ReLU(inplace=True),
        )

        # All subsequent layers: standard convolutions (early fusion from here on)
        layers = []
        for n_blocks in layer_config:
            for _ in range(n_blocks):
                layers.append(EarlyConvBlock(self.stem_out_channels, block_width, drop_prob))
            layers.append(nn.AvgPool2d(2, 2))
        self.layers = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.stem(x)
        x = self.layers(x)
        x = self.avgpool(x)
        return (x,)


@MODELS.register_module()
class ResNetBaseline(nn.Module):
    """
    Lightweight ResNet-style baseline for small patches.

    Standard early-fusion CNN in the style of ResNet, scaled down for
    24x24 / 32x32 input patches. No depthwise convolutions — all operations
    are fully-mixed standard convolutions from the first layer.

    Included as an external reference point beyond the WideModel family:
    a well-understood architecture that makes no assumptions about channel
    structure whatsoever.

    Note: unlike the WideModel family, the output dimension is base_width*4
    (not in_channels * stem_width), so the neck in_channels must be set
    accordingly in the experiment config.

    Output shape: [B, base_width * 4, 1, 1]
    """

    def __init__(
        self,
        in_channels: int,
        base_width: int = 64,
        drop_prob: float = 0.05,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = base_width * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(base_width,     base_width * 2, stride=2, drop_prob=drop_prob)
        self.layer2 = self._make_layer(base_width * 2, base_width * 4, stride=2, drop_prob=drop_prob)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _make_layer(self, in_ch: int, out_ch: int, stride: int, drop_prob: float):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(drop_prob) if drop_prob > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return (x,)
