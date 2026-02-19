from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer

from .VICReg import MVVICReg


@MODELS.register_module()
class MaskedChannelVICReg(MVVICReg):
    """
    VICReg + Masked Channel Modeling (MCM) auxiliary loss.

    On top of the standard VICReg objective, one view has a random subset of
    its marker channels replaced by a learned mask token. The backbone must
    then reconstruct the mean intensity of each masked channel from the
    remaining visible channels, forcing it to learn marker co-expression
    patterns.

    The reconstruction head is a single shared Linear(stem_width → 1) applied
    independently to each channel's D-dimensional feature slice in the backbone
    output. This works naturally with WideModel because the depthwise backbone
    keeps each channel's features in a contiguous block [c*D : (c+1)*D].

    Args:
        mask_ratio:  Fraction of channels to mask per sample (default 0.3).
        mcm_coeff:   Weight of the MCM loss relative to VICReg (default 0.1).
        *args/**kwargs: Passed through to MVVICReg (sim_coeff, std_coeff, etc.)
    """

    def __init__(
        self,
        mask_ratio: float = 0.3,
        mcm_coeff: float = 0.1,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.mask_ratio = mask_ratio
        self.mcm_coeff = mcm_coeff

        D = self.backbone.stem_width
        # Shared linear: D features → predicted mean intensity (one per channel)
        self.recon_head = nn.Linear(D, 1)
        # Learned scalar substituted for masked channels (instead of hard zero)
        self.mask_token = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # MCM loss
    # ------------------------------------------------------------------

    def _mcm_loss(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        D = self.backbone.stem_width

        # ── Random per-sample channel mask ────────────────────────────
        # mask[b, c] = 1 → keep, 0 → masked
        noise = torch.rand(B, C, device=x.device)
        mask = (noise >= self.mask_ratio).float()   # [B, C]

        # ── Targets: mean marker intensity before masking ─────────────
        targets = x.mean(dim=(-2, -1))              # [B, C]

        # ── Apply mask token to hidden channels ───────────────────────
        # Masked channels get the learned scalar; visible channels unchanged.
        x_masked = (
            x * mask[:, :, None, None]
            + (1.0 - mask[:, :, None, None]) * self.mask_token
        )

        # ── Backbone forward on masked input ──────────────────────────
        (features,) = self.backbone(x_masked)       # [B, C*D, 1, 1]
        features = features.view(B, C, D)           # [B, C, D]

        # ── Predict mean intensity per channel ────────────────────────
        preds = self.recon_head(features).squeeze(-1)   # [B, C]

        # ── MSE only on masked channels ───────────────────────────────
        mask_bool = (mask == 0)
        if not mask_bool.any():
            return x.new_zeros(1).squeeze()

        return F.mse_loss(preds[mask_bool], targets[mask_bool])

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def loss(
        self,
        inputs: List[torch.Tensor],
        data_samples: List[SelfSupDataSample],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        # Standard VICReg loss on two clean augmented views
        losses = super().loss(inputs, data_samples, **kwargs)

        # MCM auxiliary loss on view 1 with random channel masking
        loss_mcm = self._mcm_loss(inputs[0].to(self.device))

        losses['loss_mcm'] = loss_mcm
        losses['loss'] = losses['loss'] + self.mcm_coeff * loss_mcm

        return losses
