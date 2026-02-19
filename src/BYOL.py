import copy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.algorithms.base import BaseModel


class _MLP(nn.Module):
    """Two-layer MLP with BN — used as BYOL predictor."""

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid_dim, bias=False),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@MODELS.register_module()
class MVBYOL(BaseModel):
    """
    BYOL: Bootstrap Your Own Latent.
    https://arxiv.org/abs/2006.07733

    Online network:  backbone → projector (neck) → predictor
    Target network:  EMA(backbone) → EMA(projector)   [no gradient]

    Loss: symmetric negative cosine similarity between online predictions
          and stop-gradient target projections. No negative pairs needed.
    """

    def __init__(
        self,
        momentum: float = 0.996,
        pred_in_channels: int = 256,
        pred_hid_channels: int = 512,
        pred_out_channels: int = 256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.momentum = momentum
        self.device = None

        # Predictor sits on top of the online projector output
        self.predictor = _MLP(pred_in_channels, pred_hid_channels, pred_out_channels)

        # Target network: EMA copy of online backbone + neck, no gradients
        self.target_backbone = copy.deepcopy(self.backbone)
        self.target_neck = copy.deepcopy(self.neck)
        for p in self.target_backbone.parameters():
            p.requires_grad_(False)
        for p in self.target_neck.parameters():
            p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Feature extraction (inference only — uses online backbone)
    # ------------------------------------------------------------------

    def extract_feat(self, inputs: List[torch.Tensor], data_samples=None, **kwargs):
        return self.backbone(inputs[0], **kwargs)

    # ------------------------------------------------------------------
    # Device tracking
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return result

    # ------------------------------------------------------------------
    # EMA update of target network (called after each loss step)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _update_target(self):
        tau = self.momentum
        for p_o, p_t in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            p_t.data.mul_(tau).add_((1 - tau) * p_o.data)
        for p_o, p_t in zip(self.neck.parameters(), self.target_neck.parameters()):
            p_t.data.mul_(tau).add_((1 - tau) * p_o.data)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _byol_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity (BYOL eq. 2), scaled to [0, 2]."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def loss(
        self,
        inputs: List[torch.Tensor],
        data_samples: List[SelfSupDataSample],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        assert len(inputs) == 2, f"BYOL expects exactly 2 views, got {len(inputs)}."

        x1 = inputs[0].to(self.device)
        x2 = inputs[1].to(self.device)

        # Online: backbone → projector → predictor
        z1_online = self.neck(self.backbone(x1))[0]   # [B, proj_dim]
        z2_online = self.neck(self.backbone(x2))[0]
        p1 = self.predictor(z1_online)                 # [B, pred_dim]
        p2 = self.predictor(z2_online)

        # Target: no gradient
        with torch.no_grad():
            z1_target = self.target_neck(self.target_backbone(x1))[0]
            z2_target = self.target_neck(self.target_backbone(x2))[0]

        # Symmetric loss
        loss = (
            self._byol_loss(p1, z2_target.detach())
            + self._byol_loss(p2, z1_target.detach())
        )

        # EMA update after gradient will be applied
        self._update_target()

        return dict(loss=loss)
