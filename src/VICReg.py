# Copyright (c) OpenMMLab.
from typing import Dict, List

import torch
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer
from mmselfsup.models.algorithms.base import BaseModel


@MODELS.register_module()
class MVVICReg(BaseModel):
    """
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.
    https://arxiv.org/abs/2105.04906

    Three loss terms (no contrastive negatives, no L2 normalisation):
      - Invariance : MSE between projections of the two views  (pull together)
      - Variance   : hinge loss keeping per-dim std > gamma    (prevent collapse)
      - Covariance : penalises off-diagonal covariance         (decorrelate dims)

    No head required — loss is computed directly in the model.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.device = None

    # ------------------------------------------------------------------
    # Feature extraction (inference only)
    # ------------------------------------------------------------------

    def extract_feat(
        self,
        inputs: List[torch.Tensor],
        data_samples=None,
        **kwargs,
    ):
        return self.backbone(inputs[0], **kwargs)

    # ------------------------------------------------------------------
    # Device tracking
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        self.device = next(self.parameters()).device
        return result

    # ------------------------------------------------------------------
    # VICReg loss components
    # ------------------------------------------------------------------

    @staticmethod
    def _variance_loss(z: torch.Tensor, gamma: float, eps: float = 1e-4) -> torch.Tensor:
        """Hinge loss keeping per-dim std > gamma."""
        # z: [N, D]
        std = torch.sqrt(z.var(dim=0) + eps)          # [D]
        loss = F.relu(gamma - std).mean()
        return loss

    @staticmethod
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        """Off-diagonal covariance penalty."""
        N, D = z.shape
        z = z - z.mean(dim=0)                          # centre
        cov = (z.T @ z) / (N - 1)                      # [D, D]
        # zero the diagonal, penalise everything else
        off_diag = cov.pow(2)
        off_diag.fill_diagonal_(0.0)
        loss = off_diag.sum() / D
        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def loss(
        self,
        inputs: List[torch.Tensor],
        data_samples: List[SelfSupDataSample],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:

        assert len(inputs) == 2, (
            f"VICReg expects exactly 2 views, got {len(inputs)}. "
            "Check your augmentation pipeline."
        )

        # Backbone + neck for both views  →  z1, z2: [B, D]
        z1 = self.neck(self.backbone(inputs[0].to(self.device)))[0]
        z2 = self.neck(self.backbone(inputs[1].to(self.device)))[0]

        # Gather across GPUs (no-op in single-GPU mode)
        z1 = torch.cat(GatherLayer.apply(z1), dim=0)
        z2 = torch.cat(GatherLayer.apply(z2), dim=0)

        # ── Invariance loss ───────────────────────────────────────────
        loss_inv = F.mse_loss(z1, z2)

        # ── Variance loss ─────────────────────────────────────────────
        loss_var = (
            self._variance_loss(z1, self.gamma)
            + self._variance_loss(z2, self.gamma)
        )

        # ── Covariance loss ───────────────────────────────────────────
        loss_cov = (
            self._covariance_loss(z1)
            + self._covariance_loss(z2)
        )

        total = (
            self.sim_coeff * loss_inv
            + self.std_coeff * loss_var
            + self.cov_coeff * loss_cov
        )

        return dict(
            loss=total,
            loss_inv=loss_inv,
            loss_var=loss_var,
            loss_cov=loss_cov,
        )
