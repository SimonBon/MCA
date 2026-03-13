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

        assert len(inputs) >= 2, (
            f"VICReg expects at least 2 views, got {len(inputs)}. "
            "Check your augmentation pipeline."
        )

        # Backbone + neck for all views  →  list of [B, D]
        zs = [
            torch.cat(GatherLayer.apply(
                self.neck(self.backbone(x.to(self.device)))[0]
            ), dim=0)
            for x in inputs
        ]

        # ── Invariance loss: mean MSE over all unique pairs ────────────
        n_views = len(zs)
        pairs = [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]
        loss_inv = sum(F.mse_loss(zs[i], zs[j]) for i, j in pairs) / len(pairs)

        # ── Variance loss: mean over all views ────────────────────────
        loss_var = sum(self._variance_loss(z, self.gamma) for z in zs) / n_views

        # ── Covariance loss: mean over all views ──────────────────────
        loss_cov = sum(self._covariance_loss(z) for z in zs) / n_views

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
