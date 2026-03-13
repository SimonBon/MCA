# Copyright (c) OpenMMLab.
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from mmengine.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer
from mmselfsup.models.algorithms.base import BaseModel


class SampleCentroidBank:
    """
    EMA memory bank of per-sample embedding centroids.

    Stores detached EMA centroids as stable reference points. The alignment
    loss pulls the CURRENT batch's live per-sample means toward the global
    centroid mean, so gradients flow through the model.

    Args:
        dim:      embedding dimensionality
        momentum: EMA momentum for centroid update (higher = slower update)
    """

    def __init__(self, dim: int, momentum: float = 0.99):
        self.dim = dim
        self.momentum = momentum
        self._centroids: Dict[str, torch.Tensor] = {}  # detached EMA references

    @torch.no_grad()
    def update(self, z: torch.Tensor, sample_ids: List[str]):
        """Update EMA centroids for samples present in the current batch."""
        for sid in set(sample_ids):
            mask = torch.tensor([s == sid for s in sample_ids], device=z.device)
            z_mean = z[mask].mean(dim=0).detach()
            if sid not in self._centroids:
                self._centroids[sid] = z_mean
            else:
                self._centroids[sid] = (
                    self.momentum * self._centroids[sid]
                    + (1 - self.momentum) * z_mean
                )

    def alignment_loss(self, z: torch.Tensor, sample_ids: List[str]) -> torch.Tensor:
        """
        For each sample in the batch, pull its current mean embedding toward
        the global centroid mean. Gradients flow through z (live embeddings).

        Returns 0 if fewer than 2 distinct samples are in the bank.
        """
        if len(self._centroids) < 2:
            return z.sum() * 0.0  # zero with gradient graph attached

        # Global mean of all EMA centroids (detached reference)
        global_mean = torch.stack(
            list(self._centroids.values()), dim=0
        ).mean(dim=0)  # [D] — no gradient

        loss = z.sum() * 0.0  # start at zero, keeps device/dtype
        unique_ids = set(sample_ids)
        for sid in unique_ids:
            mask = torch.tensor([s == sid for s in sample_ids], device=z.device)
            z_s = z[mask].mean(dim=0)                 # [D] — has gradient
            loss = loss + F.mse_loss(z_s, global_mean) # pull toward global mean

        return loss / len(unique_ids)


@MODELS.register_module()
class MVVICReg(BaseModel):
    """
    VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.
    https://arxiv.org/abs/2105.04906

    Three loss terms (no contrastive negatives, no L2 normalisation):
      - Invariance : MSE between projections of the views   (pull together)
      - Variance   : hinge loss keeping per-dim std > gamma (prevent collapse)
      - Covariance : penalises off-diagonal covariance      (decorrelate dims)

    Optionally adds a sample-alignment term via a centroid memory bank:
      - Alignment  : variance of per-sample EMA centroids   (reduce batch effects)

    Supports N >= 2 views.
    """

    def __init__(
        self,
        sim_coeff: float = 25.0,
        std_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        gamma: float = 1.0,
        align_coeff: float = 0.0,
        centroid_momentum: float = 0.99,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma
        self.align_coeff = align_coeff
        self.device = None

        # Memory bank — only created when align_coeff > 0, dim set lazily
        self._centroid_bank: Optional[SampleCentroidBank] = None
        self._centroid_momentum = centroid_momentum

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
        std = torch.sqrt(z.var(dim=0) + eps)          # [D]
        loss = F.relu(gamma - std).mean()
        return loss

    @staticmethod
    def _covariance_loss(z: torch.Tensor) -> torch.Tensor:
        """Off-diagonal covariance penalty."""
        N, D = z.shape
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (N - 1)                      # [D, D]
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

        # ── Sample alignment loss (optional) ──────────────────────────
        loss_align = torch.tensor(0.0, device=zs[0].device)
        if self.align_coeff > 0.0 and data_samples is not None:
            # Extract sample IDs — C_MultiView stores per-view lists, take first view
            # data_samples['sample_id'] shape: [n_views, batch_size]
            raw = data_samples['sample_id']
            sample_ids = list(raw[0]) if isinstance(raw[0], (list, tuple)) else list(raw)

            # Lazy init of centroid bank
            if self._centroid_bank is None:
                self._centroid_bank = SampleCentroidBank(
                    dim=zs[0].shape[-1],
                    momentum=self._centroid_momentum,
                )

            # Update EMA bank with detached mean across views
            z_mean_views = torch.stack(zs, dim=0).mean(dim=0)  # [B, D]
            self._centroid_bank.update(z_mean_views, sample_ids)

            # Alignment loss: pull live embeddings toward global centroid mean
            loss_align = self._centroid_bank.alignment_loss(z_mean_views, sample_ids)
            total = total + self.align_coeff * loss_align

        return dict(
            loss=total,
            loss_inv=loss_inv,
            loss_var=loss_var,
            loss_cov=loss_cov,
            loss_align=loss_align,
        )
