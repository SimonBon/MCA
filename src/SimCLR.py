# Copyright (c) OpenMMLab.
from typing import Dict, List
import torch
import torch.nn.functional as F
import einops

from mmengine.registry import MODELS
from mmselfsup.structures import SelfSupDataSample
from mmselfsup.models.utils import GatherLayer
from mmselfsup.models.algorithms.base import BaseModel


@MODELS.register_module()
class MVSimCLR(BaseModel):
    """
    Clean SimCLR implementation using backbone → neck → head.
    The head is mmselfsup's ContrastiveHead, which expects:
        - pos_similarity:  (N, 1)
        - neg_similarity:  (N, K)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = None

    # Feature extraction (for inference only)
    def extract_feat(self, inputs: List[torch.Tensor],
                     data_samples=None,
                     **kwargs):
        return self.backbone(inputs[0], **kwargs)

    # Normalize + gather across GPUs
    def _gather(self, z: torch.Tensor) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        z = torch.cat(GatherLayer.apply(z), dim=0)
        return z

    # Device tracking
    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        # backbone.backbone is the actual ResNet inside LRPModel
        self.device = next(self.parameters()).device
        return result
    
    def loss(
        self,
        inputs: List[torch.Tensor],
        data_samples: List[SelfSupDataSample],
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        V = len(inputs)                       # number of views
        B = inputs[0].shape[0]                # batch size                               # alias
        
        proj_list = []
        for v in range(V):
            feats = self.backbone(inputs[v].to(self.device), mode="training")
            z = self.neck(feats)[0]           # (B, D)
            proj_list.append(z)

        z = torch.stack(proj_list, dim=0)

        z = einops.rearrange(z, "v b d -> (v b) d")

        z = self._gather(z)
        N = z.size(0)                         

        sim = torch.matmul(z, z.T)          
        identity = torch.eye(N, device=z.device).bool()
        sim = sim.masked_fill(identity, 0)

        pos_mask = torch.zeros_like(sim, dtype=torch.bool)

        idx = torch.arange(N, device=z.device).view(V, B)  # (V,B)

        for b in range(B):
            views = idx[:, b]          
            for i in range(V):
                for j in range(i+1, V):
                    a = views[i]
                    b = views[j]
                    pos_mask[a, b] = True
                    pos_mask[b, a] = True

        pos_sim = sim[pos_mask].view(N, -1).mean(dim=1, keepdim=True)

        neg_mask = (~identity) & (~pos_mask)
        neg_sim = sim[neg_mask].view(N, -1)

        loss = self.head(pos_sim, neg_sim)

        return dict(loss=loss)