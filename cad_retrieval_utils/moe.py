import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEImgHead(nn.Module):
    """Mixture of Experts head для проекции изображений"""

    def __init__(self, in_dim: int, out_dim: int, n_experts: int = 8) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(in_dim, out_dim) for _ in range(n_experts)
        ])
        self.gate = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, n_experts))

    def forward(self, feats: torch.Tensor, normalize: bool) -> torch.Tensor:
        logits = self.gate(feats)
        w = torch.softmax(logits, dim=-1)
        outs = torch.stack([e(feats) for e in self.experts], dim=1)
        out = (w.unsqueeze(-1) * outs).sum(1)
        return F.normalize(out, dim=-1) if normalize else out
