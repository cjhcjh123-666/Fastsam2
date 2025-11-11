import torch
from typing import Optional

class MemoryAdapter:
    """
    Minimal streaming memory adapter for video/VOS.
    Stores a lightweight summary of recent object kernels per batch index.
    """
    def __init__(self, max_len: int = 5, topk: int = 0, use_attention: bool = True):
        self.max_len = max_len
        self.topk = topk
        self.use_attention = use_attention
        self.buf = []  # list of tensors [B, C] per time (summary per time step)

    @torch.no_grad()
    def reset(self):
        self.buf = []

    @torch.no_grad()
    def update(self, object_kernels: torch.Tensor, iou_preds: torch.Tensor = None):
        """
        object_kernels: [B, Q, C]
        iou_preds: [B, Q, 1] optional for Top-K selection
        Store per-batch summary as memory (mean over selected queries).
        """
        if object_kernels is None:
            return
        if self.topk and self.topk > 0 and iou_preds is not None:
            # select top-k queries by IoU confidence
            scores = torch.sigmoid(iou_preds.squeeze(-1))  # [B, Q]
            k = min(self.topk, scores.shape[1])
            topk_idx = torch.topk(scores, k=k, dim=1).indices  # [B, k]
            B = object_kernels.shape[0]
            gather = []
            for b in range(B):
                gather.append(object_kernels[b, topk_idx[b]])  # [k, C]
            gathered = torch.stack(gather, 0)  # [B, k, C]
            kernel_mean = gathered.mean(dim=1)  # [B, C]
        else:
            kernel_mean = object_kernels.mean(dim=1)  # [B, C]
        self.buf.append(kernel_mean.detach())
        if len(self.buf) > self.max_len:
            self.buf.pop(0)

    @torch.no_grad()
    def fetch(self, current_query: torch.Tensor = None) -> Optional[torch.Tensor]:
        """
        If attention is enabled and current_query provided, compute cosine attention
        over temporal memory; otherwise return temporal mean. Output: [B, C]
        current_query: [B, Q, C] or [B, C]
        """
        if len(self.buf) == 0:
            return None
        # align on min batch size across stored entries
        min_b = min(t.shape[0] for t in self.buf)
        if min_b == 0:
            return None
        stacked = torch.stack([t[:min_b] for t in self.buf], dim=0)  # [T, B, C]
        if self.use_attention and (current_query is not None):
            if current_query.dim() == 3:
                # [B, Q, C] -> [B, C]
                cq = current_query[:, :, :].mean(dim=1)
            else:
                cq = current_query  # [B, C]
            # cosine sim over time dimension
            # normalize
            cq_norm = cq / (cq.norm(dim=-1, keepdim=True) + 1e-6)         # [B, C]
            mem_norm = stacked / (stacked.norm(dim=-1, keepdim=True) + 1e-6)  # [T, B, C]
            # [T, B]
            sim = (mem_norm * cq_norm.unsqueeze(0)).sum(dim=-1)
            w = torch.softmax(sim, dim=0).unsqueeze(-1)  # [T, B, 1]
            fused = (w * stacked).sum(dim=0)  # [B, C]
            return fused
        return stacked.mean(dim=0)  # [B, C]


