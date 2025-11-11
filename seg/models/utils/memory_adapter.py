import torch
from typing import Optional

class MemoryAdapter:
    """
    Minimal streaming memory adapter for video/VOS.
    Stores a lightweight summary of recent object kernels per batch index.
    """
    def __init__(self, max_len: int = 5):
        self.max_len = max_len
        self.buf = []  # list of tensors [B, C] per time

    @torch.no_grad()
    def reset(self):
        self.buf = []

    @torch.no_grad()
    def update(self, object_kernels: torch.Tensor):
        """
        object_kernels: [B, Q, C]
        Store per-batch mean kernel as memory summary.
        """
        if object_kernels is None:
            return
        kernel_mean = object_kernels.mean(dim=1)  # [B, C]
        self.buf.append(kernel_mean.detach())
        if len(self.buf) > self.max_len:
            self.buf.pop(0)

    @torch.no_grad()
    def fetch(self) -> Optional[torch.Tensor]:
        """
        Returns averaged memory summary over time: [B, C]
        """
        if len(self.buf) == 0:
            return None
        # align on min batch size across stored entries
        min_b = min(t.shape[0] for t in self.buf)
        if min_b == 0:
            return None
        stacked = torch.stack([t[:min_b] for t in self.buf], dim=0)  # [T, B, C]
        return stacked.mean(dim=0)  # [B, C]


