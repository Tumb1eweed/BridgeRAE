from __future__ import annotations

import torch
import torch.nn as nn
from tqdm.auto import tqdm


class LatentNormalizer(nn.Module):
    """Channel-wise normalization for encoder latent tokens.

    Collects running mean/std from data, then normalizes/denormalizes.
    Uses registered buffers so stats move with device and save/load automatically.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('std', torch.ones(dim))
        self.register_buffer('_collected', torch.tensor(False))

    @property
    def is_collected(self) -> bool:
        return bool(self._collected.item())

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * (self.std + self.eps) + self.mean

    @torch.no_grad()
    def collect_stats(
        self,
        encoder: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        max_batches: int | None = None,
    ) -> None:
        """One pass over data to compute channel-wise mean and std."""
        encoder.eval()
        sum_x = torch.zeros(self.dim, device=device, dtype=torch.float64)
        sum_x2 = torch.zeros(self.dim, device=device, dtype=torch.float64)
        total_tokens = 0

        num_batches = 0
        for batch in tqdm(dataloader, desc='Collecting latent stats', leave=False):
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            enc = encoder(complete_points)
            tokens = enc.tokens.float()  # (B, N, D)
            tokens_flat = tokens.reshape(-1, self.dim).to(torch.float64)
            sum_x += tokens_flat.sum(dim=0)
            sum_x2 += (tokens_flat ** 2).sum(dim=0)
            total_tokens += tokens_flat.shape[0]
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

        mean = (sum_x / total_tokens).float()
        std = ((sum_x2 / total_tokens - mean.double() ** 2).clamp_min(0).sqrt()).float()
        self.mean.copy_(mean)
        self.std.copy_(std)
        self._collected.fill_(True)
