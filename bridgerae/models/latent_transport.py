from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .completion_decoder import CrossAttention, Mlp, SelfAttention


@dataclass(frozen=True)
class LatentTransportOutput:
    velocity: torch.Tensor
    transported_tokens: torch.Tensor


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half))
        self.register_buffer('freqs', freqs)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_proj = t[:, None].float() * self.freqs[None, :]  # (B, half)
        emb = torch.cat([t_proj.sin(), t_proj.cos()], dim=-1)  # (B, dim)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return self.mlp(emb)


class AdaLN(nn.Module):
    """Adaptive Layer Normalization conditioned on a per-sample vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D), cond: (B, D)
        scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class LatentTransportBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.self_norm = AdaLN(dim)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.cross_norm_q = AdaLN(dim)
        self.cross_norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.mlp_norm = AdaLN(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.self_norm(x, time_cond))
        x = x + self.cross_attn(self.cross_norm_q(x, time_cond), self.cross_norm_ctx(cond))
        x = x + self.mlp(self.mlp_norm(x, time_cond))
        return x


class LatentTransportModel(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = TimeEmbedding(hidden_dim)
        self.center_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            LatentTransportBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])
        self.norm = AdaLN(hidden_dim)
        self.velocity_head = nn.Linear(hidden_dim, hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
        # Zero-init velocity head so transport starts as identity (v≈0)
        nn.init.zeros_(self.velocity_head.weight)
        nn.init.zeros_(self.velocity_head.bias)
        # Zero-init AdaLN projections so modulation starts as standard LN
        for module in self.modules():
            if isinstance(module, AdaLN):
                nn.init.zeros_(module.proj[-1].weight)
                nn.init.zeros_(module.proj[-1].bias)

    def forward(
        self,
        state_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        centers: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> LatentTransportOutput:
        time_cond = self.time_embed(time_steps)  # (B, D)
        center_embed = self.center_pos_embed(centers)
        x = self.input_proj(state_tokens) + center_embed
        cond = self.cond_proj(source_tokens) + center_embed
        for block in self.blocks:
            x = block(x, cond, time_cond)
        velocity = self.velocity_head(self.norm(x, time_cond))
        transported_tokens = state_tokens + velocity
        return LatentTransportOutput(velocity=velocity, transported_tokens=transported_tokens)

    def transport_train(
        self,
        source_tokens: torch.Tensor,
        centers: torch.Tensor,
        num_steps: int = 8,
    ) -> torch.Tensor:
        """Euler integration with gradients for training."""
        z = source_tokens
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = torch.full(
                (source_tokens.shape[0],),
                step / float(num_steps),
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            )
            velocity = self(z, source_tokens, centers, t).velocity
            z = z + dt * velocity
        return z

    @torch.no_grad()
    def transport(
        self,
        source_tokens: torch.Tensor,
        centers: torch.Tensor,
        num_steps: int = 8,
    ) -> torch.Tensor:
        z = source_tokens
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t = torch.full((source_tokens.shape[0],), step / float(num_steps), device=source_tokens.device, dtype=source_tokens.dtype)
            velocity = self(z, source_tokens, centers, t).velocity
            z = z + dt * velocity
        return z
