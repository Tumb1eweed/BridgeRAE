from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from .completion_decoder import CrossAttention, Mlp, SelfAttention


@dataclass(frozen=True)
class LatentDiffusionDiTOutput:
    noise_pred: torch.Tensor
    x0_pred: torch.Tensor


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_dim: int, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = timesteps.float()[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return self.mlp(emb)


class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1 + scale[:, None, :]) + shift[:, None, :]


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.self_norm = AdaLayerNorm(hidden_dim)
        self.self_attn = SelfAttention(hidden_dim, num_heads=num_heads, proj_drop=drop)
        self.cross_norm_q = AdaLayerNorm(hidden_dim)
        self.cross_norm_ctx = nn.LayerNorm(hidden_dim)
        self.cross_attn = CrossAttention(hidden_dim, num_heads=num_heads, proj_drop=drop)
        self.mlp_norm = AdaLayerNorm(hidden_dim)
        self.mlp = Mlp(hidden_dim, mlp_ratio=mlp_ratio, drop=drop)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6),
        )

    def forward(self, x: torch.Tensor, cond_tokens: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_cross, scale_cross, gate_mlp = self.modulation(global_cond).chunk(6, dim=-1)
        x = x + gate_msa[:, None, :] * self.self_attn(self.self_norm(x, shift_msa, scale_msa))
        x = x + self.cross_attn(self.cross_norm_q(x, shift_cross, scale_cross), self.cross_norm_ctx(cond_tokens))
        x = x + gate_mlp[:, None, :] * self.mlp(self.mlp_norm(x, shift_cross, scale_cross))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = AdaLayerNorm(hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(global_cond).chunk(2, dim=-1)
        return self.proj(self.norm(x, shift, scale))


class LatentDiffusionDiT(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        diffusion_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.diffusion_steps = diffusion_steps

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), alphas_cumprod[:-1]], dim=0)

        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alphas_cumprod', alphas_cumprod, persistent=False)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev, persistent=False)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod), persistent=False)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod), persistent=False)
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod), persistent=False)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1.0), persistent=False)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance.clamp_min(1e-20), persistent=False)
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            persistent=False,
        )
        self.register_buffer(
            'posterior_mean_coef2',
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
            persistent=False,
        )

        self.time_embed = TimestepEmbedder(hidden_dim)
        self.center_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cond_pool = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop)
            for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.elementwise_affine:
                    nn.init.constant_(module.bias, 0)
                    nn.init.constant_(module.weight, 1.0)
        for block in self.blocks:
            nn.init.constant_(block.modulation[-1].weight, 0)
            nn.init.constant_(block.modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.proj.weight, 0)
        nn.init.constant_(self.final_layer.proj.bias, 0)

    def _extract(self, coeff: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = coeff.gather(0, timesteps)
        return out.view(target.shape[0], 1, 1).to(dtype=target.dtype)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        xt = self._extract(self.sqrt_alphas_cumprod, timesteps, x0) * x0
        xt = xt + self._extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0) * noise
        return xt, noise

    def predict_x0_from_noise(self, xt: torch.Tensor, timesteps: torch.Tensor, noise_pred: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_recip_alphas_cumprod, timesteps, xt) * xt - self._extract(self.sqrt_recipm1_alphas_cumprod, timesteps, xt) * noise_pred

    def forward(
        self,
        noisy_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        centers: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> LatentDiffusionDiTOutput:
        center_embed = self.center_pos_embed(centers)
        x = self.input_proj(noisy_tokens) + center_embed
        cond_tokens = self.cond_proj(source_tokens) + center_embed
        global_cond = self.time_embed(timesteps) + self.cond_pool(source_tokens.mean(dim=1))
        for block in self.blocks:
            x = block(x, cond_tokens, global_cond)
        noise_pred = self.final_layer(x, global_cond)
        x0_pred = self.predict_x0_from_noise(noisy_tokens, timesteps, noise_pred)
        return LatentDiffusionDiTOutput(noise_pred=noise_pred, x0_pred=x0_pred)

    @torch.no_grad()
    def sample(
        self,
        source_tokens: torch.Tensor,
        centers: torch.Tensor,
        num_steps: int = 100,
    ) -> torch.Tensor:
        batch_size = source_tokens.shape[0]
        device = source_tokens.device
        x = torch.randn_like(source_tokens)
        schedule = torch.linspace(self.diffusion_steps - 1, 0, steps=num_steps, device=device).round().long().unique_consecutive()

        for idx, timestep in enumerate(schedule):
            t = torch.full((batch_size,), int(timestep.item()), device=device, dtype=torch.long)
            out = self(x, source_tokens, centers, t)
            x0_pred = out.x0_pred.clamp(-5.0, 5.0)
            if idx == len(schedule) - 1:
                x = x0_pred
                continue
            posterior_mean = self._extract(self.posterior_mean_coef1, t, x) * x0_pred
            posterior_mean = posterior_mean + self._extract(self.posterior_mean_coef2, t, x) * x
            next_t = schedule[idx + 1]
            if int(next_t.item()) == 0:
                x = posterior_mean
                continue
            noise = torch.randn_like(x)
            x = posterior_mean + torch.sqrt(self._extract(self.posterior_variance, t, x)) * noise
        return x
