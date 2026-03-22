from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


@dataclass(frozen=True)
class CompletionDecoderOutput:
    query_tokens: torch.Tensor
    coarse_points: torch.Tensor


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        bsz, nq, dim = query.shape
        nk = context.shape[1]
        q = self.q(query).reshape(bsz, nq, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, nq, dim)
        return self.proj_drop(self.proj(x))


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, ntok, dim = x.shape
        qkv = self.qkv(x).reshape(bsz, ntok, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(bsz, ntok, dim)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QueryDecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.cross_norm_q = nn.LayerNorm(dim)
        self.cross_norm_ctx = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio=mlp_ratio, drop=drop)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = query + self.self_attn(self.self_norm(query))
        query = query + self.cross_attn(self.cross_norm_q(query), self.cross_norm_ctx(context))
        query = query + self.mlp(self.mlp_norm(query))
        return query


class QueryCompletionDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 384,
        num_queries: int = 256,
        num_heads: int = 6,
        depth: int = 6,
        mlp_ratio: float = 4.0,
        output_points: int = 2048,
    ) -> None:
        super().__init__()
        if output_points % num_queries != 0:
            raise ValueError('output_points must be divisible by num_queries for the current coarse decoder.')
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.output_points = output_points
        self.points_per_query = output_points // num_queries

        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        self.query_pos = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
        self.center_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            QueryDecoderBlock(dim=hidden_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.point_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.points_per_query * 3),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        trunc_normal_(self.query_tokens, std=0.02)
        trunc_normal_(self.query_pos, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, encoder_tokens: torch.Tensor, encoder_centers: torch.Tensor) -> CompletionDecoderOutput:
        batch_size = encoder_tokens.shape[0]
        query = self.query_tokens.expand(batch_size, -1, -1) + self.query_pos.expand(batch_size, -1, -1)
        context = encoder_tokens + self.center_pos_embed(encoder_centers)
        for block in self.blocks:
            query = block(query, context)
        query = self.norm(query)
        coarse_points = self.point_head(query).reshape(batch_size, self.output_points, 3)
        return CompletionDecoderOutput(query_tokens=query, coarse_points=coarse_points)
