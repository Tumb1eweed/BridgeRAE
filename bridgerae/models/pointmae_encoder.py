from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from pointnet2_ops import pointnet2_utils
from timm.models.layers import DropPath, trunc_normal_

try:
    from knn_cuda import KNN
except Exception:  # pragma: no cover
    KNN = None


@dataclass(frozen=True)
class PointMAEEncoderOutput:
    tokens: torch.Tensor
    centers: torch.Tensor
    neighborhoods: torch.Tensor


def fps(points: torch.Tensor, num_group: int) -> torch.Tensor:
    fps_idx = pointnet2_utils.furthest_point_sample(points, num_group)
    return pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()


class Group(nn.Module):
    def __init__(self, num_group: int, group_size: int) -> None:
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=group_size, transpose_mode=True) if KNN is not None else None

    @staticmethod
    def _square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        dist = -2 * torch.matmul(src, dst.transpose(1, 2))
        dist += torch.sum(src ** 2, dim=-1, keepdim=True)
        dist += torch.sum(dst ** 2, dim=-1).unsqueeze(1)
        return dist

    def _gather_neighbors(self, xyz: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = xyz.shape
        if self.knn is not None and xyz.is_cuda and centers.is_cuda:
            _, idx = self.knn(xyz, centers)
        else:
            distances = self._square_distance(centers, xyz)
            idx = distances.topk(k=self.group_size, dim=-1, largest=False).indices
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).reshape(-1)
        neighborhoods = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhoods = neighborhoods.view(batch_size, centers.shape[1], self.group_size, 3).contiguous()
        return neighborhoods - centers.unsqueeze(2)

    def forward(self, xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centers = fps(xyz, self.num_group)
        neighborhoods = self._gather_neighbors(xyz, centers)
        return neighborhoods, centers

    def forward_with_centers(self, xyz: torch.Tensor, centers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        neighborhoods = self._gather_neighbors(xyz, centers)
        return neighborhoods, centers


class LocalEncoder(nn.Module):
    def __init__(self, encoder_channel: int) -> None:
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        batch_size, num_group, group_size, _ = point_groups.shape
        point_groups = point_groups.reshape(batch_size * num_group, group_size, 3)
        feature = self.first_conv(point_groups.transpose(2, 1))
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]
        feature = torch.cat([feature_global.expand(-1, -1, group_size), feature], dim=1)
        feature = self.second_conv(feature)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]
        return feature_global.reshape(batch_size, num_group, self.encoder_channel)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Attention(nn.Module):
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
        batch_size, num_tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, depth: int, num_heads: int, drop_path_rate: float = 0.0) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([Block(dim=embed_dim, num_heads=num_heads, drop_path=dpr[i]) for i in range(depth)])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x + pos)
        return x


class PointMAEEncoderBackbone(nn.Module):
    def __init__(self, trans_dim: int = 384, encoder_dims: int = 384, depth: int = 12, num_heads: int = 6, drop_path_rate: float = 0.1) -> None:
        super().__init__()
        self.encoder = LocalEncoder(encoder_channel=encoder_dims)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )
        self.blocks = TransformerEncoder(embed_dim=trans_dim, depth=depth, num_heads=num_heads, drop_path_rate=drop_path_rate)
        self.norm = nn.LayerNorm(trans_dim)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, neighborhoods: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        group_tokens = self.encoder(neighborhoods)
        pos = self.pos_embed(centers)
        x = self.blocks(group_tokens, pos)
        return self.norm(x)


class PointMAEEncoder(nn.Module):
    def __init__(
        self,
        num_group: int = 64,
        group_size: int = 32,
        trans_dim: int = 384,
        encoder_dims: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        drop_path_rate: float = 0.1,
        pretrained_ckpt: str | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        self.output_dim = trans_dim
        self.group_divider = Group(num_group=num_group, group_size=group_size)
        self.backbone = PointMAEEncoderBackbone(
            trans_dim=trans_dim,
            encoder_dims=encoder_dims,
            depth=depth,
            num_heads=num_heads,
            drop_path_rate=drop_path_rate,
        )
        if pretrained_ckpt is not None:
            self.load_pretrained(pretrained_ckpt)
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = True
        self.train()

    def load_pretrained(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_state = checkpoint.get('base_model', checkpoint)
        mapped_state = {}
        prefix = 'module.MAE_encoder.'
        for key, value in base_state.items():
            if key.startswith(prefix):
                mapped_state[key[len(prefix):]] = value
        missing, unexpected = self.backbone.load_state_dict(mapped_state, strict=False)
        if missing:
            print(f'[PointMAEEncoder] missing keys: {missing}')
        if unexpected:
            print(f'[PointMAEEncoder] unexpected keys: {unexpected}')

    def forward(self, points: torch.Tensor, centers: torch.Tensor | None = None) -> PointMAEEncoderOutput:
        if centers is None:
            neighborhoods, centers = self.group_divider(points)
        else:
            neighborhoods, centers = self.group_divider.forward_with_centers(points, centers)
        tokens = self.backbone(neighborhoods, centers)
        return PointMAEEncoderOutput(tokens=tokens, centers=centers, neighborhoods=neighborhoods)
