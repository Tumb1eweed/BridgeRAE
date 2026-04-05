from __future__ import annotations

import sys
from pathlib import Path

import torch


_CHAMFER_IMPORT_ERROR: Exception | None = None
_CHAMFER_AVAILABLE = False
_CHAMFER_EXT_PATH = Path('/root/autodl-tmp/projects/Point-MAE/extensions/chamfer_dist')
if _CHAMFER_EXT_PATH.exists():
    sys.path.insert(0, str(_CHAMFER_EXT_PATH))
try:
    import chamfer  # type: ignore

    _CHAMFER_AVAILABLE = True
except Exception as exc:  # pragma: no cover
    _CHAMFER_IMPORT_ERROR = exc


class _ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1: torch.Tensor, xyz2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist1, dist2, idx1, idx2 = chamfer.forward(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1: torch.Tensor, grad_dist2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = chamfer.backward(xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2)
        return grad_xyz1, grad_xyz2


def _chamfer_extension_distances(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not _CHAMFER_AVAILABLE:
        raise RuntimeError(f'Chamfer extension is not available: {_CHAMFER_IMPORT_ERROR}')
    if pred.is_cuda and target.is_cuda:
        return _ChamferFunction.apply(pred.float(), target.float())
    raise RuntimeError('Chamfer extension requires CUDA tensors')


def chamfer_distance_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if _CHAMFER_AVAILABLE and pred.is_cuda and target.is_cuda:
        dist1_sq, dist2_sq = _chamfer_extension_distances(pred, target)
        return dist1_sq.clamp_min(0.0).sqrt().mean() + dist2_sq.clamp_min(0.0).sqrt().mean()

    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return pred_to_target.mean() + target_to_pred.mean()


def chamfer_distance_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, target, p=1)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return pred_to_target.mean() + target_to_pred.mean()


def get_chamfer_backend(device: torch.device | None = None) -> str:
    if device is not None and device.type == 'cuda' and _CHAMFER_AVAILABLE:
        return 'chamfer_extension'
    if device is not None and device.type != 'cuda':
        return 'torch_cdist_fallback'
    return 'chamfer_extension' if _CHAMFER_AVAILABLE else 'torch_cdist_fallback'


def repulsion_loss(pred: torch.Tensor, k: int = 8, eps: float = 1e-6) -> torch.Tensor:
    """Penalize nearby points to encourage uniform distribution.

    For each point, computes the mean negative squared distance to its k nearest
    neighbors (excluding itself).  Minimizing this pushes points apart.

    Args:
        pred: (B, N, 3) predicted point cloud.
        k: number of nearest neighbors.
        eps: small constant for numerical stability.
    """
    # (B, N, N) pairwise squared distances
    diff = pred.unsqueeze(2) - pred.unsqueeze(1)  # (B, N, N, 3)
    dist_sq = (diff * diff).sum(dim=-1)            # (B, N, N)
    # exclude self (diagonal) by setting to large value
    diag_mask = torch.eye(pred.shape[1], device=pred.device, dtype=torch.bool).unsqueeze(0)
    dist_sq = dist_sq.masked_fill(diag_mask, float('inf'))
    # k nearest neighbors
    knn_sq, _ = dist_sq.topk(k, dim=-1, largest=False)  # (B, N, k)
    # weight: closer neighbors get stronger penalty  h(r) = max(0, eps - r)
    knn_dist = knn_sq.clamp_min(0).sqrt()  # (B, N, k)
    penalty = (-knn_dist).exp()  # exponential repulsion
    return penalty.mean()
