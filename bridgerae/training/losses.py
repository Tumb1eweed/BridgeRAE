from __future__ import annotations

import sys
from pathlib import Path

import torch

try:
    from knn_cuda import KNN
except Exception:  # pragma: no cover
    KNN = None


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
        return 0.5 * (dist1_sq.clamp_min(0.0).mean() + dist2_sq.clamp_min(0.0).mean())

    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return 0.5 * (pred_to_target.pow(2).mean() + target_to_pred.pow(2).mean())


def chamfer_distance_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if _CHAMFER_AVAILABLE and pred.is_cuda and target.is_cuda:
        dist1_sq, dist2_sq = _chamfer_extension_distances(pred, target)
        return 0.5 * (dist1_sq.clamp_min(1e-12).sqrt().mean() + dist2_sq.clamp_min(1e-12).sqrt().mean())

    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return 0.5 * (pred_to_target.mean() + target_to_pred.mean())


def get_chamfer_backend(device: torch.device | None = None) -> str:
    if device is not None and device.type == 'cuda' and _CHAMFER_AVAILABLE:
        return 'chamfer_extension'
    if device is not None and device.type != 'cuda':
        return 'torch_cdist_fallback'
    return 'chamfer_extension' if _CHAMFER_AVAILABLE else 'torch_cdist_fallback'


def repulsion_loss(pred: torch.Tensor, k: int = 8, eps: float = 1e-6, chunk_size: int = 64) -> torch.Tensor:
    """Penalize nearby points to encourage uniform distribution.

    For each point, computes the mean negative squared distance to its k nearest
    neighbors (excluding itself).  Minimizing this pushes points apart.

    Args:
        pred: (B, N, 3) predicted point cloud.
        k: number of nearest neighbors.
        eps: small constant for numerical stability.
        chunk_size: query chunk size for fallback KNN computation.
    """
    batch_size, num_points, _ = pred.shape
    if num_points <= 1:
        return pred.new_tensor(0.0)

    neighbor_count = min(k + 1, num_points)
    search_points = pred.detach()
    batch_idx = torch.arange(batch_size, device=pred.device).view(batch_size, 1, 1)

    if KNN is not None and pred.is_cuda:
        _, idx = KNN(k=neighbor_count, transpose_mode=True)(search_points, search_points)
        neighbors = pred[batch_idx, idx]
        knn_dist = (neighbors - pred.unsqueeze(2)).pow(2).sum(dim=-1).clamp_min(0.0).sqrt()
        if neighbor_count > k:
            knn_dist = knn_dist[:, :, 1:]
        else:
            knn_dist = knn_dist[:, :, :k]
    else:
        knn_chunks: list[torch.Tensor] = []
        arange_all = torch.arange(num_points, device=pred.device)
        for start in range(0, num_points, chunk_size):
            end = min(start + chunk_size, num_points)
            chunk = search_points[:, start:end, :]
            dist = torch.cdist(chunk, search_points, p=2)
            diag_idx = arange_all[start:end] - start
            dist[:, diag_idx, arange_all[start:end]] = float('inf')
            knn_chunks.append(dist.topk(min(k, num_points - 1), dim=-1, largest=False).indices)
        idx = torch.cat(knn_chunks, dim=1)
        neighbors = pred[batch_idx, idx]
        knn_dist = (neighbors - pred.unsqueeze(2)).pow(2).sum(dim=-1).clamp_min(0.0).sqrt()

    # closer neighbors get stronger penalty
    penalty = (-knn_dist.clamp_min(eps)).exp()
    return penalty.mean()
