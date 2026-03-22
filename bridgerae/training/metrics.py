from __future__ import annotations

import sys
from pathlib import Path

import torch

from .losses import chamfer_distance_l1, chamfer_distance_l2


_EMD_IMPORT_OK = False
_EMD_IMPORT_ERROR: Exception | None = None
_POINTMAE_EMD_PATH = Path('/root/autodl-tmp/projects/Point-MAE/extensions/emd')
if _POINTMAE_EMD_PATH.exists():
    sys.path.insert(0, str(_POINTMAE_EMD_PATH))
    try:
        from emd import earth_mover_distance  # type: ignore

        _EMD_IMPORT_OK = True
    except Exception as exc:  # pragma: no cover
        _EMD_IMPORT_ERROR = exc


@torch.no_grad()
def chamfer_distance_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return chamfer_distance_l2(pred, target)


@torch.no_grad()
def chamfer_distance_l1_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return chamfer_distance_l1(pred, target)


@torch.no_grad()
def earth_mover_distance_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not _EMD_IMPORT_OK:
        raise RuntimeError(f'EMD extension is not available: {_EMD_IMPORT_ERROR}')
    emd_fn = earth_mover_distance().to(pred.device)
    return emd_fn(pred, target)


@torch.no_grad()
def voxel_iou_metric(pred: torch.Tensor, target: torch.Tensor, resolution: int = 32) -> torch.Tensor:
    pred_occ = _points_to_occupancy(pred, resolution)
    target_occ = _points_to_occupancy(target, resolution)
    intersection = (pred_occ & target_occ).sum(dim=(1, 2, 3)).float()
    union = (pred_occ | target_occ).sum(dim=(1, 2, 3)).float().clamp_min(1.0)
    return (intersection / union).mean()


@torch.no_grad()
def compute_completion_metrics(pred: torch.Tensor, target: torch.Tensor, iou_resolution: int = 32) -> dict[str, float]:
    cd_l2 = float(chamfer_distance_metric(pred, target).item()) * 100.0
    cd_l1 = float(chamfer_distance_l1_metric(pred, target).item())
    emd = float(earth_mover_distance_metric(pred, target).item())
    iou = float(voxel_iou_metric(pred, target, resolution=iou_resolution).item()) * 100.0
    return {
        'chamfer_distance': cd_l2,
        'chamfer_distance_l1': cd_l1,
        'emd': emd,
        'iou': iou,
    }


@torch.no_grad()
def _points_to_occupancy(points: torch.Tensor, resolution: int) -> torch.Tensor:
    coords = ((points.clamp(-1.0, 1.0) + 1.0) * 0.5 * (resolution - 1)).round().long()
    coords = coords.clamp_(0, resolution - 1)
    batch_size = coords.shape[0]
    occupancy = torch.zeros((batch_size, resolution, resolution, resolution), dtype=torch.bool, device=points.device)
    batch_idx = torch.arange(batch_size, device=points.device).view(-1, 1).expand(-1, coords.shape[1])
    occupancy[batch_idx, coords[..., 2], coords[..., 1], coords[..., 0]] = True
    return occupancy
