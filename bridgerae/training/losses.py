from __future__ import annotations

import torch


def chamfer_distance_l2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, target, p=2)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return pred_to_target.mean() + target_to_pred.mean()


def chamfer_distance_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dist = torch.cdist(pred, target, p=1)
    pred_to_target = dist.min(dim=2)[0]
    target_to_pred = dist.min(dim=1)[0]
    return pred_to_target.mean() + target_to_pred.mean()
