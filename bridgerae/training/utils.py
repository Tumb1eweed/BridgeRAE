from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader, Subset

from bridgerae.datasets import build_pointcloud_dataset, shapenet_point_collate_fn


def build_loader(
    args: argparse.Namespace,
    split: str,
    split_set: str,
    shuffle: bool,
    batch_size: int,
    max_samples: int | None = None,
) -> DataLoader:
    dataset = build_pointcloud_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split=split,
        class_id=args.class_id,
        num_input_points=args.num_input_points,
        num_complete_points=args.num_complete_points,
        split_set=split_set,
        pair_data_root=args.pair_data_root,
    )
    if max_samples is not None:
        dataset = Subset(dataset, range(min(max_samples, len(dataset))))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=shapenet_point_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )


def build_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    lr: float,
    lr_min: float,
    last_epoch: int = -1,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return max((epoch_idx + 1) / max(warmup_epochs, 1), lr_min / lr)
        progress = (epoch_idx - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(cosine, lr_min / lr)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
