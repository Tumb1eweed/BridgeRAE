from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from bridgerae.datasets import build_pointcloud_dataset, resolve_point_counts, shapenet_point_collate_fn
from bridgerae.models import PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 smoke training')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'pcn'])
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--pair-data-root', type=Path, default=None)
    parser.add_argument('--num-input-points', type=int, default=None)
    parser.add_argument('--num-complete-points', type=int, default=None)
    parser.add_argument('--split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-samples', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.num_input_points, args.num_complete_points = resolve_point_counts(
        args.dataset,
        args.num_input_points,
        args.num_complete_points,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this smoke training script.')

    device = torch.device('cuda')
    dataset = build_pointcloud_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split='train',
        class_id=args.class_id,
        num_input_points=args.num_input_points,
        num_complete_points=args.num_complete_points,
        split_set=args.split_set,
        pair_data_root=args.pair_data_root,
    )
    subset = Subset(dataset, list(range(min(args.max_samples, len(dataset)))))
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=shapenet_point_collate_fn,
        drop_last=False,
    )

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = QueryCompletionDecoder(
        hidden_dim=384,
        num_queries=256,
        num_heads=6,
        depth=6,
        output_points=args.num_complete_points,
    ).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=0.05)

    decoder.train()
    step = 0
    while step < args.steps:
        for batch in loader:
            partial_points = batch['partial_points'].to(device)
            complete_points = batch['complete_points'].to(device)

            with torch.no_grad():
                enc = encoder(partial_points)
            dec = decoder(enc.tokens, enc.centers)
            loss = chamfer_distance_l2(dec.coarse_points, complete_points)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print({
                'step': step,
                'batch_size': int(partial_points.shape[0]),
                'partial_shape': tuple(partial_points.shape),
                'complete_shape': tuple(complete_points.shape),
                'pred_shape': tuple(dec.coarse_points.shape),
                'loss': float(loss.item()),
            })
            step += 1
            if step >= args.steps:
                break


if __name__ == '__main__':
    main()
