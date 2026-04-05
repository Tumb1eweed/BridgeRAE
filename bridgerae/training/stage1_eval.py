from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bridgerae.datasets import build_pointcloud_dataset, resolve_point_counts, shapenet_point_collate_fn
from bridgerae.models import LatentNormalizer, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 evaluation')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'pcn'])
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--pair-data-root', type=Path, default=None)
    parser.add_argument('--num-input-points', type=int, default=None)
    parser.add_argument('--num-complete-points', type=int, default=None)
    parser.add_argument('--split-set', type=str, default=None)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    return parser.parse_args()


def load_decoder(checkpoint_path: Path, device: torch.device, output_points: int) -> tuple[QueryCompletionDecoder, LatentNormalizer | None]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = QueryCompletionDecoder(
        hidden_dim=384,
        num_queries=256,
        num_heads=6,
        depth=6,
        output_points=output_points,
    ).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    normalizer: LatentNormalizer | None = None
    if 'normalizer' in checkpoint:
        normalizer = LatentNormalizer(dim=384).to(device)
        normalizer.load_state_dict(checkpoint['normalizer'])
        normalizer.eval()
    return decoder, normalizer


def main() -> None:
    args = parse_args()
    args.num_input_points, args.num_complete_points = resolve_point_counts(
        args.dataset,
        args.num_input_points,
        args.num_complete_points,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage1_eval.py')

    device = torch.device('cuda')
    dataset = build_pointcloud_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split=args.split,
        class_id=args.class_id,
        num_input_points=args.num_input_points,
        num_complete_points=args.num_complete_points,
        split_set=args.split_set,
        pair_data_root=args.pair_data_root,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=shapenet_point_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder, normalizer = load_decoder(args.checkpoint, device=device, output_points=args.num_complete_points)

    totals = {
        'chamfer_distance': 0.0,
        'chamfer_distance_l1': 0.0,
        'emd': 0.0,
        'iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            enc = encoder(complete_points)
            tokens = enc.tokens
            if normalizer is not None:
                tokens = normalizer.normalize(tokens)
            pred = decoder(tokens, enc.centers).coarse_points
            metrics = compute_completion_metrics(pred, complete_points, iou_resolution=args.iou_resolution, metric_points=args.metric_points)
            for key, value in metrics.items():
                totals[key] += value
            num_batches += 1
            print(json.dumps({'batch_idx': batch_idx, **metrics}), flush=True)

            if args.max_batches is not None and num_batches >= args.max_batches:
                break

    result = {key: value / max(num_batches, 1) for key, value in totals.items()}
    result.update({
        'num_batches': num_batches,
        'dataset': args.dataset,
        'num_input_points': args.num_input_points,
        'num_complete_points': args.num_complete_points,
        'split': args.split,
        'split_set': args.split_set,
        'class_id': args.class_id or 'all',
    })
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
