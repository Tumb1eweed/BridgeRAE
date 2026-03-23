from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import LatentTransportModel, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-2 latent transport evaluation')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--stage2-ckpt', type=Path, required=True)
    parser.add_argument('--stage1-ckpt', type=Path, required=True)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--transport-steps', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    return parser.parse_args()


def load_stage1_decoder(checkpoint_path: Path, device: torch.device) -> QueryCompletionDecoder:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    return decoder


def load_stage2_transport(checkpoint_path: Path, device: torch.device) -> LatentTransportModel:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = LatentTransportModel(hidden_dim=384, depth=6, num_heads=6).to(device)
    model.load_state_dict(checkpoint['transport'])
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage2_eval.py')

    device = torch.device('cuda')
    dataset = ShapeNetPointCloudDataset(
        data_root=args.data_root,
        split=args.split,
        class_id=args.class_id,
        num_input_points=2048,
        num_complete_points=8192,
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
    decoder = load_stage1_decoder(args.stage1_ckpt, device)
    transport = load_stage2_transport(args.stage2_ckpt, device)

    totals = {
        'loss': 0.0,
        'chamfer_distance': 0.0,
        'chamfer_distance_l1': 0.0,
        'emd': 0.0,
        'iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            src = encoder(partial_points)
            pred_tokens = transport.transport(src.tokens, src.centers, num_steps=args.transport_steps)
            pred_points = decoder(pred_tokens, src.centers).coarse_points
            loss = chamfer_distance_l2(pred_points, complete_points)
            metrics = compute_completion_metrics(pred_points, complete_points, iou_resolution=args.iou_resolution, metric_points=args.metric_points)
            totals['loss'] += float(loss.item())
            for key in ('chamfer_distance', 'chamfer_distance_l1', 'emd', 'iou'):
                totals[key] += metrics[key]
            num_batches += 1
            print(json.dumps({'batch_idx': batch_idx, 'loss': float(loss.item()), **metrics}), flush=True)
            if args.max_batches is not None and num_batches >= args.max_batches:
                break

    result = {key: value / max(num_batches, 1) for key, value in totals.items()}
    result.update({'num_batches': num_batches, 'split': args.split, 'class_id': args.class_id or 'all'})
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
