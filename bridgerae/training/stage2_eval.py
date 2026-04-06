from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bridgerae.datasets import build_pointcloud_dataset, resolve_point_counts, shapenet_point_collate_fn
from bridgerae.models import LatentNormalizer, LatentTransportModel, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l1
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-2 latent transport evaluation')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'pcn'])
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--pair-data-root', type=Path, default=None)
    parser.add_argument('--num-input-points', type=int, default=None)
    parser.add_argument('--num-complete-points', type=int, default=None)
    parser.add_argument('--split-set', type=str, default=None)
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--stage2-ckpt', type=Path, required=True)
    parser.add_argument('--stage1-ckpt', type=Path, default=None)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--transport-steps', type=int, default=8)
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    return parser.parse_args()


def _build_decoder(device: torch.device, output_points: int, refine: bool = False) -> QueryCompletionDecoder:
    return QueryCompletionDecoder(
        hidden_dim=384,
        num_queries=256,
        num_heads=6,
        depth=6,
        output_points=output_points,
        refine=refine,
    ).to(device)


def load_transport_and_decoder(
    stage2_checkpoint_path: Path,
    stage1_checkpoint_path: Path | None,
    device: torch.device,
    output_points: int,
) -> tuple[LatentTransportModel, QueryCompletionDecoder, LatentNormalizer | None, str]:
    stage2_checkpoint = torch.load(stage2_checkpoint_path, map_location='cpu')

    transport = LatentTransportModel(hidden_dim=384, depth=6, num_heads=6).to(device)
    transport.load_state_dict(stage2_checkpoint['transport'])
    transport.eval()

    decoder_state = stage2_checkpoint.get('decoder', {})
    refine = 'seed_head.weight' in decoder_state
    decoder = _build_decoder(device, output_points, refine=refine)
    decoder_source = 'stage2'
    normalizer: LatentNormalizer | None = None
    if 'decoder' in stage2_checkpoint:
        decoder.load_state_dict(stage2_checkpoint['decoder'])
        if 'normalizer' in stage2_checkpoint:
            normalizer = LatentNormalizer(dim=384).to(device)
            normalizer.load_state_dict(stage2_checkpoint['normalizer'])
            normalizer.eval()
    else:
        if stage1_checkpoint_path is None:
            raise ValueError(
                'No decoder found in stage2 checkpoint. Please provide --stage1-ckpt for fallback decoder loading.'
            )
        stage1_checkpoint = torch.load(stage1_checkpoint_path, map_location='cpu')
        s1_refine = stage1_checkpoint.get('args', {}).get('refine', False)
        if s1_refine != refine:
            decoder = _build_decoder(device, output_points, refine=s1_refine)
        decoder.load_state_dict(stage1_checkpoint['decoder'])
        decoder_source = 'stage1'
        if 'normalizer' in stage1_checkpoint:
            normalizer = LatentNormalizer(dim=384).to(device)
            normalizer.load_state_dict(stage1_checkpoint['normalizer'])
            normalizer.eval()
    decoder.eval()

    return transport, decoder, normalizer, decoder_source


def main() -> None:
    args = parse_args()
    args.num_input_points, args.num_complete_points = resolve_point_counts(
        args.dataset,
        args.num_input_points,
        args.num_complete_points,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage2_eval.py')

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
    transport, decoder, normalizer, decoder_source = load_transport_and_decoder(
        args.stage2_ckpt,
        args.stage1_ckpt,
        device,
        output_points=args.num_complete_points,
    )

    print(json.dumps({
        'stage2_ckpt': str(args.stage2_ckpt),
        'stage1_ckpt': str(args.stage1_ckpt) if args.stage1_ckpt is not None else None,
        'decoder_source': decoder_source,
        'dataset': args.dataset,
        'num_input_points': args.num_input_points,
        'num_complete_points': args.num_complete_points,
        'split': args.split,
        'split_set': args.split_set,
        'class_id': args.class_id or 'all',
        'dataset_size': len(dataset),
    }), flush=True)

    totals = {
        'loss': 0.0,
        'chamfer_distance': 0.0,
        'chamfer_distance_l1': 0.0,
        'emd': 0.0,
        'f1_1pct': 0.0,
        'iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            src = encoder(partial_points)
            src_tokens = normalizer.normalize(src.tokens) if normalizer is not None else src.tokens
            pred_tokens = transport.transport(src_tokens, src.centers, num_steps=args.transport_steps)
            pred_tokens_dec = normalizer.denormalize(pred_tokens) if normalizer is not None else pred_tokens
            pred_points = decoder(pred_tokens_dec, src.centers).coarse_points
            loss = chamfer_distance_l1(pred_points, complete_points)
            metrics = compute_completion_metrics(pred_points, complete_points, iou_resolution=args.iou_resolution, metric_points=args.metric_points)
            totals['loss'] += float(loss.item())
            for key in ('chamfer_distance', 'chamfer_distance_l1', 'emd', 'f1_1pct', 'iou'):
                totals[key] += metrics[key]
            num_batches += 1
            print(json.dumps({'batch_idx': batch_idx, 'loss': float(loss.item()), **metrics}), flush=True)
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
        'decoder_source': decoder_source,
    })
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
