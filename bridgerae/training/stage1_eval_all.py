from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 evaluate all checkpoints')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--max-batches', type=int, default=None)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--output-jsonl', type=Path, default=None)
    return parser.parse_args()


def load_decoder_state(checkpoint_path: Path, device: torch.device) -> tuple[QueryCompletionDecoder, dict]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    return decoder, checkpoint


def evaluate_checkpoint(
    checkpoint_path: Path,
    encoder: PointMAEEncoder,
    loader: DataLoader,
    device: torch.device,
    iou_resolution: int,
    max_batches: int | None,
) -> dict[str, float]:
    decoder, checkpoint = load_decoder_state(checkpoint_path, device)
    totals = {
        'eval_loss': 0.0,
        'eval_chamfer_distance': 0.0,
        'eval_chamfer_distance_l1': 0.0,
        'eval_emd': 0.0,
        'eval_iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            enc = encoder(partial_points)
            pred = decoder(enc.tokens, enc.centers).coarse_points
            loss = chamfer_distance_l2(pred, complete_points)
            metrics = compute_completion_metrics(pred, complete_points, iou_resolution=iou_resolution)
            totals['eval_loss'] += float(loss.item())
            totals['eval_chamfer_distance'] += metrics['chamfer_distance']
            totals['eval_chamfer_distance_l1'] += metrics['chamfer_distance_l1']
            totals['eval_emd'] += metrics['emd']
            totals['eval_iou'] += metrics['iou']
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    averaged = {key: value / max(num_batches, 1) for key, value in totals.items()}
    epoch = int(checkpoint.get('epoch', -1))
    step = int(checkpoint.get('step', -1))
    return {
        'epoch': epoch,
        'step': step,
        **averaged,
        'checkpoint': str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage1_eval_all.py')

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

    ckpts = sorted(args.ckpt_dir.glob('epoch_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f'No epoch checkpoints found in {args.ckpt_dir}')

    output_jsonl = args.output_jsonl or (args.ckpt_dir / f'{args.split}_eval_metrics.jsonl')
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open('w', encoding='utf-8') as f:
        for ckpt_path in ckpts:
            result = evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                encoder=encoder,
                loader=loader,
                device=device,
                iou_resolution=args.iou_resolution,
                max_batches=args.max_batches,
            )
            line = json.dumps(result)
            print(line, flush=True)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
