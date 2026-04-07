from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from bridgerae.datasets.factory import resolve_point_counts
from bridgerae.datasets.pcn_pairs import PCNPointCloudDataset
from bridgerae.datasets.shapenet_pairs import normalize_pair, sample_fixed_size
from bridgerae.models import PointMAEEncoder
from bridgerae.training.stage2_eval import load_transport_and_decoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export random stage-2 prediction pairs as PCD files')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--class-id', type=str, required=True)
    parser.add_argument('--stage2-ckpt', type=Path, required=True)
    parser.add_argument('--stage1-ckpt', type=Path, required=True)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--num-samples', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260406)
    parser.add_argument('--transport-steps', type=int, default=8)
    return parser.parse_args()


def write_ascii_pcd(path: Path, points: torch.Tensor) -> None:
    pts = points.detach().cpu().float().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        handle.write('# .PCD v0.7 - Point Cloud Data file format\n')
        handle.write('VERSION 0.7\n')
        handle.write('FIELDS x y z\n')
        handle.write('SIZE 4 4 4\n')
        handle.write('TYPE F F F\n')
        handle.write('COUNT 1 1 1\n')
        handle.write(f'WIDTH {len(pts)}\n')
        handle.write('HEIGHT 1\n')
        handle.write('VIEWPOINT 0 0 0 1 0 0 0\n')
        handle.write(f'POINTS {len(pts)}\n')
        handle.write('DATA ascii\n')
        for x, y, z in pts:
            handle.write(f'{x:.8f} {y:.8f} {z:.8f}\n')


def main() -> None:
    args = parse_args()
    num_input_points, num_complete_points = resolve_point_counts('pcn', None, None)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for export_stage2_random_pairs.py')

    dataset = PCNPointCloudDataset(
        data_root=args.data_root,
        split=args.split,
        class_id=args.class_id,
        num_input_points=num_input_points,
        num_complete_points=num_complete_points,
        normalize_pair=False,
        include_all_views=True,
    )

    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset.samples)), k=min(args.num_samples, len(dataset.samples)))
    selected = [dataset.samples[i] for i in indices]
    selected.sort(key=lambda item: (str(item['scan_id']), str(item['view_id'])))

    device = torch.device('cuda')
    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    transport, decoder, normalizer, decoder_source = load_transport_and_decoder(
        args.stage2_ckpt,
        args.stage1_ckpt,
        device,
        output_points=num_complete_points,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'seed': args.seed,
        'split': args.split,
        'class_id': args.class_id,
        'num_input_points': num_input_points,
        'num_complete_points': num_complete_points,
        'stage2_ckpt': str(args.stage2_ckpt),
        'stage1_ckpt': str(args.stage1_ckpt),
        'decoder_source': decoder_source,
        'samples': [],
    }

    with torch.no_grad():
        for rank, sample in enumerate(selected, start=1):
            partial_raw = dataset._load_pcd_points(Path(sample['partial_path']))
            complete_raw = dataset._load_pcd_points(Path(sample['complete_path']))
            partial = sample_fixed_size(partial_raw, num_input_points)
            complete = sample_fixed_size(complete_raw, num_complete_points)
            partial_norm, complete_norm = normalize_pair(partial, complete)

            enc = encoder(partial_norm.unsqueeze(0).to(device))
            src_tokens = normalizer.normalize(enc.tokens) if normalizer is not None else enc.tokens
            pred_tokens = transport.transport(src_tokens, enc.centers, num_steps=args.transport_steps)
            pred_tokens = normalizer.denormalize(pred_tokens) if normalizer is not None else pred_tokens
            pred_complete = decoder(pred_tokens, enc.centers).coarse_points[0].cpu()

            sample_dir = args.output_dir / f'{rank:02d}_{sample["scan_id"]}_view{sample["view_id"]}'
            write_ascii_pcd(sample_dir / 'partial.pcd', partial_norm)
            write_ascii_pcd(sample_dir / 'pred_complete.pcd', pred_complete)
            write_ascii_pcd(sample_dir / 'gt_complete.pcd', complete_norm)

            summary['samples'].append({
                'rank': rank,
                'scan_id': str(sample['scan_id']),
                'view_id': str(sample['view_id']),
                'partial_path': str(sample['partial_path']),
                'complete_path': str(sample['complete_path']),
                'output_dir': str(sample_dir),
            })

    (args.output_dir / 'metadata.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({'output_dir': str(args.output_dir), 'num_samples': len(summary['samples']), 'samples': summary['samples']}, indent=2))


if __name__ == '__main__':
    main()
