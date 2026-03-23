from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 training')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/stage1'))
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=50)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--val-ratio', type=float, default=0.01)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    return parser.parse_args()


def save_checkpoint(path: Path, epoch: int, step: int, decoder: QueryCompletionDecoder, optimizer: torch.optim.Optimizer, args: argparse.Namespace, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'epoch': epoch,
            'step': step,
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'metrics': metrics,
        },
        path,
    )


def build_train_val_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
    dataset = ShapeNetPointCloudDataset(
        data_root=args.data_root,
        split='train',
        class_id=args.class_id,
        num_input_points=2048,
        num_complete_points=8192,
    )
    num_samples = len(dataset)
    num_val = max(1, int(num_samples * args.val_ratio))
    num_train = num_samples - num_val
    generator = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(num_samples, generator=generator).tolist()
    train_indices = perm[:num_train]
    val_indices = perm[num_train:]

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=shapenet_point_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=shapenet_point_collate_fn,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader, num_train, num_val


def evaluate(
    encoder: PointMAEEncoder,
    decoder: QueryCompletionDecoder,
    loader: DataLoader,
    device: torch.device,
    iou_resolution: int,
    metric_points: int | None,
    max_batches: int | None,
) -> dict[str, float]:
    decoder.eval()
    totals = {
        'val_loss': 0.0,
        'val_chamfer_distance': 0.0,
        'val_chamfer_distance_l1': 0.0,
        'val_emd': 0.0,
        'val_iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            enc = encoder(partial_points)
            pred = decoder(enc.tokens, enc.centers).coarse_points
            loss = chamfer_distance_l2(pred, complete_points)
            metrics = compute_completion_metrics(pred, complete_points, iou_resolution=iou_resolution, metric_points=metric_points)
            totals['val_loss'] += float(loss.item())
            totals['val_chamfer_distance'] += metrics['chamfer_distance']
            totals['val_chamfer_distance_l1'] += metrics['chamfer_distance_l1']
            totals['val_emd'] += metrics['emd']
            totals['val_iou'] += metrics['iou']
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    if num_batches == 0:
        return totals
    return {key: value / num_batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage1_train.py')

    device = torch.device('cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, num_train, num_val = build_train_val_loaders(args)

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    meta = {
        'train_size': num_train,
        'val_size': num_val,
        'batch_size': args.batch_size,
        'train_steps_per_epoch': len(train_loader),
        'val_steps_per_epoch': len(val_loader),
        'class_id': args.class_id or 'all',
        'amp': args.amp,
        'epochs': args.epochs,
    }
    print(json.dumps(meta), flush=True)

    global_step = 0
    best_val_loss = float('inf')
    epoch_bar = tqdm(range(args.epochs), desc='Epochs', dynamic_ncols=True, file=sys.stdout)
    for epoch in epoch_bar:
        decoder.train()
        epoch_start = time.time()
        running_loss = 0.0
        steps_this_epoch = 0
        torch.cuda.reset_peak_memory_stats(device)
        tqdm.write(f'[stage1] epoch {epoch + 1}/{args.epochs} start', file=sys.stdout)

        train_iter = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{args.epochs}',
            dynamic_ncols=True,
            leave=False,
            file=sys.stdout,
        )
        for batch_idx, batch in train_iter:
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)

            with torch.no_grad():
                enc = encoder(partial_points)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                dec = decoder(enc.tokens, enc.centers)
                loss = chamfer_distance_l2(dec.coarse_points, complete_points)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            steps_this_epoch += 1
            global_step += 1

            train_iter.set_postfix({
                'loss': f'{float(loss.item()):.4f}',
                'step': global_step,
            })

            if batch_idx % args.log_every == 0:
                max_mem = torch.cuda.max_memory_allocated(device=device) / 1024**3
                print(json.dumps({
                    'phase': 'train',
                    'epoch': epoch,
                    'step': global_step,
                    'batch_idx': batch_idx,
                    'loss': float(loss.item()),
                    'max_mem_gb': round(max_mem, 3),
                }), flush=True)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        train_loss = running_loss / max(steps_this_epoch, 1)
        val_metrics = evaluate(
            encoder=encoder,
            decoder=decoder,
            loader=val_loader,
            device=device,
            iou_resolution=args.iou_resolution,
            metric_points=args.metric_points,
            max_batches=args.max_val_batches,
        )
        epoch_time = time.time() - epoch_start
        summary = {
            'epoch': epoch,
            'train_loss': train_loss,
            **val_metrics,
            'epoch_time_sec': round(epoch_time, 2),
        }
        print(json.dumps(summary), flush=True)
        epoch_bar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'val_loss': f"{val_metrics['val_loss']:.4f}",
            'val_emd': f"{val_metrics['val_emd']:.4f}",
            'val_iou': f"{val_metrics['val_iou']:.4f}",
        })

        ckpt_path = args.save_dir / f'epoch_{epoch:03d}.pth'
        save_checkpoint(ckpt_path, epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            save_checkpoint(args.save_dir / 'best.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)
        save_checkpoint(args.save_dir / 'last.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
