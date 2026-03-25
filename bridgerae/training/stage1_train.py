from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2, get_chamfer_backend
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 training')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55'))
    parser.add_argument('--pair-data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--train-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--val-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/stage1'))
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--resume-ckpt', type=Path, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=50)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    parser.add_argument('--eval-every', type=int, default=5)
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


def load_resume_checkpoint(
    checkpoint_path: Path,
    decoder: QueryCompletionDecoder,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    global_step = int(checkpoint.get('step', 0))
    metrics = checkpoint.get('metrics', {}) or {}
    best_val_loss = float(metrics.get('val_loss', float('inf')))
    return start_epoch, global_step, best_val_loss


def build_loader(args: argparse.Namespace, split: str, split_set: str, shuffle: bool, batch_size: int) -> DataLoader:
    dataset = ShapeNetPointCloudDataset(
        data_root=args.data_root,
        pair_data_root=args.pair_data_root,
        split=split,
        split_set=split_set,
        class_id=args.class_id,
        num_input_points=2048,
        num_complete_points=8192,
    )
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
    train_loader = build_loader(args, split='train', split_set=args.train_split_set, shuffle=True, batch_size=args.batch_size)
    val_loader = build_loader(args, split='val', split_set=args.val_split_set, shuffle=False, batch_size=args.batch_size)

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if args.resume_ckpt is not None:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(args.resume_ckpt, decoder, optimizer)

    total_target_epoch = start_epoch + args.epochs
    meta = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'batch_size': args.batch_size,
        'train_steps_per_epoch': len(train_loader),
        'val_steps_per_epoch': len(val_loader),
        'class_id': args.class_id or 'all',
        'train_split_set': args.train_split_set,
        'val_split_set': args.val_split_set,
        'amp': args.amp,
        'epochs': args.epochs,
        'start_epoch': start_epoch,
        'target_epoch': total_target_epoch,
        'resume_ckpt': str(args.resume_ckpt) if args.resume_ckpt is not None else None,
        'chamfer_backend': get_chamfer_backend(device),
    }
    print(json.dumps(meta), flush=True)

    epoch_bar = tqdm(range(start_epoch, total_target_epoch), desc='Epochs', dynamic_ncols=True, file=sys.stdout)
    for epoch in epoch_bar:
        decoder.train()
        epoch_start = time.time()
        running_loss = 0.0
        steps_this_epoch = 0
        torch.cuda.reset_peak_memory_stats(device)
        tqdm.write(f'[stage1] epoch {epoch + 1}/{total_target_epoch} start', file=sys.stdout)

        train_iter = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f'Epoch {epoch + 1}/{total_target_epoch}',
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
        should_eval = ((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == total_target_epoch)
        epoch_time = time.time() - epoch_start
        summary = {
            'epoch': epoch,
            'train_loss': train_loss,
            'epoch_time_sec': round(epoch_time, 2),
            'eval_ran': should_eval,
        }
        if should_eval:
            val_metrics = evaluate(
                encoder=encoder,
                decoder=decoder,
                loader=val_loader,
                device=device,
                iou_resolution=args.iou_resolution,
                metric_points=args.metric_points,
                max_batches=args.max_val_batches,
            )
            summary.update(val_metrics)
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f"{val_metrics['val_loss']:.4f}",
                'val_emd': f"{val_metrics['val_emd']:.4f}",
                'val_iou': f"{val_metrics['val_iou']:.4f}",
            })
        else:
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'eval': f'every_{args.eval_every}',
            })

        print(json.dumps(summary), flush=True)

        ckpt_path = args.save_dir / f'epoch_{epoch:03d}.pth'
        save_checkpoint(ckpt_path, epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)
        if should_eval and summary['val_loss'] < best_val_loss:
            best_val_loss = summary['val_loss']
            save_checkpoint(args.save_dir / 'best.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)
        save_checkpoint(args.save_dir / 'last.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary)

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
