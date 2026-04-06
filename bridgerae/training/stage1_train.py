from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bridgerae.datasets import build_pointcloud_dataset, resolve_point_counts, shapenet_point_collate_fn
from bridgerae.models import LatentNormalizer, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2, get_chamfer_backend, repulsion_loss
from bridgerae.training.metrics import compute_completion_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-1 training')
    parser.add_argument('--dataset', type=str, default='shapenet', choices=['shapenet', 'pcn'])
    parser.add_argument('--data-root', type=Path, default=None)
    parser.add_argument('--pair-data-root', type=Path, default=None)
    parser.add_argument('--num-input-points', type=int, default=None)
    parser.add_argument('--num-complete-points', type=int, default=None)
    parser.add_argument('--train-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--val-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=384)
    parser.add_argument('--val-batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--train-shuffle', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/stage1'))
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--resume-ckpt', type=Path, default=None)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=50)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--iou-resolution', type=int, default=32)
    parser.add_argument('--metric-points', type=int, default=2048)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--latent-normalize', action='store_true', help='Enable channel-wise latent normalization')
    parser.add_argument('--latent-noise-std', type=float, default=0.1, help='Noise std added to normalized latent during training (0 = disabled)')
    parser.add_argument('--latent-stats-batches', type=int, default=None, help='Max batches for stats collection (None = full pass)')
    parser.add_argument('--latent-stats-cache-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/latent_stats_cache'))
    parser.add_argument('--repulsion-weight', type=float, default=0.0, help='Weight for repulsion loss (0 = disabled)')
    parser.add_argument('--repulsion-k', type=int, default=8, help='Number of nearest neighbors for repulsion loss')
    parser.add_argument('--lr-scheduler', type=str, default='cosine', choices=['none', 'cosine'], help='LR scheduler type')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Linear warmup epochs before cosine decay')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum LR for cosine scheduler')
    parser.add_argument('--save-epoch-checkpoints', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False, help='Use torch.compile for encoder/decoder')
    return parser.parse_args()


def _slugify(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value)


def build_latent_stats_cache_path(args: argparse.Namespace) -> Path:
    class_id = args.class_id or 'all'
    encoder_name = _slugify(args.encoder_ckpt.stem)
    split_set = _slugify(args.train_split_set)
    return args.latent_stats_cache_dir / (
        f'{args.dataset}_class-{class_id}_split-{split_set}_'
        f'complete-{args.num_complete_points}_encoder-{encoder_name}.pth'
    )


def try_load_latent_stats_cache(path: Path, normalizer: LatentNormalizer) -> bool:
    if not path.exists():
        return False
    payload = torch.load(path, map_location='cpu')
    state_dict = payload.get('normalizer') if isinstance(payload, dict) else None
    if not isinstance(state_dict, dict):
        return False
    normalizer.load_state_dict(state_dict)
    return normalizer.is_collected


def save_latent_stats_cache(path: Path, args: argparse.Namespace, normalizer: LatentNormalizer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'normalizer': normalizer.state_dict(),
        'meta': {
            'dataset': args.dataset,
            'data_root': str(args.data_root) if args.data_root is not None else None,
            'class_id': args.class_id or 'all',
            'train_split_set': args.train_split_set,
            'num_complete_points': args.num_complete_points,
            'encoder_ckpt': str(args.encoder_ckpt),
        },
    }
    torch.save(payload, path)


def save_checkpoint(path: Path, epoch: int, step: int, decoder: QueryCompletionDecoder, optimizer: torch.optim.Optimizer, args: argparse.Namespace, metrics: dict[str, float], normalizer: LatentNormalizer | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'epoch': epoch,
        'step': step,
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'args': vars(args),
        'metrics': metrics,
    }
    if normalizer is not None:
        data['normalizer'] = normalizer.state_dict()
    torch.save(data, path)


def load_resume_checkpoint(
    checkpoint_path: Path,
    decoder: QueryCompletionDecoder,
    optimizer: torch.optim.Optimizer,
    normalizer: LatentNormalizer | None = None,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder.load_state_dict(checkpoint['decoder'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if normalizer is not None and 'normalizer' in checkpoint:
        normalizer.load_state_dict(checkpoint['normalizer'])
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    global_step = int(checkpoint.get('step', 0))
    metrics = checkpoint.get('metrics', {}) or {}
    best_val_loss = float(metrics.get('val_loss', float('inf')))
    return start_epoch, global_step, best_val_loss


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


def evaluate(
    encoder: PointMAEEncoder,
    decoder: QueryCompletionDecoder,
    loader: DataLoader,
    device: torch.device,
    iou_resolution: int,
    metric_points: int | None,
    max_batches: int | None,
    normalizer: LatentNormalizer | None = None,
) -> dict[str, float]:
    decoder.eval()
    totals = {
        'val_loss': 0.0,
        'val_chamfer_distance': 0.0,
        'val_chamfer_distance_l1': 0.0,
        'val_emd': 0.0,
        'val_f1_1pct': 0.0,
        'val_iou': 0.0,
    }
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            enc = encoder(complete_points)
            tokens = enc.tokens
            if normalizer is not None:
                tokens = normalizer.normalize(tokens)
            pred = decoder(tokens, enc.centers).coarse_points
            loss = chamfer_distance_l2(pred, complete_points)
            metrics = compute_completion_metrics(pred, complete_points, iou_resolution=iou_resolution, metric_points=metric_points)
            totals['val_loss'] += float(loss.item())
            totals['val_chamfer_distance'] += metrics['chamfer_distance']
            totals['val_chamfer_distance_l1'] += metrics['chamfer_distance_l1']
            totals['val_emd'] += metrics['emd']
            totals['val_f1_1pct'] += metrics['f1_1pct']
            totals['val_iou'] += metrics['iou']
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

    if num_batches == 0:
        return totals
    return {key: value / num_batches for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.num_input_points, args.num_complete_points = resolve_point_counts(
        args.dataset,
        args.num_input_points,
        args.num_complete_points,
    )
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage1_train.py')

    device = torch.device('cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_loader = build_loader(
        args,
        split='train',
        split_set=args.train_split_set,
        shuffle=args.train_shuffle,
        batch_size=args.batch_size,
        max_samples=args.max_train_samples,
    )
    val_batch_size = args.val_batch_size or args.batch_size
    val_loader = build_loader(
        args,
        split='val',
        split_set=args.val_split_set,
        shuffle=False,
        batch_size=val_batch_size,
        max_samples=args.max_val_samples,
    )

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = QueryCompletionDecoder(
        hidden_dim=384,
        num_queries=256,
        num_heads=6,
        depth=6,
        output_points=args.num_complete_points,
    ).to(device)
    normalizer: LatentNormalizer | None = None
    if args.latent_normalize:
        normalizer = LatentNormalizer(dim=384).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    if args.resume_ckpt is not None:
        start_epoch, global_step, best_val_loss = load_resume_checkpoint(args.resume_ckpt, decoder, optimizer, normalizer)

    if args.compile:
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)

    if normalizer is not None and not normalizer.is_collected:
        cache_path = build_latent_stats_cache_path(args)
        if try_load_latent_stats_cache(cache_path, normalizer):
            print(json.dumps({
                'latent_stats': 'loaded_from_cache',
                'cache_path': str(cache_path),
                'mean_norm': float(normalizer.mean.norm().item()),
                'std_mean': float(normalizer.std.mean().item()),
            }), flush=True)
        else:
            stats_loader = build_loader(
                args, split='train', split_set=args.train_split_set,
                shuffle=False, batch_size=args.batch_size, max_samples=args.max_train_samples,
            )
            normalizer.collect_stats(encoder, stats_loader, device, max_batches=args.latent_stats_batches)
            del stats_loader
            save_latent_stats_cache(cache_path, args, normalizer)
            print(json.dumps({
                'latent_stats': 'collected',
                'cache_path': str(cache_path),
                'mean_norm': float(normalizer.mean.norm().item()),
                'std_mean': float(normalizer.std.mean().item()),
            }), flush=True)

    total_target_epoch = start_epoch + args.epochs

    scheduler = None
    if args.lr_scheduler == 'cosine':
        def lr_lambda(epoch_idx: int) -> float:
            if epoch_idx < args.warmup_epochs:
                return max((epoch_idx + 1) / max(args.warmup_epochs, 1), args.lr_min / args.lr)
            progress = (epoch_idx - args.warmup_epochs) / max(total_target_epoch - args.warmup_epochs, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(cosine, args.lr_min / args.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=start_epoch - 1 if start_epoch > 0 else -1)

    meta = {
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'batch_size': args.batch_size,
        'val_batch_size': val_batch_size,
        'train_steps_per_epoch': len(train_loader),
        'val_steps_per_epoch': len(val_loader),
        'class_id': args.class_id or 'all',
        'dataset': args.dataset,
        'num_input_points': args.num_input_points,
        'num_complete_points': args.num_complete_points,
        'train_split_set': args.train_split_set,
        'val_split_set': args.val_split_set,
        'max_train_samples': args.max_train_samples,
        'max_val_samples': args.max_val_samples,
        'train_shuffle': args.train_shuffle,
        'amp': args.amp,
        'epochs': args.epochs,
        'start_epoch': start_epoch,
        'target_epoch': total_target_epoch,
        'resume_ckpt': str(args.resume_ckpt) if args.resume_ckpt is not None else None,
        'chamfer_backend': get_chamfer_backend(device),
        'latent_normalize': args.latent_normalize,
        'latent_noise_std': args.latent_noise_std,
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
            complete_points = batch['complete_points'].to(device, non_blocking=True)

            with torch.no_grad():
                enc = encoder(complete_points)
                tokens = enc.tokens
                if normalizer is not None:
                    tokens = normalizer.normalize(tokens)
                if args.latent_noise_std > 0:
                    tokens = tokens + args.latent_noise_std * torch.randn_like(tokens)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                dec = decoder(tokens, enc.centers)
                cd_loss = chamfer_distance_l2(dec.coarse_points, complete_points)
                loss = cd_loss
                if args.repulsion_weight > 0:
                    rep_loss = repulsion_loss(dec.coarse_points, k=args.repulsion_k)
                    loss = loss + args.repulsion_weight * rep_loss

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
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step()
        should_eval = ((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == total_target_epoch)
        epoch_time = time.time() - epoch_start
        summary = {
            'epoch': epoch,
            'train_loss': train_loss,
            'lr': current_lr,
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
                normalizer=normalizer,
            )
            summary.update(val_metrics)
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f"{val_metrics['val_loss']:.4f}",
                'val_emd': f"{val_metrics['val_emd']:.4f}",
                'val_f1@1%': f"{val_metrics['val_f1_1pct']:.2f}",
                'val_iou': f"{val_metrics['val_iou']:.4f}",
            })
        else:
            epoch_bar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'eval': f'every_{args.eval_every}',
            })

        print(json.dumps(summary), flush=True)

        if args.save_epoch_checkpoints:
            ckpt_path = args.save_dir / f'epoch_{epoch:03d}.pth'
            save_checkpoint(ckpt_path, epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary, normalizer=normalizer)
        if should_eval and summary['val_loss'] < best_val_loss:
            best_val_loss = summary['val_loss']
            save_checkpoint(args.save_dir / 'best.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary, normalizer=normalizer)
        save_checkpoint(args.save_dir / 'last.pth', epoch=epoch, step=global_step, decoder=decoder, optimizer=optimizer, args=args, metrics=summary, normalizer=normalizer)

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
