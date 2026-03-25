from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import LatentTransportModel, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2, get_chamfer_backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-2 latent transport training')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55'))
    parser.add_argument('--pair-data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--train-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--val-split-set', type=str, default='ShapeNet-34')
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--stage1-ckpt', type=Path, required=True)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--resume-ckpt', type=Path, default=None)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/stage2'))
    parser.add_argument('--amp', action='store_true')

    parser.add_argument('--latent-noise-std', type=float, default=0.01)
    parser.add_argument('--latent-noise-start', type=float, default=0.0)
    parser.add_argument('--latent-noise-warmup-epochs', type=int, default=0)

    parser.add_argument('--flow-weight', type=float, default=1.0)
    parser.add_argument('--flow-weight-start', type=float, default=0.0)
    parser.add_argument('--flow-warmup-epochs', type=int, default=0)
    parser.add_argument('--recon-weight', type=float, default=1.0)

    parser.add_argument('--transport-steps', type=int, default=8)

    parser.add_argument('--decoder-train-mode', type=str, default='none', choices=['none', 'last_n', 'all'])
    parser.add_argument('--decoder-train-last-n', type=int, default=2)
    parser.add_argument('--decoder-lr-mult', type=float, default=0.1)

    parser.add_argument('--best-metric', type=str, default='val_recon_loss', choices=['val_recon_loss', 'val_total_loss'])

    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    parser.add_argument('--eval-every', type=int, default=5)
    return parser.parse_args()


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


def load_frozen_decoder(checkpoint_path: Path, device: torch.device) -> QueryCompletionDecoder:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


def configure_decoder_training(
    decoder: QueryCompletionDecoder,
    train_mode: str,
    train_last_n: int,
) -> list[torch.nn.Parameter]:
    for p in decoder.parameters():
        p.requires_grad = False
    decoder.eval()

    if train_mode == 'none':
        return []

    if train_mode == 'all':
        decoder.train()
        for p in decoder.parameters():
            p.requires_grad = True
        return [p for p in decoder.parameters() if p.requires_grad]

    num_blocks = len(decoder.blocks)
    block_count = min(max(train_last_n, 1), num_blocks)
    trainable_blocks = list(decoder.blocks[num_blocks - block_count:])
    for block in trainable_blocks:
        block.train()
        for p in block.parameters():
            p.requires_grad = True

    decoder.norm.train()
    for p in decoder.norm.parameters():
        p.requires_grad = True

    decoder.point_head.train()
    for p in decoder.point_head.parameters():
        p.requires_grad = True

    decoder.query_tokens.requires_grad = True
    decoder.query_pos.requires_grad = True
    return [p for p in decoder.parameters() if p.requires_grad]


def load_transport_resume_checkpoint(
    checkpoint_path: Path,
    decoder: QueryCompletionDecoder,
    transport: LatentTransportModel,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    transport.load_state_dict(checkpoint['transport'])
    decoder_state = checkpoint.get('decoder')
    if decoder_state is not None:
        decoder.load_state_dict(decoder_state)
    try:
        optimizer.load_state_dict(checkpoint['optimizer'])
    except ValueError as exc:
        print(json.dumps({
            'warning': 'optimizer_state_skipped',
            'reason': str(exc),
        }), flush=True)
    start_epoch = int(checkpoint.get('epoch', -1)) + 1
    global_step = int(checkpoint.get('step', 0))
    metrics = checkpoint.get('metrics', {}) or {}
    best_val = float(metrics.get('val_total_loss', float('inf')))
    return start_epoch, global_step, best_val


def sample_bridge_state(source_tokens: torch.Tensor, target_tokens: torch.Tensor, noise_std: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = source_tokens.shape[0]
    t = torch.rand(batch_size, device=source_tokens.device, dtype=source_tokens.dtype)
    interp = (1.0 - t)[:, None, None] * source_tokens + t[:, None, None] * target_tokens
    if noise_std > 0:
        noise_scale = noise_std * torch.sqrt((t * (1.0 - t)).clamp_min(1e-4))[:, None, None]
        interp = interp + noise_scale * torch.randn_like(interp)
    velocity_target = target_tokens - source_tokens
    return interp, velocity_target, t


def evaluate(
    encoder: PointMAEEncoder,
    decoder: QueryCompletionDecoder,
    transport: LatentTransportModel,
    loader: DataLoader,
    device: torch.device,
    transport_steps: int,
    max_batches: int | None,
    flow_weight: float,
    recon_weight: float,
) -> dict[str, float]:
    transport.eval()
    totals = {'val_total_loss': 0.0, 'val_flow_loss': 0.0, 'val_recon_loss': 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            src = encoder(partial_points)
            tgt = encoder(complete_points, centers=src.centers)
            bridge_state, velocity_target, t = sample_bridge_state(src.tokens, tgt.tokens, noise_std=0.0)
            out = transport(bridge_state, src.tokens, src.centers, t)
            flow_loss = F.mse_loss(out.velocity, velocity_target)
            pred_tokens = transport.transport(src.tokens, src.centers, num_steps=transport_steps)
            pred_points = decoder(pred_tokens, src.centers).coarse_points
            recon_loss = chamfer_distance_l2(pred_points, complete_points)
            totals['val_flow_loss'] += float(flow_loss.item())
            totals['val_recon_loss'] += float(recon_loss.item())
            totals['val_total_loss'] += float((flow_weight * flow_loss + recon_weight * recon_loss).item())
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break
    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def linear_warmup_value(epoch_idx: int, start: float, end: float, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return end
    progress = min(max((epoch_idx + 1) / float(warmup_epochs), 0.0), 1.0)
    return start + (end - start) * progress


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage2_train.py')
    device = torch.device('cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_loader = build_loader(args, split='train', split_set=args.train_split_set, shuffle=True, batch_size=args.batch_size)
    val_loader = build_loader(args, split='val', split_set=args.val_split_set, shuffle=False, batch_size=args.batch_size)

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = load_frozen_decoder(args.stage1_ckpt, device)
    decoder_trainable_params = configure_decoder_training(decoder, args.decoder_train_mode, args.decoder_train_last_n)
    transport = LatentTransportModel(hidden_dim=384, depth=6, num_heads=6).to(device)

    optimizer_param_groups: list[dict[str, object]] = [
        {'params': list(transport.parameters()), 'lr': args.lr},
    ]
    if decoder_trainable_params:
        optimizer_param_groups.append({'params': decoder_trainable_params, 'lr': args.lr * args.decoder_lr_mult})
    optimizer = torch.optim.AdamW(optimizer_param_groups, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    global_step = 0
    best_val = float('inf')
    if args.resume_ckpt is not None:
        start_epoch, global_step, best_val = load_transport_resume_checkpoint(args.resume_ckpt, decoder, transport, optimizer)

    total_target_epoch = start_epoch + args.epochs
    print(json.dumps({
        'train_size': len(train_loader.dataset),
        'val_size': len(val_loader.dataset),
        'batch_size': args.batch_size,
        'train_steps_per_epoch': len(train_loader),
        'val_steps_per_epoch': len(val_loader),
        'epochs': args.epochs,
        'start_epoch': start_epoch,
        'target_epoch': total_target_epoch,
        'class_id': args.class_id or 'all',
        'train_split_set': args.train_split_set,
        'val_split_set': args.val_split_set,
        'stage1_ckpt': str(args.stage1_ckpt),
        'resume_ckpt': str(args.resume_ckpt) if args.resume_ckpt is not None else None,
        'decoder_train_mode': args.decoder_train_mode,
        'decoder_train_last_n': args.decoder_train_last_n,
        'decoder_trainable_params': sum(p.numel() for p in decoder_trainable_params),
        'decoder_lr': args.lr * args.decoder_lr_mult if decoder_trainable_params else 0.0,
        'best_metric': args.best_metric,
        'chamfer_backend': get_chamfer_backend(device),
    }), flush=True)

    epoch_bar = tqdm(range(start_epoch, total_target_epoch), desc='Stage2', dynamic_ncols=True)
    for epoch in epoch_bar:
        transport.train()
        if args.decoder_train_mode == 'all':
            decoder.train()

        epoch_start = time.time()
        running_total = 0.0
        running_flow = 0.0
        running_recon = 0.0
        steps = 0

        flow_weight = linear_warmup_value(epoch, args.flow_weight_start, args.flow_weight, args.flow_warmup_epochs)
        noise_std = linear_warmup_value(epoch, args.latent_noise_start, args.latent_noise_std, args.latent_noise_warmup_epochs)

        torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, batch in enumerate(train_loader):
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            with torch.no_grad():
                src = encoder(partial_points)
                tgt = encoder(complete_points, centers=src.centers)
                bridge_state, velocity_target, t = sample_bridge_state(src.tokens, tgt.tokens, noise_std=noise_std)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                out = transport(bridge_state, src.tokens, src.centers, t)
                flow_loss = F.mse_loss(out.velocity, velocity_target)
                pred_tokens = transport.transport(src.tokens, src.centers, num_steps=args.transport_steps)
                pred_points = decoder(pred_tokens, src.centers).coarse_points
                recon_loss = chamfer_distance_l2(pred_points, complete_points)
                loss = flow_weight * flow_loss + args.recon_weight * recon_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total += float(loss.item())
            running_flow += float(flow_loss.item())
            running_recon += float(recon_loss.item())
            steps += 1
            global_step += 1

            if batch_idx % args.log_every == 0:
                print(json.dumps({
                    'phase': 'train',
                    'epoch': epoch,
                    'step': global_step,
                    'batch_idx': batch_idx,
                    'loss': float(loss.item()),
                    'flow_loss': float(flow_loss.item()),
                    'recon_loss': float(recon_loss.item()),
                    'flow_weight': flow_weight,
                    'noise_std': noise_std,
                    'max_mem_gb': round(torch.cuda.max_memory_allocated(device=device) / 1024**3, 3),
                }), flush=True)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        should_eval = ((epoch + 1) % args.eval_every == 0) or ((epoch + 1) == total_target_epoch)
        summary = {
            'epoch': epoch,
            'train_total_loss': running_total / max(steps, 1),
            'train_flow_loss': running_flow / max(steps, 1),
            'train_recon_loss': running_recon / max(steps, 1),
            'flow_weight': flow_weight,
            'noise_std': noise_std,
            'epoch_time_sec': round(time.time() - epoch_start, 2),
            'eval_ran': should_eval,
        }
        if should_eval:
            val_metrics = evaluate(
                encoder,
                decoder,
                transport,
                val_loader,
                device,
                args.transport_steps,
                args.max_val_batches,
                flow_weight=flow_weight,
                recon_weight=args.recon_weight,
            )
            summary.update(val_metrics)
            epoch_bar.set_postfix({
                'train_total': f"{summary['train_total_loss']:.4f}",
                'val_total': f"{summary['val_total_loss']:.4f}",
                'val_recon': f"{summary['val_recon_loss']:.4f}",
            })
        else:
            epoch_bar.set_postfix({
                'train_total': f"{summary['train_total_loss']:.4f}",
                'eval': f'every_{args.eval_every}',
            })

        print(json.dumps(summary), flush=True)

        ckpt = {
            'epoch': epoch,
            'step': global_step,
            'transport': transport.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'metrics': summary,
        }
        if decoder_trainable_params:
            ckpt['decoder'] = decoder.state_dict()

        torch.save(ckpt, args.save_dir / f'epoch_{epoch:03d}.pth')
        torch.save(ckpt, args.save_dir / 'last.pth')
        if should_eval and summary[args.best_metric] < best_val:
            best_val = summary[args.best_metric]
            torch.save(ckpt, args.save_dir / 'best.pth')

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
