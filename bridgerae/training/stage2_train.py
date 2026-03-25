from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from bridgerae.datasets import ShapeNetPointCloudDataset, shapenet_point_collate_fn
from bridgerae.models import LatentDiffusionDiT, PointMAEEncoder, QueryCompletionDecoder
from bridgerae.training.losses import chamfer_distance_l2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BridgeRAE stage-2 latent diffusion DiT training')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'))
    parser.add_argument('--class-id', type=str, default=None)
    parser.add_argument('--stage1-ckpt', type=Path, required=True)
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-2)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--save-dir', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/stage2_dit'))
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--split-seed', type=int, default=42)
    parser.add_argument('--diffusion-steps', type=int, default=1000)
    parser.add_argument('--sample-steps', type=int, default=100)
    parser.add_argument('--noise-weight', type=float, default=1.0)
    parser.add_argument('--recon-weight', type=float, default=1.0)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-val-batches', type=int, default=None)
    return parser.parse_args()


def build_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
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
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=shapenet_point_collate_fn, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=shapenet_point_collate_fn, pin_memory=True, drop_last=False, persistent_workers=args.num_workers > 0)
    return train_loader, val_loader, num_train, num_val


def load_frozen_decoder(checkpoint_path: Path, device: torch.device) -> QueryCompletionDecoder:
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    decoder = QueryCompletionDecoder(hidden_dim=384, num_queries=256, num_heads=6, depth=6, output_points=8192).to(device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False
    return decoder


def evaluate(
    encoder: PointMAEEncoder,
    decoder: QueryCompletionDecoder,
    dit: LatentDiffusionDiT,
    loader: DataLoader,
    device: torch.device,
    sample_steps: int,
    max_batches: int | None,
) -> dict[str, float]:
    dit.eval()
    totals = {'val_total_loss': 0.0, 'val_noise_loss': 0.0, 'val_recon_loss': 0.0}
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            src = encoder(partial_points)
            tgt = encoder(complete_points, centers=src.centers)
            t = torch.randint(0, dit.diffusion_steps, (src.tokens.shape[0],), device=device, dtype=torch.long)
            noisy_tokens, noise = dit.q_sample(tgt.tokens, t)
            out = dit(noisy_tokens, src.tokens, src.centers, t)
            noise_loss = F.mse_loss(out.noise_pred, noise)
            pred_tokens = dit.sample(src.tokens, src.centers, num_steps=sample_steps)
            pred_points = decoder(pred_tokens, src.centers).coarse_points
            recon_loss = chamfer_distance_l2(pred_points, complete_points)
            totals['val_noise_loss'] += float(noise_loss.item())
            totals['val_recon_loss'] += float(recon_loss.item())
            totals['val_total_loss'] += float((noise_loss + recon_loss).item())
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break
    return {key: value / max(num_batches, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for stage2_train.py')
    device = torch.device('cuda')
    args.save_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, num_train, num_val = build_loaders(args)

    encoder = PointMAEEncoder(pretrained_ckpt=str(args.encoder_ckpt), freeze=True).to(device)
    decoder = load_frozen_decoder(args.stage1_ckpt, device)
    dit = LatentDiffusionDiT(hidden_dim=384, depth=8, num_heads=6, diffusion_steps=args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    print(json.dumps({
        'train_size': num_train,
        'val_size': num_val,
        'batch_size': args.batch_size,
        'train_steps_per_epoch': len(train_loader),
        'val_steps_per_epoch': len(val_loader),
        'epochs': args.epochs,
        'class_id': args.class_id or 'all',
        'stage1_ckpt': str(args.stage1_ckpt),
        'diffusion_steps': args.diffusion_steps,
        'sample_steps': args.sample_steps,
    }), flush=True)

    global_step = 0
    best_val = float('inf')
    epoch_bar = tqdm(range(args.epochs), desc='Stage2-DiT', dynamic_ncols=True)
    for epoch in epoch_bar:
        dit.train()
        epoch_start = time.time()
        running_total = 0.0
        running_noise = 0.0
        running_recon = 0.0
        steps = 0
        torch.cuda.reset_peak_memory_stats(device)

        for batch_idx, batch in enumerate(train_loader):
            partial_points = batch['partial_points'].to(device, non_blocking=True)
            complete_points = batch['complete_points'].to(device, non_blocking=True)
            with torch.no_grad():
                src = encoder(partial_points)
                tgt = encoder(complete_points, centers=src.centers)
                t = torch.randint(0, dit.diffusion_steps, (src.tokens.shape[0],), device=device, dtype=torch.long)
                noisy_tokens, noise = dit.q_sample(tgt.tokens, t)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=args.amp):
                out = dit(noisy_tokens, src.tokens, src.centers, t)
                noise_loss = F.mse_loss(out.noise_pred, noise)
                pred_points = decoder(out.x0_pred, src.centers).coarse_points
                recon_loss = chamfer_distance_l2(pred_points, complete_points)
                loss = args.noise_weight * noise_loss + args.recon_weight * recon_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total += float(loss.item())
            running_noise += float(noise_loss.item())
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
                    'noise_loss': float(noise_loss.item()),
                    'recon_loss': float(recon_loss.item()),
                    'max_mem_gb': round(torch.cuda.max_memory_allocated(device=device) / 1024**3, 3),
                }), flush=True)

            if args.max_steps is not None and global_step >= args.max_steps:
                break

        val_metrics = evaluate(encoder, decoder, dit, val_loader, device, args.sample_steps, args.max_val_batches)
        summary = {
            'epoch': epoch,
            'train_total_loss': running_total / max(steps, 1),
            'train_noise_loss': running_noise / max(steps, 1),
            'train_recon_loss': running_recon / max(steps, 1),
            **val_metrics,
            'epoch_time_sec': round(time.time() - epoch_start, 2),
        }
        print(json.dumps(summary), flush=True)
        epoch_bar.set_postfix({
            'train_total': f"{summary['train_total_loss']:.4f}",
            'val_total': f"{summary['val_total_loss']:.4f}",
            'val_recon': f"{summary['val_recon_loss']:.4f}",
        })

        ckpt = {
            'epoch': epoch,
            'step': global_step,
            'dit': dit.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'metrics': summary,
        }
        torch.save(ckpt, args.save_dir / f'epoch_{epoch:03d}.pth')
        torch.save(ckpt, args.save_dir / 'last.pth')
        if summary['val_total_loss'] < best_val:
            best_val = summary['val_total_loss']
            torch.save(ckpt, args.save_dir / 'best.pth')

        if args.max_steps is not None and global_step >= args.max_steps:
            break


if __name__ == '__main__':
    main()
