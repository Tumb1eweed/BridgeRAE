from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run one-batch PCN stage1/stage2 experiment and collect curves')
    parser.add_argument('--class-id', type=str, default='03001627')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--metric-points', type=int, default=2048)
    parser.add_argument('--transport-steps', type=int, default=8)
    parser.add_argument('--stage1-lr', type=float, default=5e-4)
    parser.add_argument('--stage2-lr', type=float, default=2e-4)
    parser.add_argument('--save-root', type=Path, default=Path('/root/autodl-tmp/projects/BridgeRAE/outputs/pcn_onebatch'))
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/PCN'))
    parser.add_argument('--encoder-ckpt', type=Path, default=Path('/root/autodl-tmp/projects/Point-MAE/checkpoint/pretrain.pth'))
    parser.add_argument('--latent-normalize', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--latent-noise-std', type=float, default=0.1)
    parser.add_argument('--decoder-train-mode', type=str, default='last_n', choices=['none', 'last_n', 'all'])
    parser.add_argument('--decoder-train-last-n', type=int, default=2)
    parser.add_argument('--decoder-lr-mult', type=float, default=0.1)
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--skip-stage1', action='store_true')
    parser.add_argument('--skip-stage2', action='store_true')
    parser.add_argument('--stage1-ckpt', type=Path, default=None)
    return parser.parse_args()


def _run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    with log_path.open('w', encoding='utf-8') as handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            handle.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _extract_epoch_rows(log_path: Path, keys: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in log_path.read_text(encoding='utf-8').splitlines():
        line = raw_line.strip()
        json_start = line.find('{')
        if json_start < 0:
            continue
        try:
            record = json.loads(line[json_start:])
        except json.JSONDecodeError:
            continue
        if 'epoch' in record and any(key in record for key in keys):
            rows.append(record)
    rows.sort(key=lambda item: int(item['epoch']))
    return rows


def _write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_curves(stage1_rows: list[dict[str, Any]], stage2_rows: list[dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(json.dumps({'warning': 'plot_skipped', 'reason': str(exc)}), flush=True)
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if stage1_rows:
        stage1_epochs = [int(row['epoch']) + 1 for row in stage1_rows]
        axes[0, 0].plot(stage1_epochs, [row.get('train_loss', float('nan')) for row in stage1_rows], label='stage1 train_loss')
        axes[0, 0].plot(stage1_epochs, [row.get('val_loss', float('nan')) for row in stage1_rows], label='stage1 val_loss')
        axes[0, 1].plot(stage1_epochs, [row.get('val_chamfer_distance', float('nan')) for row in stage1_rows], label='stage1 val CD-L2')
        axes[1, 0].plot(stage1_epochs, [row.get('val_emd', float('nan')) for row in stage1_rows], label='stage1 val EMD')
        axes[1, 1].plot(stage1_epochs, [row.get('val_f1_1pct', float('nan')) for row in stage1_rows], label='stage1 val F1@1%')

    if stage2_rows:
        stage2_epochs = [int(row['epoch']) + 1 for row in stage2_rows]
        axes[0, 0].plot(stage2_epochs, [row.get('train_total_loss', float('nan')) for row in stage2_rows], label='stage2 train_total')
        axes[0, 0].plot(stage2_epochs, [row.get('val_recon_loss', float('nan')) for row in stage2_rows], label='stage2 val_recon')
        axes[0, 1].plot(stage2_epochs, [row.get('val_chamfer_distance', float('nan')) for row in stage2_rows], label='stage2 val CD-L2')
        axes[1, 0].plot(stage2_epochs, [row.get('val_emd', float('nan')) for row in stage2_rows], label='stage2 val EMD')
        axes[1, 1].plot(stage2_epochs, [row.get('val_f1_1pct', float('nan')) for row in stage2_rows], label='stage2 val F1@1%')

    titles = ['Loss', 'Val CD-L2', 'Val EMD', 'Val F1@1%']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(True, alpha=0.3)
        if ax.lines:
            ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)


def _final_metric(rows: list[dict[str, Any]], key: str) -> float | None:
    for row in reversed(rows):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _best_metric(rows: list[dict[str, Any]], key: str, mode: str = 'min') -> float | None:
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return min(values) if mode == 'min' else max(values)


def _build_stage1_command(args: argparse.Namespace, stage1_dir: Path) -> list[str]:
    command = [
        sys.executable, '-m', 'bridgerae.training.stage1_train',
        '--dataset', 'pcn',
        '--data-root', str(args.data_root),
        '--class-id', args.class_id,
        '--batch-size', str(args.batch_size),
        '--val-batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--epochs', str(args.epochs),
        '--max-train-samples', str(args.batch_size),
        '--max-val-samples', str(args.batch_size),
        '--eval-every', str(args.eval_every),
        '--max-val-batches', '1',
        '--metric-points', str(args.metric_points),
        '--lr', str(args.stage1_lr),
        '--save-dir', str(stage1_dir),
        '--encoder-ckpt', str(args.encoder_ckpt),
        '--no-train-shuffle',
        '--no-save-epoch-checkpoints',
    ]
    if args.latent_normalize:
        command.append('--latent-normalize')
    if args.latent_noise_std > 0:
        command.extend(['--latent-noise-std', str(args.latent_noise_std)])
    if args.amp:
        command.append('--amp')
    return command


def _build_stage2_command(args: argparse.Namespace, stage2_dir: Path, stage1_ckpt: Path) -> list[str]:
    command = [
        sys.executable, '-m', 'bridgerae.training.stage2_train',
        '--dataset', 'pcn',
        '--data-root', str(args.data_root),
        '--class-id', args.class_id,
        '--stage1-ckpt', str(stage1_ckpt),
        '--batch-size', str(args.batch_size),
        '--num-workers', str(args.num_workers),
        '--epochs', str(args.epochs),
        '--max-train-samples', str(args.batch_size),
        '--max-val-samples', str(args.batch_size),
        '--eval-every', str(args.eval_every),
        '--max-val-batches', '1',
        '--metric-points', str(args.metric_points),
        '--transport-steps', str(args.transport_steps),
        '--lr', str(args.stage2_lr),
        '--save-dir', str(stage2_dir),
        '--encoder-ckpt', str(args.encoder_ckpt),
        '--decoder-train-mode', args.decoder_train_mode,
        '--decoder-train-last-n', str(args.decoder_train_last_n),
        '--decoder-lr-mult', str(args.decoder_lr_mult),
        '--no-save-epoch-checkpoints',
    ]
    if args.amp:
        command.append('--amp')
    return command


def main() -> None:
    args = parse_args()
    run_root = args.save_root / f'class_{args.class_id}_bs{args.batch_size}_ep{args.epochs}'
    stage1_dir = run_root / 'stage1'
    stage2_dir = run_root / 'stage2'
    run_root.mkdir(parents=True, exist_ok=True)

    stage1_log = run_root / 'stage1.log'
    stage2_log = run_root / 'stage2.log'

    stage1_ckpt = args.stage1_ckpt or (stage1_dir / 'best.pth')

    if not args.skip_stage1:
        _run_and_log(_build_stage1_command(args, stage1_dir), stage1_log)
    elif not stage1_log.exists():
        raise FileNotFoundError(f'stage1 log not found: {stage1_log}')

    if not args.skip_stage2:
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f'stage1 checkpoint not found: {stage1_ckpt}')
        _run_and_log(_build_stage2_command(args, stage2_dir, stage1_ckpt), stage2_log)
    elif not stage2_log.exists():
        raise FileNotFoundError(f'stage2 log not found: {stage2_log}')

    stage1_rows = _extract_epoch_rows(stage1_log, {'train_loss', 'val_loss'}) if stage1_log.exists() else []
    stage2_rows = _extract_epoch_rows(stage2_log, {'train_total_loss', 'val_total_loss', 'val_recon_loss'}) if stage2_log.exists() else []

    _write_csv(stage1_rows, run_root / 'stage1_metrics.csv')
    _write_csv(stage2_rows, run_root / 'stage2_metrics.csv')
    _plot_curves(stage1_rows, stage2_rows, run_root / 'curves.png')

    summary = {
        'class_id': args.class_id,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'stage1_final_val_loss': _final_metric(stage1_rows, 'val_loss'),
        'stage1_final_val_cd_l2': _final_metric(stage1_rows, 'val_chamfer_distance'),
        'stage1_final_val_emd': _final_metric(stage1_rows, 'val_emd'),
        'stage1_final_val_f1_1pct': _final_metric(stage1_rows, 'val_f1_1pct'),
        'stage1_best_val_loss': _best_metric(stage1_rows, 'val_loss', mode='min'),
        'stage1_best_val_cd_l2': _best_metric(stage1_rows, 'val_chamfer_distance', mode='min'),
        'stage1_best_val_emd': _best_metric(stage1_rows, 'val_emd', mode='min'),
        'stage1_best_val_f1_1pct': _best_metric(stage1_rows, 'val_f1_1pct', mode='max'),
        'stage2_final_val_total_loss': _final_metric(stage2_rows, 'val_total_loss'),
        'stage2_final_val_recon_loss': _final_metric(stage2_rows, 'val_recon_loss'),
        'stage2_final_val_cd_l2': _final_metric(stage2_rows, 'val_chamfer_distance'),
        'stage2_final_val_emd': _final_metric(stage2_rows, 'val_emd'),
        'stage2_final_val_f1_1pct': _final_metric(stage2_rows, 'val_f1_1pct'),
        'stage2_best_val_total_loss': _best_metric(stage2_rows, 'val_total_loss', mode='min'),
        'stage2_best_val_recon_loss': _best_metric(stage2_rows, 'val_recon_loss', mode='min'),
        'stage2_best_val_cd_l2': _best_metric(stage2_rows, 'val_chamfer_distance', mode='min'),
        'stage2_best_val_emd': _best_metric(stage2_rows, 'val_emd', mode='min'),
        'stage2_best_val_f1_1pct': _best_metric(stage2_rows, 'val_f1_1pct', mode='max'),
        'stage1_log': str(stage1_log),
        'stage2_log': str(stage2_log),
        'stage1_csv': str(run_root / 'stage1_metrics.csv'),
        'stage2_csv': str(run_root / 'stage2_metrics.csv'),
        'curves_png': str(run_root / 'curves.png'),
    }
    (run_root / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
