from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


LOSS_KEYS = ['train_loss', 'val_loss']
VAL_METRIC_KEYS = ['val_chamfer_distance', 'val_emd']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot BridgeRAE stage-1 training curves')
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--loss-output', type=Path, default=None)
    parser.add_argument('--val-output', type=Path, default=None)
    return parser.parse_args()


def load_epoch_metrics(ckpt_dir: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for ckpt_path in sorted(ckpt_dir.glob('epoch_*.pth')):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metrics = ckpt.get('metrics', {}) or {}
        row = {'epoch': int(ckpt.get('epoch', -1))}
        for key in LOSS_KEYS + VAL_METRIC_KEYS:
            row[key] = float(metrics.get(key, float('nan')))
        rows.append(row)
    return rows


def filter_valid_points(epochs: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    valid_epochs: list[int] = []
    valid_values: list[float] = []
    for epoch, value in zip(epochs, values):
        if value == value:
            valid_epochs.append(epoch)
            valid_values.append(value)
    return valid_epochs, valid_values


def plot_loss_curves(rows: list[dict[str, float]], output: Path) -> None:
    epochs = [row['epoch'] for row in rows]
    plt.figure(figsize=(10, 6))
    for key, marker in [('train_loss', 'o'), ('val_loss', 's')]:
        plot_epochs, plot_values = filter_valid_points(epochs, [row[key] for row in rows])
        if plot_epochs:
            plt.plot(plot_epochs, plot_values, marker=marker, linewidth=2, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BridgeRAE Stage-1 Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    plt.close()


def plot_val_metric_curves(rows: list[dict[str, float]], output: Path) -> None:
    epochs = [row['epoch'] for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    for ax, key, marker in zip(axes, VAL_METRIC_KEYS, ['^', 'd']):
        plot_epochs, plot_values = filter_valid_points(epochs, [row[key] for row in rows])
        if plot_epochs:
            ax.plot(plot_epochs, plot_values, marker=marker, linewidth=2, label=key)
            ax.legend()
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Epoch')
    fig.suptitle('BridgeRAE Stage-1 Validation Metrics')
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_epoch_metrics(args.ckpt_dir)
    if not rows:
        raise FileNotFoundError(f'No epoch checkpoints found in {args.ckpt_dir}')

    loss_output = args.loss_output or (args.ckpt_dir / 'loss_curves.png')
    val_output = args.val_output or (args.ckpt_dir / 'val_metric_curves.png')
    plot_loss_curves(rows, loss_output)
    plot_val_metric_curves(rows, val_output)
    print(loss_output)
    print(val_output)


if __name__ == '__main__':
    main()
