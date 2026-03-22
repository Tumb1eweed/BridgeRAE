from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot BridgeRAE stage-1 loss curves')
    parser.add_argument('--ckpt-dir', type=Path, required=True)
    parser.add_argument('--eval-jsonl', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=None)
    return parser.parse_args()


def load_train_val_metrics(ckpt_dir: Path) -> list[dict]:
    rows = []
    for ckpt_path in sorted(ckpt_dir.glob('epoch_*.pth')):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        metrics = ckpt.get('metrics', {})
        rows.append({
            'epoch': int(ckpt.get('epoch', -1)),
            'train_loss': float(metrics.get('train_loss', float('nan'))),
            'val_loss': float(metrics.get('val_loss', float('nan'))),
        })
    return rows


def load_eval_metrics(eval_jsonl: Path | None) -> dict[int, float]:
    if eval_jsonl is None or not eval_jsonl.exists():
        return {}
    eval_loss_by_epoch: dict[int, float] = {}
    with eval_jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            eval_loss_by_epoch[int(row['epoch'])] = float(row['eval_loss'])
    return eval_loss_by_epoch


def main() -> None:
    args = parse_args()
    rows = load_train_val_metrics(args.ckpt_dir)
    if not rows:
        raise FileNotFoundError(f'No epoch checkpoints found in {args.ckpt_dir}')
    eval_jsonl = args.eval_jsonl or (args.ckpt_dir / 'test_eval_metrics.jsonl')
    eval_loss_by_epoch = load_eval_metrics(eval_jsonl)

    epochs = [row['epoch'] for row in rows]
    train_loss = [row['train_loss'] for row in rows]
    val_loss = [row['val_loss'] for row in rows]
    eval_epochs = [epoch for epoch in epochs if epoch in eval_loss_by_epoch]
    eval_loss = [eval_loss_by_epoch[epoch] for epoch in eval_epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, marker='o', linewidth=2, label='train_loss')
    plt.plot(epochs, val_loss, marker='s', linewidth=2, label='val_loss')
    if eval_epochs:
        plt.plot(eval_epochs, eval_loss, marker='^', linewidth=2, label='eval_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BridgeRAE Stage-1 Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output = args.output or (args.ckpt_dir / 'loss_curves.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=200)
    print(output)


if __name__ == '__main__':
    main()
