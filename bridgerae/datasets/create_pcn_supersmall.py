from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create a tiny PCN subset for fast overfit tests.')
    parser.add_argument('--source-root', type=Path, default=Path('/root/autodl-tmp/datasets/PCN'))
    parser.add_argument('--target-root', type=Path, default=Path('/root/autodl-tmp/datasets/PCN-supersmall'))
    parser.add_argument('--class-id', type=str, default='02691156')
    parser.add_argument('--train-complete-count', type=int, default=30)
    parser.add_argument('--val-complete-count', type=int, default=30)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def select_complete_files(source_root: Path, split: str, class_id: str, count: int) -> list[Path]:
    complete_dir = source_root / split / 'complete' / class_id
    if not complete_dir.exists():
        raise FileNotFoundError(f'Complete directory not found: {complete_dir}')

    selected: list[Path] = []
    for complete_path in sorted(complete_dir.glob('*.pcd')):
        partial_dir = source_root / split / 'partial' / class_id / complete_path.stem
        if partial_dir.exists() and any(partial_dir.glob('*.pcd')):
            selected.append(complete_path)
        if len(selected) == count:
            break

    if len(selected) < count:
        raise RuntimeError(
            f'Only found {len(selected)} complete clouds with partial views for '
            f'{split}/{class_id}; requested {count}.'
        )
    return selected


def copy_split(source_root: Path, target_root: Path, split: str, class_id: str, count: int) -> tuple[int, int]:
    complete_files = select_complete_files(source_root, split, class_id, count)
    target_complete_dir = target_root / split / 'complete' / class_id
    target_partial_class_dir = target_root / split / 'partial' / class_id
    target_complete_dir.mkdir(parents=True, exist_ok=True)
    target_partial_class_dir.mkdir(parents=True, exist_ok=True)

    partial_count = 0
    for complete_path in complete_files:
        scan_id = complete_path.stem
        shutil.copy2(complete_path, target_complete_dir / complete_path.name)

        source_partial_dir = source_root / split / 'partial' / class_id / scan_id
        target_partial_dir = target_partial_class_dir / scan_id
        target_partial_dir.mkdir(parents=True, exist_ok=True)
        for partial_path in sorted(source_partial_dir.glob('*.pcd')):
            shutil.copy2(partial_path, target_partial_dir / partial_path.name)
            partial_count += 1

    return len(complete_files), partial_count


def main() -> None:
    args = parse_args()
    if args.target_root.exists():
        if not args.overwrite:
            raise FileExistsError(f'Target already exists: {args.target_root}. Pass --overwrite to replace it.')
        shutil.rmtree(args.target_root)

    train_counts = copy_split(
        args.source_root,
        args.target_root,
        'train',
        args.class_id,
        args.train_complete_count,
    )
    val_counts = copy_split(
        args.source_root,
        args.target_root,
        'val',
        args.class_id,
        args.val_complete_count,
    )

    print(f'Created {args.target_root}')
    print(f'train: complete={train_counts[0]}, partial_pairs={train_counts[1]}')
    print(f'val: complete={val_counts[0]}, partial_pairs={val_counts[1]}')


if __name__ == '__main__':
    main()
