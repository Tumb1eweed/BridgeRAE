from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from .shapenet_pairs import ShapeNetPointCloudSample, normalize_pair, sample_fixed_size


_VALID_SPLITS = {"train", "val", "test"}


class PCNPointCloudDataset(Dataset[ShapeNetPointCloudSample]):
    """PCN point-pair dataset with ShapeNet-compatible sample fields."""

    def __init__(
        self,
        data_root: str | Path,
        split: str = 'train',
        class_id: str | None = None,
        num_input_points: int = 2048,
        num_complete_points: int = 8192,
        normalize_pair: bool = True,
        include_all_views: bool = True,
        view_id: str | None = None,
        input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {_VALID_SPLITS}.")

        self.data_root = Path(data_root)
        self.split = split
        self.class_id = class_id
        self.num_input_points = num_input_points
        self.num_complete_points = num_complete_points
        self.normalize_pair = normalize_pair
        self.include_all_views = include_all_views
        self.view_id = view_id
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.samples = self._build_index()

    def _build_index(self) -> list[dict[str, object]]:
        split_root = self.data_root / self.split
        complete_root = split_root / 'complete'
        partial_root = split_root / 'partial'
        if not complete_root.exists():
            raise FileNotFoundError(f'Complete point root not found: {complete_root}')
        if not partial_root.exists():
            raise FileNotFoundError(f'Partial point root not found: {partial_root}')

        class_dirs = [complete_root / self.class_id] if self.class_id else sorted(p for p in complete_root.iterdir() if p.is_dir())
        samples: list[dict[str, object]] = []
        for class_dir in class_dirs:
            if not class_dir.exists():
                raise FileNotFoundError(f'Class directory not found: {class_dir}')
            taxonomy_id = class_dir.name
            for complete_path in sorted(class_dir.glob('*.pcd')):
                scan_id = complete_path.stem
                partial_dir = partial_root / taxonomy_id / scan_id
                if not partial_dir.exists():
                    continue

                if self.view_id is not None:
                    partial_paths = [partial_dir / f'{self.view_id}.pcd']
                elif self.include_all_views:
                    partial_paths = sorted(partial_dir.glob('*.pcd'))
                else:
                    partial_paths = sorted(partial_dir.glob('*.pcd'))[:1]

                for partial_path in partial_paths:
                    if not partial_path.exists():
                        continue
                    samples.append(
                        {
                            'scan_id': scan_id,
                            'class_id': taxonomy_id,
                            'view_id': partial_path.stem,
                            'partial_path': partial_path,
                            'complete_path': complete_path,
                        }
                    )

        if not samples:
            raise RuntimeError(
                f'No PCN samples found under {split_root}. '
                f'class_id={self.class_id!r}, view_id={self.view_id!r}'
            )
        return samples

    @staticmethod
    def _load_pcd_points(path: Path) -> torch.Tensor:
        header_complete = False
        points: list[list[float]] = []
        with path.open('r', encoding='utf-8') as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                if not header_complete:
                    if stripped.upper().startswith('DATA '):
                        if stripped.lower() != 'data ascii':
                            raise ValueError(f'Only ASCII PCD files are supported, got {stripped!r} at {path}')
                        header_complete = True
                    continue

                values = stripped.split()
                if len(values) < 3:
                    raise ValueError(f'Expected at least 3 columns in {path}, got line: {stripped!r}')
                points.append([float(values[0]), float(values[1]), float(values[2])])

        if not header_complete:
            raise ValueError(f'Invalid PCD file without DATA header: {path}')
        if not points:
            raise ValueError(f'No points found in PCD file: {path}')

        points_array = np.asarray(points, dtype=np.float32)
        return torch.from_numpy(points_array).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ShapeNetPointCloudSample:
        sample = self.samples[index]
        partial_points = self._load_pcd_points(sample['partial_path'])
        complete_points = self._load_pcd_points(sample['complete_path'])

        partial_points = sample_fixed_size(partial_points, self.num_input_points)
        complete_points = sample_fixed_size(complete_points, self.num_complete_points)

        if self.normalize_pair:
            partial_points, complete_points = normalize_pair(partial_points, complete_points)

        if self.input_transform is not None:
            partial_points = self.input_transform(partial_points)
        if self.target_transform is not None:
            complete_points = self.target_transform(complete_points)

        return ShapeNetPointCloudSample(
            scan_id=str(sample['scan_id']),
            class_id=str(sample['class_id']),
            view_id=str(sample['view_id']),
            partial_path=Path(sample['partial_path']),
            complete_path=Path(sample['complete_path']),
            partial_points=partial_points,
            complete_points=complete_points,
        )
