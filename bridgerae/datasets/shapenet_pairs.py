from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


_VALID_SPLITS = {"train", "val", "test"}


def sample_fixed_size(points: torch.Tensor, count: int) -> torch.Tensor:
    """Sample or pad a point cloud to exactly *count* points."""
    if points.shape[0] == count:
        return points
    if points.shape[0] > count:
        perm = torch.randperm(points.shape[0])[:count]
        return points[perm]
    extra = torch.randint(0, points.shape[0], (count - points.shape[0],))
    return torch.cat([points, points[extra]], dim=0)


def normalize_pair(
    partial_points: torch.Tensor,
    complete_points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Centroid-normalize a partial/complete pair using the complete centroid."""
    centroid = complete_points.mean(dim=0, keepdim=True)
    complete_centered = complete_points - centroid
    partial_centered = partial_points - centroid
    scale = complete_centered.norm(dim=1).max().clamp_min(1e-6)
    return partial_centered / scale, complete_centered / scale


@dataclass(frozen=True)
class ShapeNetPointCloudSample:
    scan_id: str
    class_id: str
    view_id: str
    partial_path: Path
    complete_path: Path
    partial_points: torch.Tensor
    complete_points: torch.Tensor


def shapenet_point_collate_fn(samples: list[ShapeNetPointCloudSample]) -> dict[str, object]:
    return {
        'scan_id': [sample.scan_id for sample in samples],
        'class_id': [sample.class_id for sample in samples],
        'view_id': [sample.view_id for sample in samples],
        'partial_path': [sample.partial_path for sample in samples],
        'complete_path': [sample.complete_path for sample in samples],
        'partial_points': torch.stack([sample.partial_points for sample in samples], dim=0),
        'complete_points': torch.stack([sample.complete_points for sample in samples], dim=0),
    }


class ShapeNetPointCloudDataset(Dataset[ShapeNetPointCloudSample]):
    """Point-pair dataset supporting both pair directories and split-file ShapeNet55/34 layouts."""

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
        split_set: str | None = None,
        pair_data_root: str | Path | None = None,
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
        self.split_set = split_set
        self.pair_data_root = Path(pair_data_root) if pair_data_root is not None else None
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.samples = self._build_index()

    def _build_index(self) -> list[dict[str, object]]:
        if (self.data_root / 'shapenet_pc').exists():
            return self._build_split_file_index()
        return self._build_pair_directory_index()

    def _build_pair_directory_index(self) -> list[dict[str, object]]:
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
            for complete_path in sorted(class_dir.glob('*.npy')):
                scan_id = complete_path.stem
                partial_dir = partial_root / taxonomy_id / scan_id
                if not partial_dir.exists():
                    continue

                if self.view_id is not None:
                    partial_paths = [partial_dir / f'{self.view_id}.npy']
                elif self.include_all_views:
                    partial_paths = sorted(partial_dir.glob('*.npy'))
                else:
                    partial_paths = sorted(partial_dir.glob('*.npy'))[:1]

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
                f'No point-pair samples found under {split_root}. '
                f'class_id={self.class_id!r}, view_id={self.view_id!r}'
            )
        return samples

    def _resolve_pair_root(self) -> Path:
        if self.pair_data_root is not None:
            return self.pair_data_root
        sibling = self.data_root.parent / 'ShapeNet55_PoinTrPairs'
        if sibling.exists():
            return sibling
        raise FileNotFoundError(
            'ShapeNet55 pair root not found. Pass pair_data_root explicitly or place ShapeNet55_PoinTrPairs next to data_root.'
        )

    def _resolve_split_set(self) -> str:
        if self.split_set is not None:
            return self.split_set
        if self.split in {'train', 'val'}:
            return 'ShapeNet-34'
        return 'ShapeNet-Unseen21'

    def _index_pair_partials(self, pair_root: Path) -> dict[tuple[str, str], list[Path]]:
        partial_map: dict[tuple[str, str], list[Path]] = {}
        for pair_split in ('train', 'test'):
            partial_root = pair_root / pair_split / 'partial'
            if not partial_root.exists():
                continue
            class_dirs = [partial_root / self.class_id] if self.class_id else sorted(p for p in partial_root.iterdir() if p.is_dir())
            for class_dir in class_dirs:
                if not class_dir.exists():
                    continue
                taxonomy_id = class_dir.name
                for scan_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
                    partials = sorted(scan_dir.glob('*.npy'))
                    if partials:
                        partial_map[(taxonomy_id, scan_dir.name)] = partials
        return partial_map

    def _build_split_file_index(self) -> list[dict[str, object]]:
        split_set = self._resolve_split_set()
        split_file = self.data_root / split_set / f'{self.split}.txt'
        if not split_file.exists():
            raise FileNotFoundError(f'Split file not found: {split_file}')
        pc_root = self.data_root / 'shapenet_pc'
        if not pc_root.exists():
            raise FileNotFoundError(f'Point cloud root not found: {pc_root}')
        pair_root = self._resolve_pair_root()
        partial_map = self._index_pair_partials(pair_root)

        samples: list[dict[str, object]] = []
        for line in split_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            taxonomy_id, model_stub = line.split('-', 1)
            if self.class_id is not None and taxonomy_id != self.class_id:
                continue
            scan_id = model_stub.rsplit('.', 1)[0]
            complete_path = pc_root / line
            if not complete_path.exists():
                continue

            partial_paths = partial_map.get((taxonomy_id, scan_id), [])
            if self.view_id is not None:
                partial_paths = [p for p in partial_paths if p.stem == self.view_id]
            elif not self.include_all_views:
                partial_paths = partial_paths[:1]

            for partial_path in partial_paths:
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
                f'No point-pair samples found for split file {split_file}. '
                f'class_id={self.class_id!r}, view_id={self.view_id!r}'
            )
        return samples

    @staticmethod
    def _load_points(path: Path) -> torch.Tensor:
        points = np.load(path)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f'Expected Nx3 point cloud at {path}, got shape {points.shape}')
        return torch.from_numpy(points[:, :3]).float()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> ShapeNetPointCloudSample:
        sample = self.samples[index]
        partial_points = self._load_points(sample['partial_path'])
        complete_points = self._load_points(sample['complete_path'])

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
