from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .epn_voxel import EPNVoxelDataset


@dataclass(frozen=True)
class EPNPointCloudSample:
    scan_id: str
    class_id: str
    sample_path: Path
    partial_points: torch.Tensor
    complete_points: torch.Tensor
    input_sdf: torch.Tensor
    target_df: torch.Tensor


def epn_point_collate_fn(samples: list[EPNPointCloudSample]) -> dict[str, object]:
    return {
        'scan_id': [sample.scan_id for sample in samples],
        'class_id': [sample.class_id for sample in samples],
        'sample_path': [sample.sample_path for sample in samples],
        'partial_points': torch.stack([sample.partial_points for sample in samples], dim=0),
        'complete_points': torch.stack([sample.complete_points for sample in samples], dim=0),
        'input_sdf': torch.stack([sample.input_sdf for sample in samples], dim=0),
        'target_df': torch.stack([sample.target_df for sample in samples], dim=0),
    }


class EPNPointCloudDataset(Dataset[EPNPointCloudSample]):
    """Surface-point wrapper over the 3D-EPN paired voxel dataset.

    The partial point cloud is extracted from the recovered signed SDF surface,
    while the complete point cloud is extracted from near-zero voxels of the
    complete distance field.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        class_id: str | None = None,
        per_class: bool = True,
        num_input_points: int = 768,
        num_complete_points: int = 2048,
        surface_threshold: float = 1.5,
        trunc_distance: float | None = 3.0,
        log_df: bool = False,
        normalize_pair: bool = True,
    ) -> None:
        self.voxel_dataset = EPNVoxelDataset(
            data_root=data_root,
            split=split,
            class_id=class_id,
            per_class=per_class,
            trunc_distance=trunc_distance,
            log_df=log_df,
        )
        self.num_input_points = num_input_points
        self.num_complete_points = num_complete_points
        self.surface_threshold = surface_threshold
        self.normalize_pair = normalize_pair

    def __len__(self) -> int:
        return len(self.voxel_dataset)

    @staticmethod
    def _recover_signed_sdf(input_sdf: torch.Tensor) -> torch.Tensor:
        return input_sdf[0] * input_sdf[1]

    @staticmethod
    def _voxel_indices_to_points(indices: torch.Tensor, resolution: int) -> torch.Tensor:
        points = (indices.float() + 0.5) / float(resolution)
        points = points * 2.0 - 1.0
        return points[:, [2, 1, 0]]

    def _surface_points_from_field(
        self,
        field: torch.Tensor,
        threshold: float,
        count: int,
        absolute: bool,
    ) -> torch.Tensor:
        values = field.abs() if absolute else field
        mask = values <= threshold
        indices = mask.nonzero(as_tuple=False)

        if indices.numel() == 0:
            flat = values.reshape(-1)
            k = min(count, flat.numel())
            topk = torch.topk(flat, k=k, largest=False).indices
            indices = torch.stack(torch.unravel_index(topk, values.shape), dim=1)

        points = self._voxel_indices_to_points(indices, resolution=field.shape[-1])
        return self._sample_fixed_size(points, count)

    @staticmethod
    def _sample_fixed_size(points: torch.Tensor, count: int) -> torch.Tensor:
        if points.shape[0] == count:
            return points
        if points.shape[0] > count:
            perm = torch.randperm(points.shape[0])[:count]
            return points[perm]
        extra = torch.randint(0, points.shape[0], (count - points.shape[0],))
        return torch.cat([points, points[extra]], dim=0)

    @staticmethod
    def _normalize_pair(
        partial_points: torch.Tensor,
        complete_points: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        centroid = complete_points.mean(dim=0, keepdim=True)
        complete_centered = complete_points - centroid
        partial_centered = partial_points - centroid
        scale = complete_centered.norm(dim=1).max().clamp_min(1e-6)
        return partial_centered / scale, complete_centered / scale

    def __getitem__(self, index: int) -> EPNPointCloudSample:
        voxel_sample = self.voxel_dataset[index]
        signed_sdf = self._recover_signed_sdf(voxel_sample.input_sdf)

        partial_points = self._surface_points_from_field(
            signed_sdf,
            threshold=self.surface_threshold,
            count=self.num_input_points,
            absolute=True,
        )
        complete_points = self._surface_points_from_field(
            voxel_sample.target_df,
            threshold=self.surface_threshold,
            count=self.num_complete_points,
            absolute=False,
        )

        if self.normalize_pair:
            partial_points, complete_points = self._normalize_pair(partial_points, complete_points)

        return EPNPointCloudSample(
            scan_id=voxel_sample.scan_id,
            class_id=voxel_sample.class_id,
            sample_path=voxel_sample.sample_path,
            partial_points=partial_points,
            complete_points=complete_points,
            input_sdf=voxel_sample.input_sdf,
            target_df=voxel_sample.target_df,
        )
