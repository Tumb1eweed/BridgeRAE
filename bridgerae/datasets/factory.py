from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from .pcn_pairs import PCNPointCloudDataset
from .shapenet_pairs import ShapeNetPointCloudDataset


DEFAULT_DATA_ROOTS = {
    'shapenet': Path('/root/autodl-tmp/datasets/ShapeNet55'),
    'pcn': Path('/root/autodl-tmp/datasets/PCN'),
}

DEFAULT_PAIR_DATA_ROOTS = {
    'shapenet': Path('/root/autodl-tmp/datasets/ShapeNet55_PoinTrPairs'),
}

DEFAULT_POINT_COUNTS = {
    'shapenet': {
        'num_input_points': 2048,
        'num_complete_points': 8192,
    },
    'pcn': {
        'num_input_points': 2048,
        'num_complete_points': 16384,
    },
}


def resolve_point_counts(
    dataset_name: str,
    num_input_points: int | None = None,
    num_complete_points: int | None = None,
) -> tuple[int, int]:
    if dataset_name not in DEFAULT_POINT_COUNTS:
        raise ValueError(f'Unsupported dataset {dataset_name!r}. Expected one of {sorted(DEFAULT_POINT_COUNTS)}.')
    defaults = DEFAULT_POINT_COUNTS[dataset_name]
    return (
        int(num_input_points if num_input_points is not None else defaults['num_input_points']),
        int(num_complete_points if num_complete_points is not None else defaults['num_complete_points']),
    )


def build_pointcloud_dataset(
    dataset_name: str,
    data_root: str | Path | None,
    split: str,
    class_id: str | None,
    num_input_points: int,
    num_complete_points: int,
    split_set: str | None = None,
    pair_data_root: str | Path | None = None,
    normalize_pair: bool = True,
    include_all_views: bool = True,
    view_id: str | None = None,
) -> Dataset:
    if dataset_name == 'shapenet':
        resolved_data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOTS['shapenet']
        resolved_pair_root = Path(pair_data_root) if pair_data_root is not None else DEFAULT_PAIR_DATA_ROOTS['shapenet']
        return ShapeNetPointCloudDataset(
            data_root=resolved_data_root,
            pair_data_root=resolved_pair_root,
            split=split,
            split_set=split_set,
            class_id=class_id,
            num_input_points=num_input_points,
            num_complete_points=num_complete_points,
            normalize_pair=normalize_pair,
            include_all_views=include_all_views,
            view_id=view_id,
        )

    if dataset_name == 'pcn':
        resolved_data_root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOTS['pcn']
        return PCNPointCloudDataset(
            data_root=resolved_data_root,
            split=split,
            class_id=class_id,
            num_input_points=num_input_points,
            num_complete_points=num_complete_points,
            normalize_pair=normalize_pair,
            include_all_views=include_all_views,
            view_id=view_id,
        )

    raise ValueError(f'Unsupported dataset {dataset_name!r}. Expected one of {sorted(DEFAULT_DATA_ROOTS)}.')
