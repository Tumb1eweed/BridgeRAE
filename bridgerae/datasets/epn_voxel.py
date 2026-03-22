from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


_VALID_SPLITS = {"train", "test"}


@dataclass(frozen=True)
class EPNVoxelSample:
    scan_id: str
    class_id: str
    sample_path: Path
    input_sdf: torch.Tensor
    target_df: torch.Tensor


class EPNVoxelDataset(Dataset[EPNVoxelSample]):
    """Paired 3D-EPN voxel dataset used as BridgeRAE stage-1 data source.

    Each `.pth` file stores a tuple:
    - `input_sdf`: `(2, 32, 32, 32)` with `[abs(sdf), sign(sdf)]`
    - `target_df`: `(32, 32, 32)` complete-shape distance field

    This loader keeps the semantics aligned with DiffComplete and provides a
    stable BridgeRAE-facing interface for later voxel-to-point conversion.
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        class_id: str | None = None,
        per_class: bool = True,
        suffix: str = ".pth",
        trunc_distance: float | None = 3.0,
        log_df: bool = False,
        input_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        target_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if split not in _VALID_SPLITS:
            raise ValueError(f"Unsupported split '{split}'. Expected one of {_VALID_SPLITS}.")

        self.data_root = Path(data_root)
        self.split = split
        self.class_id = class_id
        self.per_class = per_class
        self.suffix = suffix
        self.trunc_distance = trunc_distance
        self.log_df = log_df
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.split_file = self._resolve_split_file()
        self.relative_paths = self._read_split_file(self.split_file)

    def _resolve_split_file(self) -> Path:
        split_dir = self.data_root / "splits"
        if self.per_class:
            if not self.class_id:
                raise ValueError("`class_id` is required when `per_class=True`.")
            split_file = split_dir / f"{self.split}_{self.class_id}.txt"
        else:
            split_file = split_dir / f"{self.split}.txt"

        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        return split_file

    @staticmethod
    def _read_split_file(split_file: Path) -> list[str]:
        with split_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _load_pth(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
        input_sdf, target_df = torch.load(sample_path, map_location="cpu")
        return input_sdf, target_df

    def _postprocess_input(self, input_sdf: np.ndarray) -> torch.Tensor:
        input_tensor = torch.from_numpy(np.asarray(input_sdf)).float()
        if self.trunc_distance is not None:
            input_tensor[0] = input_tensor[0].clamp_(0.0, float(self.trunc_distance))
        if self.input_transform is not None:
            input_tensor = self.input_transform(input_tensor)
        return input_tensor

    def _postprocess_target(self, target_df: np.ndarray) -> torch.Tensor:
        target_tensor = torch.from_numpy(np.asarray(target_df)).float()
        if self.trunc_distance is not None:
            target_tensor = target_tensor.clamp_(0.0, float(self.trunc_distance))
        if self.log_df:
            target_tensor = torch.log(target_tensor + 1.0)
        if self.target_transform is not None:
            target_tensor = self.target_transform(target_tensor)
        return target_tensor

    def __len__(self) -> int:
        return len(self.relative_paths)

    def __getitem__(self, index: int) -> EPNVoxelSample:
        relative_path = self.relative_paths[index]
        sample_path = self.data_root / relative_path
        class_id = sample_path.parent.name
        scan_id = sample_path.name.replace(self.suffix, "")

        input_sdf, target_df = self._load_pth(sample_path)
        input_tensor = self._postprocess_input(input_sdf)
        target_tensor = self._postprocess_target(target_df)

        return EPNVoxelSample(
            scan_id=scan_id,
            class_id=class_id,
            sample_path=sample_path,
            input_sdf=input_tensor,
            target_df=target_tensor,
        )
