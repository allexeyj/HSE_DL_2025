import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset

from .augmentations import train_transforms_torch
from .type_defs import PathLike, TrainBatch, TrainItem


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    # Избегаем деления на ноль для пустых или вырожденных облаков точек
    if m < 1e-6:
        return pc
    pc = pc / m
    return pc


def create_pc_tensor_with_dummy_color(pc: np.ndarray, npoints: int) -> torch.Tensor:
    pc_with_dummy_color = np.zeros((npoints, 6), dtype=np.float32)
    pc_with_dummy_color[:, :3] = pc
    pc_with_dummy_color[:, 3:6] = (
        0.5  # Модель ReConV2 ожидает 6 каналов (XYZ + RGB), поэтому добавляем нейтральный серый цвет.
    )
    return torch.from_numpy(pc_with_dummy_color).float()


class TextMeshDataset(Dataset):
    def __init__(
            self,
            root_dir: PathLike,
            captions_file: PathLike,
            npoints: int,
            model_ids: list[str],
            pc_augment: bool,
            base_seed: int = 42,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.npoints = npoints
        self.pc_augment = pc_augment
        self.base_seed = base_seed

        assert model_ids is not None, "model_ids должны быть переданы!"

        # Загружаем captions
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)

        self.pc_dir = self.root_dir / "models"
        self.model_ids = model_ids

        # Фильтруем только те, для которых есть captions
        self.model_ids = [mid for mid in self.model_ids if mid in self.captions]

    def __len__(self) -> int:
        return len(self.model_ids)

    def __getitem__(self, idx: int) -> TrainItem:
        model_id = self.model_ids[idx]

        # Загружаем меш
        pc_path = self.pc_dir / f"{model_id}.stl"
        mesh = trimesh.load(pc_path, force="mesh")

        sample_seed = self.base_seed + idx
        pc, _ = trimesh.sample.sample_surface(mesh, self.npoints, seed=sample_seed)
        pc = np.array(pc, dtype=np.float32)
        pc = normalize_pc(pc)

        pc_tensor = create_pc_tensor_with_dummy_color(pc, self.npoints)

        if self.pc_augment:
            pc_tensor = pc_tensor.unsqueeze(0)
            pc_tensor = train_transforms_torch(pc_tensor)
            pc_tensor = pc_tensor.squeeze(0)

        # Получаем текст
        caption = self.captions[model_id]

        return {"id": model_id, "text": caption, "pc": pc_tensor}


def train_collate_fn(batch: list[TrainItem]) -> TrainBatch:
    ids = [item["id"] for item in batch]
    texts = [item["text"] for item in batch]
    pcs = torch.stack([item["pc"] for item in batch])
    return {"id": ids, "texts": texts, "pc": pcs}


class InferenceTextDataset(Dataset):
    def __init__(self, file_paths: list[str]) -> None:
        self.texts = []
        for path in file_paths:
            with open(path, 'r') as f:
                self.texts.append(f.read().strip())

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


class InferenceMeshDataset(Dataset):
    def __init__(
            self, file_paths: list[str], npoints: int, base_seed: int = 42
    ) -> None:
        self.file_paths = file_paths
        self.npoints = npoints
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        pc_path = self.file_paths[idx]
        mesh = trimesh.load(pc_path, force="mesh")
        sample_seed = self.base_seed + idx
        pc, _ = trimesh.sample.sample_surface(mesh, self.npoints, seed=sample_seed)
        pc = np.array(pc, dtype=np.float32)
        pc = normalize_pc(pc)

        return create_pc_tensor_with_dummy_color(pc, self.npoints)