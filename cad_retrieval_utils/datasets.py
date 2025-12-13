from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import train_transforms_torch
from .type_defs import ImageTransform, PathLike, TrainBatch, TrainItem


def normalize_pc(pc: np.ndarray) -> np.ndarray:
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        return pc
    pc = pc / m
    return pc


def create_pc_tensor_with_dummy_color(pc: np.ndarray, npoints: int) -> torch.Tensor:
    pc_with_dummy_color = np.zeros((npoints, 6), dtype=np.float32)
    pc_with_dummy_color[:, :3] = pc
    pc_with_dummy_color[:, 3:6] = 0.5
    return torch.from_numpy(pc_with_dummy_color).float()


def load_mesh_safe(mesh_path: Path, npoints: int, seed: int) -> np.ndarray:
    """Безопасная загрузка меша с обработкой Scene объектов"""
    mesh_data = trimesh.load(str(mesh_path))

    if isinstance(mesh_data, trimesh.Scene):
        # Конвертируем Scene в единый mesh
        mesh = mesh_data.to_mesh()
    else:
        mesh = mesh_data

    pc, _ = trimesh.sample.sample_surface(mesh, npoints, seed=seed)
    return np.array(pc, dtype=np.float32)


class EmbeddingCache:
    def __init__(self, embeddings_path: PathLike) -> None:
        print(f"Загрузка эмбеддингов из {embeddings_path}...")
        data = np.load(embeddings_path)
        self.embeddings = data["embeddings"]
        self.filenames = data["filenames"]

        self.embedding_dict: dict[str, np.ndarray] = {
            fname: self.embeddings[i] for i, fname in enumerate(self.filenames)
        }

        # Извлекаем уникальные model_ids из имен файлов
        self.model_ids = set()
        for fname in self.filenames:
            model_id = fname.split('_')[0]  # Извлекаем ID из имени файла
            self.model_ids.add(model_id)

        print(f"✅ Загружено {len(self.embedding_dict)} эмбеддингов")
        print(f"   Уникальных моделей: {len(self.model_ids)}")

    def get(self, filename: str) -> np.ndarray:
        if filename not in self.embedding_dict:
            raise KeyError(f"Эмбеддинг не найден для {filename}. Это ошибка в данных!")
        return self.embedding_dict[filename]


class TrainDataset(Dataset):
    def __init__(
            self,
            root_dir: PathLike,
            npoints: int,
            num_views: int,
            pc_augment: bool,
            base_seed: int = 42,
            embeddings_cache: EmbeddingCache | None = None,
    ) -> None:
        if embeddings_cache is None:
            raise ValueError("embeddings_cache обязателен!")

        self.root_dir = Path(root_dir)
        self.npoints = npoints
        self.num_views = num_views
        self.pc_augment = pc_augment
        self.base_seed = base_seed
        self.embeddings_cache = embeddings_cache

        self.pc_dir = self.root_dir / "models"

        # Все model_ids из кеша валидны по определению
        self.model_ids = sorted(list(self.embeddings_cache.model_ids))

        print(f"✅ TrainDataset инициализирован:")
        print(f"   - Моделей для обучения: {len(self.model_ids)}")
        print(f"   - Views на модель: {self.num_views}")

        # Проверяем что у всех моделей есть все views
        self._validate_views()

    def _validate_views(self) -> None:
        """Проверяет что у всех моделей есть все необходимые views"""
        missing_views = []
        for model_id in self.model_ids:
            for view_idx in range(self.num_views):
                filename = f"{model_id}_{view_idx}.png"
                if filename not in self.embeddings_cache.embedding_dict:
                    missing_views.append(filename)

        if missing_views:
            raise ValueError(f"Отсутствуют эмбеддинги для views: {missing_views[:5]}...")

    def __len__(self) -> int:
        return len(self.model_ids)

    def __getitem__(self, idx: int) -> TrainItem:
        model_id = self.model_ids[idx]
        pc_path = self.pc_dir / f"{model_id}.stl"

        if not pc_path.exists():
            raise FileNotFoundError(f"STL файл не найден: {pc_path}")

        # Безопасная загрузка с обработкой Scene
        sample_seed = self.base_seed + idx
        pc = load_mesh_safe(pc_path, self.npoints, sample_seed)
        pc = normalize_pc(pc)

        pc_tensor = create_pc_tensor_with_dummy_color(pc, self.npoints)

        if self.pc_augment:
            pc_tensor = pc_tensor.unsqueeze(0)
            pc_tensor = train_transforms_torch(pc_tensor)
            pc_tensor = pc_tensor.squeeze(0)

        # Загружаем эмбеддинги для всех views
        image_embeddings = []
        for i in range(self.num_views):
            img_filename = f"{model_id}_{i}.png"
            emb = self.embeddings_cache.get(img_filename)  # Теперь кидает исключение если нет
            image_embeddings.append(torch.from_numpy(emb).float())

        images_tensor = torch.stack(image_embeddings)

        return {"id": model_id, "images": images_tensor, "pc": pc_tensor}


def train_collate_fn(batch: list[TrainItem]) -> TrainBatch:
    ids = [item["id"] for item in batch]
    pcs = torch.stack([item["pc"] for item in batch])
    images_list = [item["images"] for item in batch]
    return {"id": ids, "images_list": images_list, "pc": pcs}


class InferenceImageDataset(Dataset):
    def __init__(self, file_paths: list[str], transform: ImageTransform) -> None:
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img)


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
        pc_path = Path(self.file_paths[idx])
        sample_seed = self.base_seed + idx
        pc = load_mesh_safe(pc_path, self.npoints, sample_seed)
        pc = normalize_pc(pc)
        return create_pc_tensor_with_dummy_color(pc, self.npoints)