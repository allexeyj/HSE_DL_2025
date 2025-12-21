import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict

import numpy as np
import torch
from PIL import Image

# --- Примитивные псевдонимы и новые типы ---
ModelID: TypeAlias = str
PathLike: TypeAlias = str | Path | os.PathLike[str]

# --- Псевдонимы для Callable ---
ImageTransform: TypeAlias = Callable[[Image.Image], torch.Tensor]


# --- Типы данных для датасетов ---
class TrainItem(TypedDict):
    id: ModelID
    images: torch.Tensor
    pc: torch.Tensor


class TrainBatch(TypedDict):
    id: list[ModelID]
    images_list: list[torch.Tensor]
    pc: torch.Tensor


# --- Типы для NumPy массивов ---
EmbeddingArray: TypeAlias = np.ndarray

# --- Типы для моделей и ансамблей ---
# Это позволяет импортировать классы только для проверки типов, избегая циклических зависимостей во время выполнения.
if TYPE_CHECKING:
    from .models import ImageEncoder, InferencePcEncoder


class CheckpointSpec(TypedDict):
    moe: PathLike
    pc_encoder: PathLike


class EnsembleItem(TypedDict):
    img_encoder: "ImageEncoder"
    pc_encoder: "InferencePcEncoder"


InferenceMode: TypeAlias = Literal["image", "mesh"]

# --- Типы для метрик ---
RecallDict: TypeAlias = dict[int, float]


class ValidationMetrics(TypedDict):
    loss: float
    recall_img2pc: RecallDict
    recall_pc2img: RecallDict


ValidationResult: TypeAlias = tuple[
    EmbeddingArray,  # pc_embs
    EmbeddingArray,  # img_embs
    list[ModelID],  # pc_ids
    list[ModelID],  # img_ids
    float,  # avg_loss
]


class BestMetricInfo(TypedDict):
    """Информация о лучшей метрике"""

    value: float
    epoch: int
    recall_img2pc: RecallDict
    recall_pc2img: RecallDict
