import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias, TypedDict, Optional

import numpy as np
import torch
from PIL import Image

# --- Примитивные псевдонимы и новые типы ---
ModelID: TypeAlias = str
PathLike: TypeAlias = str | Path | os.PathLike[str]

# --- Типы данных для датасетов ---
class TrainItem(TypedDict):
    id: ModelID
    text: str
    pc: torch.Tensor

class TrainBatch(TypedDict):
    id: list[ModelID]
    texts: list[str]
    pc: torch.Tensor


# --- Типы для NumPy массивов ---
EmbeddingArray: TypeAlias = np.ndarray

# --- Типы для моделей и ансамблей ---
# Это позволяет импортировать классы только для проверки типов, избегая циклических зависимостей во время выполнения.
if TYPE_CHECKING:
    from .models import ImageEncoder


class CheckpointSpec(TypedDict):
    text_proj: PathLike
    text_encoder: Optional[PathLike]  # Используем Optional вместо NotRequired
    pc_encoder: PathLike



InferenceMode: TypeAlias = Literal["image", "mesh"]

# --- Типы для метрик ---
RecallDict: TypeAlias = dict[int, float]


class ValidationMetrics(TypedDict):
    loss: float
    recall_text2pc: RecallDict
    recall_pc2text: RecallDict


ValidationResult: TypeAlias = tuple[
    EmbeddingArray,  # pc_embs
    EmbeddingArray,  # img_embs
    list[ModelID],  # pc_ids
    list[ModelID],  # img_ids
    float,  # avg_loss
]


class BestMetricInfo(TypedDict):
    value: float
    epoch: int
    recall_text2pc: RecallDict
    recall_pc2text: RecallDict