import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, Subset

from .datasets import EmbeddingCache, TrainDataset, train_collate_fn
from .type_defs import ModelID, PathLike


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_split_ids_from_csv(filepath: PathLike) -> tuple[list[ModelID], list[ModelID]]:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    df = pd.read_csv(filepath)
    train_ids = (
        df[df["split"] == "train"]["model_id"].apply(lambda x: f"{x:04d}").tolist()
    )
    val_ids = (
        df[df["split"] == "validation"]["model_id"].apply(lambda x: f"{x:04d}").tolist()
    )
    return train_ids, val_ids


def get_loaders(config: edict) -> tuple[DataLoader, DataLoader]:
    train_ids, val_ids = load_split_ids_from_csv(config.paths.split_filepath)

    init_fn = partial(worker_init_fn, base_seed=config.seed)

    embeddings_cache = EmbeddingCache(config.paths.train_img_embeddings)

    # Создаем датасеты с num_views=24 для stage1
    train_ds = TrainDataset(
        config.paths.train_data_root,
        npoints=config.npoints,
        num_views=24,  # Для stage1 используем 24 views
        pc_augment=True,
        base_seed=config.seed,
        embeddings_cache=embeddings_cache,
    )

    val_ds = TrainDataset(
        config.paths.train_data_root,
        npoints=config.npoints,
        num_views=24,  # Для stage1 используем 24 views
        pc_augment=False,
        base_seed=config.seed,
        embeddings_cache=embeddings_cache,
    )

    model_ids_list = train_ds.model_ids

    # Фильтруем только те IDs которые есть в сплитах
    train_indices = [
        i for i, model_id in enumerate(model_ids_list) if model_id in train_ids
    ]

    val_indices = [
        i for i, model_id in enumerate(model_ids_list) if model_id in val_ids
    ]

    if len(train_indices) == 0:
        raise ValueError("Не найдено ни одного train ID в датасете!")
    if len(val_indices) == 0:
        raise ValueError("Не найдено ни одного validation ID в датасете!")

    train_subset = Subset(train_ds, train_indices)
    val_subset = Subset(val_ds, val_indices)

    # Определяем параметры для DataLoader'ов
    # Для обучения используем оптимизированные параметры
    train_num_workers = config.train_params.train_num_workers
    val_num_workers = config.train_params.val_num_workers

    # prefetch_factor имеет смысл только когда num_workers > 0
    train_prefetch = config.train_params.prefetch_factor if train_num_workers > 0 else None
    val_prefetch = config.train_params.prefetch_factor if val_num_workers > 0 else None

    train_loader = DataLoader(
        train_subset,
        batch_size=config.train_params.batch_size,
        shuffle=True,
        num_workers=train_num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        worker_init_fn=init_fn,
        pin_memory=config.train_params.pin_memory,
        prefetch_factor=train_prefetch,
        persistent_workers=train_num_workers > 0,  # Включаем если есть воркеры
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.train_params.batch_size,
        shuffle=False,
        num_workers=val_num_workers,
        collate_fn=train_collate_fn,
        worker_init_fn=init_fn,
        pin_memory=config.train_params.pin_memory,
        prefetch_factor=val_prefetch,
        persistent_workers=val_num_workers > 0,  # Включаем если есть воркеры
    )

    print("\n✅ DataLoader'ы созданы:")
    print(f"  Train: {len(train_subset)} samples")
    print(f"    - num_workers: {train_num_workers}")
    print(f"    - pin_memory: {config.train_params.pin_memory}")
    print(f"    - prefetch_factor: {train_prefetch}")
    print(f"    - persistent_workers: {train_num_workers > 0}")
    print(f"  Val: {len(val_subset)} samples")
    print(f"    - num_workers: {val_num_workers}")
    print(f"    - pin_memory: {config.train_params.pin_memory}")
    print(f"    - prefetch_factor: {val_prefetch}")
    print(f"    - persistent_workers: {val_num_workers > 0}")

    return train_loader, val_loader