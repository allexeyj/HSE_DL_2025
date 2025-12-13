import json
import random
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

from .datasets import TextMeshDataset, train_collate_fn
from .type_defs import ModelID, PathLike


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def load_split_ids_from_csv(filepath: PathLike) -> tuple[list[ModelID], list[ModelID]]:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    # ВАЖНО: читаем model_id как строку, сохраняя ведущие нули
    df = pd.read_csv(filepath, dtype={'model_id': str})

    # Если в CSV нет заголовка, добавляем его
    if 'model_id' not in df.columns and 'split' not in df.columns:
        df.columns = ['model_id', 'split']

    train_ids = df[df["split"] == "train"]["model_id"].tolist()
    val_ids = df[df["split"] == "validation"]["model_id"].tolist()

    print(f"📊 Загружено из split файла:")
    print(f"   Train IDs: {len(train_ids)} (примеры: {train_ids[:3]})")
    print(f"   Val IDs: {len(val_ids)} (примеры: {val_ids[:3]})")

    return train_ids, val_ids


def get_loaders(config: edict) -> tuple[DataLoader, DataLoader]:
    # Проверяем существование split файла
    split_path = Path(config.paths.split_filepath)
    if not split_path.exists():
        raise FileNotFoundError(f"Split файл не найден: {split_path}")

    train_ids, val_ids = load_split_ids_from_csv(config.paths.split_filepath)

    # Проверяем что ID не пустые
    assert len(train_ids) > 0, f"Train IDs пустые!"
    assert len(val_ids) > 0, f"Val IDs пустые!"

    # Дополнительная проверка соответствия с captions
    with open(config.paths.captions_file, 'r') as f:
        captions = json.load(f)

    train_ids_found = [tid for tid in train_ids if tid in captions]
    val_ids_found = [vid for vid in val_ids if vid in captions]

    print(f"✅ Найдено в captions:")
    print(f"   Train: {len(train_ids_found)}/{len(train_ids)}")
    print(f"   Val: {len(val_ids_found)}/{len(val_ids)}")

    if len(train_ids_found) == 0:
        print("❌ Проблема с форматом ID!")
        print(f"   Примеры train IDs из split: {train_ids[:5]}")
        print(f"   Примеры ключей из captions: {list(captions.keys())[:5]}")
        raise ValueError("Не найдено соответствий между split и captions")

    init_fn = partial(worker_init_fn, base_seed=config.seed)

    # Создаем датасеты с конкретными ID
    train_ds = TextMeshDataset(
        config.paths.train_data_root,
        config.paths.captions_file,
        npoints=config.npoints,
        model_ids=train_ids,
        pc_augment=True,
        base_seed=config.seed,
    )

    val_ds = TextMeshDataset(
        config.paths.train_data_root,
        config.paths.captions_file,
        npoints=config.npoints,
        model_ids=val_ids,
        pc_augment=False,
        base_seed=config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train_params.batch_size,
        shuffle=True,
        num_workers=config.train_params.num_workers,
        collate_fn=train_collate_fn,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train_params.batch_size,
        shuffle=False,
        num_workers=config.train_params.num_workers,
        collate_fn=train_collate_fn,
        worker_init_fn=init_fn,
    )

    print("\n✅ DataLoader'ы созданы:")
    print(f"  Train: {len(train_ds)} samples")
    print(f"  Val: {len(val_ds)} samples")

    return train_loader, val_loader