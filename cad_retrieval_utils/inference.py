from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from .datasets import InferenceMeshDataset, InferenceTextDataset
from .evaluation import get_inference_embeddings_text, get_inference_embeddings_mesh
from .models import InferenceTextEncoder, InferencePcEncoder
from .type_defs import CheckpointSpec, EmbeddingArray, PathLike


def load_text_encoder(
    text_proj_path: PathLike,
    text_encoder_path: PathLike | None,
    config: edict
) -> InferenceTextEncoder:
    """Загружает Text Encoder для инференса"""
    text_encoder = InferenceTextEncoder(config).to(config.device)
    text_encoder.load_text_weights(str(text_proj_path), str(text_encoder_path) if text_encoder_path else None)
    text_encoder.eval()
    return text_encoder


def load_pc_encoder(pc_encoder_path: PathLike, config: edict) -> InferencePcEncoder:
    """Загружает PC Encoder для инференса"""
    pc_encoder = InferencePcEncoder(config).to(config.device)
    pc_encoder.load_pc_encoder_weights(str(pc_encoder_path))
    pc_encoder.eval()
    return pc_encoder


def prepare_text2mesh_data(
        config: edict,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Подготавливает данные для задачи Text-to-Mesh"""
    test_root = Path(config.paths.test_data_root)

    queries_text_paths = sorted(
        test_root.joinpath("queries_text_to_mesh").glob("*.txt")
    )
    gallery_mesh_paths = sorted(
        test_root.joinpath("gallery_mesh_for_text").glob("*.stl")
    )

    queries_text_ds = InferenceTextDataset(
        [str(p) for p in queries_text_paths]
    )
    gallery_mesh_ds = InferenceMeshDataset(
        [str(p) for p in gallery_mesh_paths], config.npoints, config.seed
    )

    # DataLoader для текстов (batch_size=1 чтобы обрабатывать по одному)
    queries_loader = DataLoader(
        queries_text_ds,
        batch_size=config.infer_text_batch_size,
        shuffle=False,
        num_workers=0,
    )
    gallery_loader = DataLoader(
        gallery_mesh_ds,
        batch_size=config.infer_pc_batch_size,
        shuffle=False,
        num_workers=0,
    )

    query_ids = [p.stem for p in queries_text_paths]
    gallery_ids = [p.stem for p in gallery_mesh_paths]

    return queries_loader, gallery_loader, query_ids, gallery_ids


def prepare_mesh2text_data(
        config: edict,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """Подготавливает данные для задачи Mesh-to-Text"""
    test_root = Path(config.paths.test_data_root)

    queries_mesh_paths = sorted(
        test_root.joinpath("queries_mesh_to_text").glob("*.stl")
    )
    gallery_text_paths = sorted(
        test_root.joinpath("gallery_text_for_mesh").glob("*.txt")
    )

    queries_mesh_ds = InferenceMeshDataset(
        [str(p) for p in queries_mesh_paths], config.npoints, config.seed
    )
    gallery_text_ds = InferenceTextDataset(
        [str(p) for p in gallery_text_paths]
    )

    queries_loader = DataLoader(
        queries_mesh_ds,
        batch_size=config.infer_pc_batch_size,
        shuffle=False,
        num_workers=0,
    )
    gallery_loader = DataLoader(
        gallery_text_ds,
        batch_size=config.infer_text_batch_size,
        shuffle=False,
        num_workers=0,
    )

    query_ids = [p.stem for p in queries_mesh_paths]
    gallery_ids = [p.stem for p in gallery_text_paths]

    return queries_loader, gallery_loader, query_ids, gallery_ids


def solve_text2mesh(
        queries_loader: DataLoader,
        gallery_loader: DataLoader,
        query_ids: list[str],
        gallery_ids: list[str],
        model_spec: CheckpointSpec,
        config: edict,
) -> pd.DataFrame:
    """Решает задачу Text to Mesh"""
    print("  📝 Получение эмбеддингов для query текстов...")
    text_encoder = load_text_encoder(
        model_spec["text_proj"],
        model_spec.get("text_encoder"),
        config
    )
    query_embs = get_inference_embeddings_text(text_encoder, queries_loader, config)
    del text_encoder
    torch.cuda.empty_cache()

    print("  📦 Получение эмбеддингов для gallery мешей...")
    pc_encoder = load_pc_encoder(model_spec["pc_encoder"], config)
    gallery_embs = get_inference_embeddings_mesh(pc_encoder, gallery_loader, config)
    del pc_encoder
    torch.cuda.empty_cache()

    # Вычисляем сходства и находим top-5
    sims = cosine_similarity(query_embs, gallery_embs)
    top5_indices = np.argsort(sims, axis=1)[:, ::-1][:, :5]

    results = {}
    for i, q_id in enumerate(query_ids):
        top_gallery_ids = [gallery_ids[j] for j in top5_indices[i]]
        results[q_id] = top_gallery_ids

    df = pd.DataFrame(
        list(results.items()), columns=["text_to_mesh_text", "text_to_mesh_mesh"]
    )

    return df.sort_values("text_to_mesh_text").reset_index(drop=True)


def solve_mesh2text(
        queries_loader: DataLoader,
        gallery_loader: DataLoader,
        query_ids: list[str],
        gallery_ids: list[str],
        model_spec: CheckpointSpec,
        config: edict,
) -> pd.DataFrame:
    """Решает задачу Mesh to Text"""
    print("  📦 Получение эмбеддингов для query мешей...")
    pc_encoder = load_pc_encoder(model_spec["pc_encoder"], config)
    query_embs = get_inference_embeddings_mesh(pc_encoder, queries_loader, config)
    del pc_encoder
    torch.cuda.empty_cache()

    print("  📝 Получение эмбеддингов для gallery текстов...")
    text_encoder = load_text_encoder(
        model_spec["text_proj"],
        model_spec.get("text_encoder"),
        config
    )
    gallery_embs = get_inference_embeddings_text(text_encoder, gallery_loader, config)
    del text_encoder
    torch.cuda.empty_cache()

    # Вычисляем сходства и находим top-5
    sims = cosine_similarity(query_embs, gallery_embs)
    top5_indices = np.argsort(sims, axis=1)[:, ::-1][:, :5]

    results = {}
    for i, q_id in enumerate(query_ids):
        top_gallery_ids = [gallery_ids[j] for j in top5_indices[i]]
        results[q_id] = top_gallery_ids

    df = pd.DataFrame(
        list(results.items()), columns=["mesh_to_text_mesh", "mesh_to_text_text"]
    )

    return df.sort_values("mesh_to_text_mesh").reset_index(drop=True)


def make_submission(config: edict) -> None:
    print("\n" + "=" * 60)
    print("🚀 Создание submission файла")
    print("=" * 60)

    model_spec = config.paths.model_spec
    output_path = config.paths.submission_save_file

    print(f"\n📊 Используемая модель:")
    print(f"   Text Projection: {model_spec['text_proj']}")
    if model_spec.get('text_encoder'):
        print(f"   Text Encoder: {model_spec['text_encoder']}")
    print(f"   PC Encoder: {model_spec['pc_encoder']}")

    # Text-to-Mesh
    print("\n📝 → 📦 Обработка задачи: Text-to-Mesh...")
    queries_text_loader, gallery_mesh_loader, query_text_ids, gallery_mesh_ids = (
        prepare_text2mesh_data(config)
    )
    text2mesh_df = solve_text2mesh(
        queries_text_loader,
        gallery_mesh_loader,
        query_text_ids,
        gallery_mesh_ids,
        model_spec,
        config,
    )
    print("✅ Text-to-Mesh завершено")

    # Mesh-to-Text
    print("\n📦 → 📝 Обработка задачи: Mesh-to-Text...")
    queries_mesh_loader, gallery_text_loader, query_mesh_ids, gallery_text_ids = (
        prepare_mesh2text_data(config)
    )
    mesh2text_df = solve_mesh2text(
        queries_mesh_loader,
        gallery_text_loader,
        query_mesh_ids,
        gallery_text_ids,
        model_spec,
        config,
    )
    print("✅ Mesh-to-Text завершено")

    # Объединяем результаты
    submission_df = pd.concat([text2mesh_df, mesh2text_df], axis=1)
    submission_df.insert(0, "id", submission_df.index)

    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Файл для сабмита успешно создан: {output_path}")
    print("=" * 60)