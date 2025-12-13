from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from .augmentations import build_img_transforms
from .datasets import InferenceImageDataset, InferenceMeshDataset
from .evaluation import get_inference_embeddings
from .models import ImageEncoder, InferencePcEncoder
from .type_defs import (
    CheckpointSpec,
    EmbeddingArray,
    InferenceMode,
    PathLike,
)


def load_image_encoder(moe_path: PathLike, config: edict) -> ImageEncoder:
    """Загружает только ImageEncoder"""
    moe_path_str = str(moe_path)

    if not Path(moe_path_str).exists():
        raise FileNotFoundError(f"MoE checkpoint not found: {moe_path_str}")

    img_encoder = ImageEncoder(config).to(config.device)
    img_encoder.load_moe_weights(moe_path_str)
    img_encoder.eval()

    return img_encoder


def load_pc_encoder(pc_encoder_path: PathLike, config: edict) -> InferencePcEncoder:
    """Загружает только InferencePcEncoder"""
    pc_encoder_path_str = str(pc_encoder_path)

    if not Path(pc_encoder_path_str).exists():
        raise FileNotFoundError(
            f"PC encoder checkpoint not found: {pc_encoder_path_str}"
        )

    pc_encoder = InferencePcEncoder(config).to(config.device)
    pc_encoder.load_pc_encoder_weights(pc_encoder_path_str)
    pc_encoder.eval()

    return pc_encoder


def get_ensemble_embeddings(
    model_specs: list[CheckpointSpec],
    loader: DataLoader,
    mode: InferenceMode,
    config: edict,
) -> EmbeddingArray:
    """
    Получает усредненные эмбеддинги от ансамбля моделей.
    Загружает только нужную модель для конкретного mode.
    """
    all_model_embeddings = []

    for i, spec in enumerate(model_specs, 1):
        print(f"  📊 Обработка модели {i}/{len(model_specs)}...")

        if mode == "image":
            img_model = load_image_encoder(spec["moe"], config)
            embeddings = get_inference_embeddings(img_model, loader, mode, config)
            del img_model
        else:  # mode == "mesh"
            pc_model = load_pc_encoder(spec["pc_encoder"], config)
            embeddings = get_inference_embeddings(pc_model, loader, mode, config)
            del pc_model

        all_model_embeddings.append(embeddings)
        torch.cuda.empty_cache()

    ensemble_embeddings = np.mean(all_model_embeddings, axis=0)

    ensemble_embeddings = normalize(ensemble_embeddings, axis=1, norm="l2")

    return cast(EmbeddingArray, ensemble_embeddings)


def prepare_img2mesh_data(
    config: edict,
) -> tuple[DataLoader, DataLoader, list[str], list[str]]:
    """
    Подготавливает данные для задачи Image-to-Mesh

    Returns:
        Tuple из (queries_loader, gallery_loader, query_ids, gallery_ids)
    """
    test_root = Path(config.paths.test_data_root)
    img_val_transform = build_img_transforms(config.img_size)

    queries_img_paths = sorted(
        test_root.joinpath("queries_image_to_mesh").glob("*.png")
    )
    gallery_mesh_paths = sorted(
        test_root.joinpath("gallery_mesh_for_image").glob("*.stl")
    )

    queries_img_ds = InferenceImageDataset(
        [str(p) for p in queries_img_paths], img_val_transform
    )
    gallery_mesh_ds = InferenceMeshDataset(
        [str(p) for p in gallery_mesh_paths], config.npoints, config.seed
    )

    queries_loader = DataLoader(
        queries_img_ds,
        batch_size=config.infer_img_batch_size,
        shuffle=False,
        num_workers=0,
    )
    gallery_loader = DataLoader(
        gallery_mesh_ds,
        batch_size=config.infer_pc_batch_size,
        shuffle=False,
        num_workers=0,
    )

    query_ids = [p.stem for p in queries_img_paths]
    gallery_ids = [p.stem for p in gallery_mesh_paths]

    return queries_loader, gallery_loader, query_ids, gallery_ids


def prepare_mesh2img_data(
    config: edict,
) -> tuple[DataLoader, DataLoader, list[str], list[Path]]:
    """
    Подготавливает данные для задачи Mesh-to-Image

    Returns:
        Tuple из (queries_loader, gallery_loader, query_ids, gallery_paths)
    """
    test_root = Path(config.paths.test_data_root)
    img_val_transform = build_img_transforms(config.img_size)

    queries_mesh_paths = sorted(
        test_root.joinpath("queries_mesh_to_image").glob("*.stl")
    )
    gallery_img_paths = sorted(
        test_root.joinpath("gallery_image_for_mesh").glob("*.png")
    )

    queries_mesh_ds = InferenceMeshDataset(
        [str(p) for p in queries_mesh_paths], config.npoints, config.seed
    )
    gallery_img_ds = InferenceImageDataset(
        [str(p) for p in gallery_img_paths], img_val_transform
    )

    queries_loader = DataLoader(
        queries_mesh_ds,
        batch_size=config.infer_pc_batch_size,
        shuffle=False,
        num_workers=0,
    )
    gallery_loader = DataLoader(
        gallery_img_ds,
        batch_size=config.infer_img_batch_size,
        shuffle=False,
        num_workers=0,
    )

    query_ids = [p.stem for p in queries_mesh_paths]

    return queries_loader, gallery_loader, query_ids, gallery_img_paths


def solve_img2mesh(
    queries_loader: DataLoader,
    gallery_loader: DataLoader,
    query_ids: list[str],
    gallery_ids: list[str],
    model_specs: list[CheckpointSpec],
    config: edict,
) -> pd.DataFrame:
    """
    Решает задачу Image to Mesh используя ансамбль моделей
    """
    print("  🖼️ Получение эмбеддингов для query изображений...")
    query_embs = get_ensemble_embeddings(model_specs, queries_loader, "image", config)

    print("  📦 Получение эмбеддингов для gallery мешей...")
    gallery_embs = get_ensemble_embeddings(model_specs, gallery_loader, "mesh", config)

    sims = cosine_similarity(query_embs, gallery_embs)
    top3_indices = np.argsort(sims, axis=1)[:, ::-1][:, :3]

    results = {}
    for i, q_id in enumerate(query_ids):
        top_gallery_ids = [gallery_ids[j] for j in top3_indices[i]]
        results[q_id] = top_gallery_ids

    df = pd.DataFrame(
        list(results.items()), columns=["image_to_mesh_image", "image_to_mesh_mesh"]
    )

    return df.sort_values("image_to_mesh_image").reset_index(drop=True)


def solve_mesh2img(
    queries_loader: DataLoader,
    gallery_loader: DataLoader,
    query_ids: list[str],
    gallery_paths: list[Path],
    model_specs: list[CheckpointSpec],
    config: edict,
) -> pd.DataFrame:
    """
    Решает задачу Mesh to Image используя ансамбль моделей
    """
    print("  📦 Получение эмбеддингов для query мешей...")
    query_embs = get_ensemble_embeddings(model_specs, queries_loader, "mesh", config)

    print("  🖼️ Получение эмбеддингов для gallery изображений...")
    gallery_embs = get_ensemble_embeddings(model_specs, gallery_loader, "image", config)

    # Группируем эмбеддинги по model_id и усредняем
    gallery_img_model_ids = [p.name.split("_")[0] for p in gallery_paths]
    df_gallery_embs = pd.DataFrame(gallery_embs)
    df_gallery_embs["model_id"] = gallery_img_model_ids

    mean_embs_df = df_gallery_embs.groupby("model_id").mean()
    averaged_gallery_embs = mean_embs_df.to_numpy()
    averaged_gallery_ids = mean_embs_df.index.tolist()

    averaged_gallery_embs = normalize(averaged_gallery_embs, axis=1)

    sims = cosine_similarity(query_embs, averaged_gallery_embs)
    top3_indices = np.argsort(sims, axis=1)[:, ::-1][:, :3]

    results = {}
    for i, q_id in enumerate(query_ids):
        top_gallery_ids = [averaged_gallery_ids[j] for j in top3_indices[i]]
        results[q_id] = top_gallery_ids

    df = pd.DataFrame(
        list(results.items()), columns=["mesh_to_image_mesh", "mesh_to_image_image"]
    )

    return df.sort_values("mesh_to_image_mesh").reset_index(drop=True)


def make_submission(
    config: edict,
) -> None:
    print("\n" + "=" * 60)
    print("🚀 Создание submission файла")
    print("=" * 60)

    model_specs = config.paths.ensemble_model_specs
    output_path = config.paths.ensemble_save_file

    print(f"\n📊 Используется ансамбль из {len(model_specs)} моделей")

    print("\n🖼️ → 📦 Обработка задачи: Image-to-Mesh...")

    queries_img_loader, gallery_mesh_loader, query_img_ids, gallery_mesh_ids = (
        prepare_img2mesh_data(config)
    )

    img2mesh_df = solve_img2mesh(
        queries_img_loader,
        gallery_mesh_loader,
        query_img_ids,
        gallery_mesh_ids,
        model_specs,
        config,
    )
    print("✅ Image-to-Mesh завершено")

    print("\n📦 → 🖼️ Обработка задачи: Mesh-to-Image...")

    queries_mesh_loader, gallery_img_loader, query_mesh_ids, gallery_img_paths = (
        prepare_mesh2img_data(config)
    )

    mesh2img_df = solve_mesh2img(
        queries_mesh_loader,
        gallery_img_loader,
        query_mesh_ids,
        gallery_img_paths,
        model_specs,
        config,
    )
    print("✅ Mesh-to-Image завершено")

    # Объединяем результаты
    submission_df = pd.concat([img2mesh_df, mesh2img_df], axis=1)
    submission_df.insert(0, "id", submission_df.index)

    submission_df.to_csv(output_path, index=False)
    print(f"\n✅ Файл для сабмита успешно создан: {output_path}")
    print("=" * 60)
