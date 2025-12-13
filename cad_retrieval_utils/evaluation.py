import numpy as np
import torch
from easydict import EasyDict as edict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import TextMeshRetrievalModel, InferenceTextEncoder, InferencePcEncoder
from .type_defs import EmbeddingArray, ModelID, RecallDict, ValidationResult


def compute_recall_at_k(
        query_embs: EmbeddingArray,
        query_ids: list[ModelID],
        gallery_embs: EmbeddingArray,
        gallery_ids: list[ModelID],
        ks: tuple[int, ...] = (1, 5, 10),
) -> RecallDict:
    """Вычисляет Recall@k для заданных эмбеддингов и ID"""
    sims = cosine_similarity(query_embs, gallery_embs)
    recalls = {}
    for k in ks:
        correct = 0
        for i, qid in enumerate(query_ids):
            gt = qid
            topk_idx = sims[i].argsort()[::-1][:k]
            topk_ids = [gallery_ids[j] for j in topk_idx]
            if gt in topk_ids:
                correct += 1
        recalls[k] = correct / len(query_ids)
    return recalls


@torch.no_grad()
def get_validation_embeddings(
        model: TextMeshRetrievalModel, loader: DataLoader, config: edict
) -> ValidationResult:
    all_pc_batches: list[np.ndarray] = []
    all_text_batches: list[np.ndarray] = []

    all_pc_ids, all_text_ids = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Collecting embeddings", leave=False):
        pc = batch["pc"].to(config.device)
        texts = batch["texts"]

        with torch.no_grad():
            loss = model(pc, texts)

        total_loss += loss.item()

        # Получаем эмбеддинги
        pc_emb = model.pc_encoder.encode_pc(pc, normalize=False).cpu().numpy()
        text_emb = model.text_encoder.encode_text(texts, normalize=False).cpu().numpy()

        all_pc_batches.append(pc_emb)
        all_text_batches.append(text_emb)
        all_pc_ids.extend(batch["id"])
        all_text_ids.extend(batch["id"])

    avg_val_loss = total_loss / len(loader)

    # Объединяем и нормализуем
    stacked_pc_embs = np.vstack(all_pc_batches)
    stacked_text_embs = np.vstack(all_text_batches)

    final_pc_embs = normalize(stacked_pc_embs, axis=1, norm="l2")
    final_text_embs = normalize(stacked_text_embs, axis=1, norm="l2")

    return (
        final_pc_embs,
        final_text_embs,
        all_pc_ids,
        all_text_ids,
        avg_val_loss,
    )


@torch.no_grad()
def get_inference_embeddings_text(
        model: InferenceTextEncoder, loader: DataLoader, config: edict
) -> EmbeddingArray:
    """Получает эмбеддинги текстов для инференса"""
    all_embs = []
    pbar = tqdm(loader, desc="Извлечение text эмбеддингов")

    for batch in pbar:
        # batch - это список строк
        embs = model.encode_text(batch, normalize=True)
        all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs)


@torch.no_grad()
def get_inference_embeddings_mesh(
        model: InferencePcEncoder, loader: DataLoader, config: edict
) -> EmbeddingArray:
    """Получает эмбеддинги мешей для инференса"""
    all_embs = []
    pbar = tqdm(loader, desc="Извлечение mesh эмбеддингов")

    for batch in pbar:
        batch = batch.to(config.device)
        embs = model.encode_pc(batch, normalize=True)
        all_embs.append(embs.cpu().numpy())

    return np.vstack(all_embs)