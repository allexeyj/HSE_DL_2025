import numpy as np
import torch
from easydict import EasyDict as edict
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .models import ImageEncoder, InferencePcEncoder, RetrievalModel
from .type_defs import EmbeddingArray, InferenceMode, ModelID, RecallDict, ValidationResult


def compute_recall_at_k(
    query_embs: EmbeddingArray,
    query_ids: list[ModelID],
    gallery_embs: EmbeddingArray,
    gallery_ids: list[ModelID],
    ks: tuple[int, ...] = (1, 5, 10),
) -> RecallDict:
    """
    Вычисляет Recall@k для заданных эмбеддингов и ID.
    """
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
    model: RetrievalModel, loader: DataLoader, config: edict
) -> ValidationResult:
    all_pc_batches: list[np.ndarray] = []
    all_img_batches: list[np.ndarray] = []

    all_pc_ids, all_img_ids = [], []
    total_loss = 0.0

    for batch in tqdm(loader, desc="Collecting embeddings", leave=False):
        pc = batch["pc"].to(config.device)
        imgs_list = batch["images_list"]

        with torch.no_grad():
            loss = model(pc, imgs_list)

        total_loss += loss.item()

        B = pc.size(0)

        pc_emb = model.encode_pc(pc, normalize=False).cpu().numpy()

        all_pc_batches.append(pc_emb)
        all_pc_ids.extend(batch["id"])

        # Получаем количество views динамически
        img_feats_per_view = torch.stack(imgs_list).to(config.device)
        num_views = img_feats_per_view.size(1)
        assert num_views in [10, 12, 24] #10, 12 в тесте, 24 в трейне

        img_feats_flat = img_feats_per_view.view(-1, img_feats_per_view.size(-1))
        img_feats_proj = model.img_proj(img_feats_flat, normalize=True)
        img_feats_per_view = img_feats_proj.view(B, num_views, -1)

        img_emb = img_feats_per_view.mean(dim=1).cpu().numpy()

        all_img_batches.append(img_emb)
        all_img_ids.extend(batch["id"])

    avg_val_loss = total_loss / len(loader)

    stacked_pc_embs = np.vstack(all_pc_batches)
    stacked_img_embs = np.vstack(all_img_batches)

    final_pc_embs = normalize(stacked_pc_embs, axis=1, norm="l2")
    final_img_embs = normalize(stacked_img_embs, axis=1, norm="l2")

    return (
        final_pc_embs,
        final_img_embs,
        all_pc_ids,
        all_img_ids,
        avg_val_loss,
    )


@torch.no_grad()
def get_inference_embeddings(
    model: ImageEncoder | InferencePcEncoder,
    loader: DataLoader,
    mode: InferenceMode,
    config: edict,
) -> EmbeddingArray:
    all_embs = []
    pbar = tqdm(loader, desc=f"Извлечение эмбеддингов ({mode})")
    for batch in pbar:
        batch = batch.to(config.device)
        if mode == "image":
            assert isinstance(model, ImageEncoder)
            embs = model.encode_image(batch, normalize=True)
        else:
            assert isinstance(model, InferencePcEncoder)
            embs = model.encode_pc(batch, normalize=True)

        all_embs.append(embs.cpu().numpy())
    return np.vstack(all_embs)