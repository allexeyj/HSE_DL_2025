from pathlib import Path

import torch
import wandb
from easydict import EasyDict as edict
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data_setup import get_loaders
from .evaluation import compute_recall_at_k, get_validation_embeddings
from .models import TextMeshRetrievalModel
from .optim_setup import build_optimizer_and_scheduler
from .type_defs import BestMetricInfo, ValidationMetrics


def train_epoch(
        model: TextMeshRetrievalModel,
        loader: DataLoader,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        scaler: GradScaler | None,
        global_step: int,
        config: edict,
) -> tuple[float, int]:
    model.train()
    total_loss_avg = 0.0
    pbar = tqdm(loader, desc="Training", leave=False)

    use_amp = config.train_params.use_amp
    amp_dtype = None

    if use_amp:
        amp_type = config.train_params.amp_type
        assert amp_type in ("fp16", "bf16"), f"Неподдерживаемый amp_type: {amp_type}"
        amp_dtype = torch.float16 if amp_type == "fp16" else torch.bfloat16

    for batch_idx, batch in enumerate(pbar):
        pc = batch["pc"].to(config.device)
        texts = batch["texts"]

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(pc, texts)
        else:
            loss = model(pc, texts)

        if use_amp and config.train_params.amp_type == "fp16":
            assert scaler is not None, "GradScaler должен быть инициализирован для fp16"
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train_params.grad_clip
            )

            scaler.step(optimizer)
            scaler.update()
        else:
            # Для bf16 и fp32 - обычный backward
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train_params.grad_clip
            )

            optimizer.step()

        scheduler.step()

        total_loss_avg += loss.item()
        global_step += 1

        log_dict: dict[str, float | int] = {
            "global_step": global_step,
            "train/loss": loss.item(),
            "train/lr": optimizer.param_groups[0]["lr"],
        }

        if use_amp and config.train_params.amp_type == "fp16" and scaler is not None:
            log_dict["train/scale"] = scaler.get_scale()

        wandb.log(log_dict)

        postfix_dict: dict[str, object] = {
            "loss": f"{loss.item():.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
        }
        if use_amp and config.train_params.amp_type == "fp16" and scaler is not None:
            postfix_dict["scale"] = f"{scaler.get_scale():.1f}"

        pbar.set_postfix(postfix_dict)

    avg_loss = total_loss_avg / len(loader)
    return avg_loss, global_step


def val_epoch(
        model: TextMeshRetrievalModel, loader: DataLoader, global_step: int, config: edict
) -> ValidationMetrics:
    model.eval()

    (
        all_pc,
        all_text,
        all_pc_ids,
        all_text_ids,
        avg_loss,
    ) = get_validation_embeddings(model, loader, config)

    recall_text2pc = compute_recall_at_k(
        all_text, all_text_ids, all_pc, all_pc_ids, config.k_list_for_recalls
    )
    recall_pc2text = compute_recall_at_k(
        all_pc, all_pc_ids, all_text, all_text_ids, config.k_list_for_recalls
    )

    val_results: ValidationMetrics = {
        "loss": avg_loss,
        "recall_text2pc": recall_text2pc,
        "recall_pc2text": recall_pc2text,
    }

    # Добавляем логирование среднего recall@5
    if 5 in recall_text2pc and 5 in recall_pc2text:
        avg_recall_5 = (recall_text2pc[5] + recall_pc2text[5]) / 2.0
        wandb.log({
            "val/recall@5/avg": avg_recall_5,
            "global_step": global_step,
        })

    wandb.log({
        "val/loss": val_results["loss"],
        **{
            f"val/recall@{k}/text2pc": v for k, v in val_results["recall_text2pc"].items()
        },
        **{
            f"val/recall@{k}/pc2text": v for k, v in val_results["recall_pc2text"].items()
        },
        "global_step": global_step,
    })
    return val_results


def save_models_and_upd_metrics(
        val_results: ValidationMetrics,
        best_metrics: dict[str, BestMetricInfo],
        config: edict,
        model: TextMeshRetrievalModel,
        epoch: int,
) -> dict[str, BestMetricInfo]:
    """
    Сохраняет веса text_proj и text_encoder.
    PC encoder не сохраняется, т.к. всегда используется из pretrained checkpoint.
    """
    recall_text2pc = val_results["recall_text2pc"]
    recall_pc2text = val_results["recall_pc2text"]

    # Сохранение по отдельным метрикам
    for k in config.save_checkpoint_at_recall_k:
        for task in ["text2pc", "pc2text"]:
            if task == "text2pc":
                recall_value = recall_text2pc[k]
            else:
                recall_value = recall_pc2text[k]

            metric_name = f"best_recall@{k}_{task}"

            current_best = best_metrics.get(metric_name, {"value": 0.0})
            if recall_value > current_best.get("value", 0.0):
                best_metrics[metric_name] = {
                    "value": recall_value,
                    "epoch": epoch,
                    "recall_text2pc": recall_text2pc.copy(),
                    "recall_pc2text": recall_pc2text.copy(),
                }

                if config.train_params.save_weights:
                    checkpoint_dir = Path(config.paths.checkpoint_dir)

                    # Сохраняем text_proj
                    text_proj_path = checkpoint_dir / f"text_proj_recall@{k}_{task}.pth"
                    torch.save(model.text_encoder.text_proj.state_dict(), text_proj_path)

                    # Сохраняем размороженные слои text_encoder
                    text_encoder_path = checkpoint_dir / f"text_encoder_recall@{k}_{task}.pth"
                    # Сохраняем только обучаемые параметры text_encoder
                    text_encoder_state = {
                        k: v for k, v in model.text_encoder.text_encoder.state_dict().items()
                        if any(p.requires_grad for n, p in model.text_encoder.text_encoder.named_parameters() if n == k)
                    }
                    torch.save(text_encoder_state, text_encoder_path)

                    print(f"✨ New best Recall@{k} {task}! Value: {recall_value:.4f}")
                    print(f"   Saved text_proj to: {text_proj_path}")
                    print(f"   Saved text_encoder to: {text_encoder_path}")
                    print(f"   PC encoder: using pretrained from {config.paths.recon_ckpt}")

    # Сохранение по среднему recall
    if config.train_params.get('save_avg_recall', False):
        for k in config.save_checkpoint_at_recall_k:
            if k in recall_text2pc and k in recall_pc2text:
                avg_recall = (recall_text2pc[k] + recall_pc2text[k]) / 2.0
                metric_name = f"best_recall@{k}_avg"

                current_best = best_metrics.get(metric_name, {"value": 0.0})
                if avg_recall > current_best.get("value", 0.0):
                    best_metrics[metric_name] = {
                        "value": avg_recall,
                        "epoch": epoch,
                        "recall_text2pc": recall_text2pc.copy(),
                        "recall_pc2text": recall_pc2text.copy(),
                    }

                    if config.train_params.save_weights:
                        checkpoint_dir = Path(config.paths.checkpoint_dir)

                        # Сохраняем text_proj
                        text_proj_path = checkpoint_dir / f"text_proj_recall@{k}_avg.pth"
                        torch.save(model.text_encoder.text_proj.state_dict(), text_proj_path)

                        # Сохраняем размороженные слои text_encoder
                        text_encoder_path = checkpoint_dir / f"text_encoder_recall@{k}_avg.pth"
                        text_encoder_state = {
                            k: v for k, v in model.text_encoder.text_encoder.state_dict().items()
                            if any(p.requires_grad for n, p in model.text_encoder.text_encoder.named_parameters() if n == k)
                        }
                        torch.save(text_encoder_state, text_encoder_path)

                        print(f"✨ New best AVG Recall@{k}! Value: {avg_recall:.4f}")
                        print(f"   Text2PC: {recall_text2pc[k]:.4f}, PC2Text: {recall_pc2text[k]:.4f}")
                        print(f"   Saved text_proj to: {text_proj_path}")
                        print(f"   Saved text_encoder to: {text_encoder_path}")

    return best_metrics


def print_final_summary(best_metrics: dict[str, BestMetricInfo], config: edict) -> None:
    """Выводит финальную сводку по всем сохраненным чекпоинтам"""
    print("\n" + "=" * 80)
    print("📊 FINAL TRAINING SUMMARY - BEST CHECKPOINTS")
    print("=" * 80)

    # Сводка по отдельным метрикам
    for k in config.save_checkpoint_at_recall_k:
        for task in ["text2pc", "pc2text"]:
            metric_name = f"best_recall@{k}_{task}"
            if metric_name in best_metrics:
                info = best_metrics[metric_name]
                print(f"\n🏆 Recall@{k} {task}:")
                print(f"   Best epoch: {info['epoch']}")
                print(f"   Best value: {info['value']:.4f}")
                print(f"   All recalls at epoch {info['epoch']}:")
                print(f"      text2pc: {info['recall_text2pc']}")
                print(f"      pc2text: {info['recall_pc2text']}")
            else:
                print(
                    f"\n❌ Recall@{k} {task}: No checkpoint saved (metric not tracked)"
                )

    # Сводка по средним метрикам
    if config.train_params.get('save_avg_recall', False):
        for k in config.save_checkpoint_at_recall_k:
            metric_name = f"best_recall@{k}_avg"
            if metric_name in best_metrics:
                info = best_metrics[metric_name]
                print(f"\n🏆 AVG Recall@{k}:")
                print(f"   Best epoch: {info['epoch']}")
                print(f"   Best avg value: {info['value']:.4f}")
                print(f"   Individual values at epoch {info['epoch']}:")
                print(f"      text2pc R@{k}: {info['recall_text2pc'][k]:.4f}")
                print(f"      pc2text R@{k}: {info['recall_pc2text'][k]:.4f}")

    print("=" * 80)


def train_loop(config: edict) -> None:
    if config.train_params.use_amp and config.train_params.amp_type == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise Exception("BF16 не поддерживается на этом GPU!")

    train_loader, val_loader = get_loaders(config)

    print("\n--- Initializing Model and Optimizer ---")
    model = TextMeshRetrievalModel(config, recon_ckpt=config.paths.recon_ckpt).to(config.device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, len(train_loader)
    )

    scaler = None
    if config.train_params.use_amp and config.train_params.amp_type == "fp16":
        scaler = GradScaler(
            "cuda",
            growth_interval=config.train_params.growth_interval,
            growth_factor=config.train_params.growth_factor,
            backoff_factor=config.train_params.backoff_factor,
            init_scale=config.train_params.init_scale,
            enabled=True,
        )
        print("✅ AMP включен в режиме fp16 (с GradScaler)")
    elif config.train_params.use_amp and config.train_params.amp_type == "bf16":
        print("✅ AMP включен в режиме bf16 (без GradScaler)")
    else:
        print("✅ AMP выключен, обучение в fp32")

    print("\n--- Starting Training Loop ---")
    best_metrics: dict[str, BestMetricInfo] = {}
    global_step = 0

    for epoch in range(1, config.train_params.max_epoch + 1):
        print(f"\n—— Epoch {epoch}/{config.train_params.max_epoch} ——")

        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, global_step, config
        )

        val_results = val_epoch(model, val_loader, global_step, config)

        print(f"\nEpoch {epoch} summary:")
        print(f" Train loss: {train_loss:.4f}")
        print(f" Val loss: {val_results['loss']:.4f}")
        for k, v in val_results["recall_text2pc"].items():
            print(f" Val text2pc R@{k}: {v:.4f}")
        for k, v in val_results["recall_pc2text"].items():
            print(f" Val pc2text R@{k}: {v:.4f}")

        # Печатаем средний recall@5 если есть
        if 5 in val_results["recall_text2pc"] and 5 in val_results["recall_pc2text"]:
            avg_r5 = (val_results["recall_text2pc"][5] + val_results["recall_pc2text"][5]) / 2.0
            print(f" Val AVG R@5: {avg_r5:.4f}")

        best_metrics = save_models_and_upd_metrics(
            val_results, best_metrics, config, model, epoch
        )

    print_final_summary(best_metrics, config)

    wandb.finish()
    print("\n--- Training Finished ---")