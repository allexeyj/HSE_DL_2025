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
from .models import RetrievalModel
from .optim_setup import build_optimizer_and_scheduler
from .type_defs import BestMetricInfo, ValidationMetrics


def train_epoch(
    model: RetrievalModel,
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
        images_list = batch["images_list"]

        optimizer.zero_grad()

        if use_amp:
            with autocast(device_type="cuda", dtype=amp_dtype):
                loss = model(pc, images_list)
        else:
            loss = model(pc, images_list)

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
    model: RetrievalModel, loader: DataLoader, global_step: int, config: edict
) -> ValidationMetrics:
    model.eval()

    (
        all_pc,
        all_img,
        all_pc_ids,
        all_img_ids,
        avg_loss,
    ) = get_validation_embeddings(model, loader, config)

    recall_img2pc = compute_recall_at_k(
        all_img, all_img_ids, all_pc, all_pc_ids, config.k_list_for_recalls
    )
    recall_pc2img = compute_recall_at_k(
        all_pc, all_pc_ids, all_img, all_img_ids, config.k_list_for_recalls
    )

    val_results: ValidationMetrics = {
        "loss": avg_loss,
        "recall_img2pc": recall_img2pc,
        "recall_pc2img": recall_pc2img,
    }

    wandb.log({
        "val/loss": val_results["loss"],
        **{
            f"val/recall@{k}/img2pc": v for k, v in val_results["recall_img2pc"].items()
        },
        **{
            f"val/recall@{k}/pc2img": v for k, v in val_results["recall_pc2img"].items()
        },
        "global_step": global_step,
    })
    return val_results


def save_models_and_upd_metrics(
    val_results: ValidationMetrics,
    best_metrics: dict[str, BestMetricInfo],
    config: edict,
    model: RetrievalModel,
    epoch: int,
) -> dict[str, BestMetricInfo]:
    """
    Сохраняет веса MoE и PC encoder отдельно для каждой метрики.
    Также сохраняет эпоху и все метрики на момент лучшего результата.
    """
    recall_img2pc = val_results["recall_img2pc"]
    recall_pc2img = val_results["recall_pc2img"]

    for k in config.save_checkpoint_at_recall_k:
        for task in ["img2pc", "pc2img"]:
            if task == "img2pc":
                recall_value = recall_img2pc[k]
            else:
                recall_value = recall_pc2img[k]

            metric_name = f"best_recall@{k}_{task}"

            current_best = best_metrics.get(metric_name, {"value": 0.0})
            if recall_value > current_best.get("value", 0.0):
                best_metrics[metric_name] = {
                    "value": recall_value,
                    "epoch": epoch,
                    "recall_img2pc": recall_img2pc.copy(),
                    "recall_pc2img": recall_pc2img.copy(),
                }

                if config.train_params.save_weights:
                    checkpoint_dir = Path(config.paths.checkpoint_dir)
                    moe_path = checkpoint_dir / f"moe_recall@{k}_{task}.pth"
                    torch.save(model.img_proj.state_dict(), moe_path)

                    pc_encoder_path = (
                        checkpoint_dir / f"pc_encoder_recall@{k}_{task}.pth"
                    )
                    torch.save(model.pc_encoder_base.state_dict(), pc_encoder_path)

                    print(f"✨ New best Recall@{k} {task}! Value: {recall_value:.4f}")
                    print(f"   Saved MoE to: {moe_path}")
                    print(f"   Saved PC encoder to: {pc_encoder_path}")

    return best_metrics


def print_final_summary(best_metrics: dict[str, BestMetricInfo], config: edict) -> None:
    """Выводит финальную сводку по всем сохраненным чекпоинтам"""
    print("\n" + "=" * 80)
    print("📊 FINAL TRAINING SUMMARY - BEST CHECKPOINTS")
    print("=" * 80)

    for k in config.save_checkpoint_at_recall_k:
        for task in ["img2pc", "pc2img"]:
            metric_name = f"best_recall@{k}_{task}"
            if metric_name in best_metrics:
                info = best_metrics[metric_name]
                print(f"\n🏆 Recall@{k} {task}:")
                print(f"   Best epoch: {info['epoch']}")
                print(f"   Best value: {info['value']:.4f}")
                print(f"   All recalls at epoch {info['epoch']}:")
                print(f"      img2pc: {info['recall_img2pc']}")
                print(f"      pc2img: {info['recall_pc2img']}")
            else:
                print(
                    f"\n❌ Recall@{k} {task}: No checkpoint saved (metric not tracked)"
                )

    print("=" * 80)


def train_loop(config: edict) -> None:
    train_loader, val_loader = get_loaders(config)

    print("\n--- Initializing Model and Optimizer ---")
    model = RetrievalModel(config, recon_ckpt=config.paths.recon_ckpt).to(config.device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, config, len(train_loader)
    )

    if config.train_params.use_amp and config.train_params.amp_type == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise Exception("BF16 не поддерживается на этом GPU!")


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
        for k, v in val_results["recall_img2pc"].items():
            print(f" Val img2pc R@{k}: {v:.4f}")
        for k, v in val_results["recall_pc2img"].items():
            print(f" Val pc2img R@{k}: {v:.4f}")

        best_metrics = save_models_and_upd_metrics(
            val_results, best_metrics, config, model, epoch
        )

    print_final_summary(best_metrics, config)

    wandb.finish()
    print("\n--- Training Finished ---")
