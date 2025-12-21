import torch
import torch.optim as optim
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer_and_scheduler(
        model: torch.nn.Module, config: edict, steps_per_epoch: int
) -> tuple[optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Создает оптимизатор для обучения только:
    - text_proj с scale_grop_factor
    - размороженных слоев text_encoder с scale_text_encoder_factor
    PC encoder всегда заморожен и не участвует в оптимизации
    """

    base_lr = config.train_params.lr
    weight_decay = config.train_params.weight_decay

    # Группы параметров
    finetune_head = []
    finetune_head_no_decay = []
    text_encoder_params = []
    text_encoder_params_no_decay = []

    # Счетчик замороженных параметров
    frozen_params_count = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_params_count += param.numel()
            continue

        # Text projection head
        if "text_proj" in name:
            if "bias" in name:
                finetune_head_no_decay.append(param)
            else:
                finetune_head.append(param)

        # Размороженные слои text encoder
        elif "text_encoder.text_encoder" in name:
            if "bias" in name or len(param.shape) == 1:
                text_encoder_params_no_decay.append(param)
            else:
                text_encoder_params.append(param)

    param_groups = [
        {
            "params": finetune_head,
            "lr": base_lr * config.train_params.scale_grop_factor,
            "weight_decay": weight_decay,
            "name": "text_proj"
        },
        {
            "params": finetune_head_no_decay,
            "lr": base_lr * config.train_params.scale_grop_factor,
            "weight_decay": 0.0,
            "name": "text_proj_no_decay"
        },
        {
            "params": text_encoder_params,
            "lr": base_lr * config.train_params.scale_text_encoder_factor,
            "weight_decay": weight_decay,
            "name": "text_encoder"
        },
        {
            "params": text_encoder_params_no_decay,
            "lr": base_lr * config.train_params.scale_text_encoder_factor,
            "weight_decay": 0.0,
            "name": "text_encoder_no_decay"
        },
    ]

    # Фильтруем пустые группы
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    warmup_epochs = config.train_params.warmup_epochs
    max_epochs = config.train_params.max_epoch

    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=config.train_params.warmup_lr_init / base_lr,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config.train_params.eta_min
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # Подсчет параметров
    num_finetune_params = sum(p.numel() for p in finetune_head + finetune_head_no_decay)
    num_text_encoder_params = sum(p.numel() for p in text_encoder_params + text_encoder_params_no_decay)
    total_trainable_params = num_finetune_params + num_text_encoder_params

    print("\n✅ Оптимизатор AdamW и планировщик настроены.")
    print(f"   - Обучаемые параметры:")
    print(f"     • Text Projection: {num_finetune_params:,} params (lr x{config.train_params.scale_grop_factor})")
    print(
        f"     • Text Encoder (last 4 layers): {num_text_encoder_params:,} params (lr x{config.train_params.scale_text_encoder_factor})")
    print(f"     • PC Encoder: FROZEN ({frozen_params_count:,} params)")
    print(f"   - Warmup: {warmup_epochs} эпох ({warmup_steps} шагов)")
    print(f"   - Cosine Annealing: {max_epochs - warmup_epochs} эпох")
    print(f"   - Total trainable params: {total_trainable_params:,} (~{total_trainable_params / 1e6:.2f}M)")

    return optimizer, scheduler