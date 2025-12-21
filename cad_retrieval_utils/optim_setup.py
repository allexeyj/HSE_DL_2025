import torch
import torch.optim as optim
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def build_optimizer_and_scheduler(
    model: torch.nn.Module, config: edict, steps_per_epoch: int
) -> tuple[optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Создает оптимизатор с группами параметров и последовательный планировщик.
    Вручную обрабатывается MoE head, остальное как в https://github.com/qizekun/ShapeLLM/blob/f754b0d488f7187a699a549dd0c5e43b2349f051/ReConV2/tools/builder.py#L63
    """

    base_lr = config.train_params.lr
    weight_decay = config.train_params.weight_decay

    decay = []
    no_decay = []
    finetune_head = []
    finetune_head_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "img_proj" in name and "pc_encoder_base" not in name:
            # Обработка MoE head
            if "bias" in name:
                print(
                    f"Adding to finetune_head_no_decay (lr * {config.train_params.scale_grop_factor}, no decay): {name}"
                )
                finetune_head_no_decay.append(param)
            elif "gate.0" in name:  # LayerNorm в gate
                print(
                    f"Adding to finetune_head_no_decay (lr * {config.train_params.scale_grop_factor}, no decay): {name}"
                )
                finetune_head_no_decay.append(param)
            elif "experts" in name and name.endswith(".weight"):
                print(
                    f"Adding to finetune_head (lr * {config.train_params.scale_grop_factor}): {name}"
                )
                finetune_head.append(param)
            elif "gate.1" in name and name.endswith(".weight"):  # Linear в gate
                print(
                    f"Adding to finetune_head (lr * {config.train_params.scale_grop_factor}): {name}"
                )
                finetune_head.append(param)
            else:
                # Другие параметры MoE (на всякий случай)
                print(
                    f"Adding to finetune_head (lr * {config.train_params.scale_grop_factor}): {name}"
                )
                finetune_head.append(param)
        elif len(param.shape) == 1 or name.endswith(".bias") or "token" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {
            "params": finetune_head,
            "lr": base_lr * config.train_params.scale_grop_factor,
            "weight_decay": weight_decay,
        },
        {
            "params": finetune_head_no_decay,
            "lr": base_lr * config.train_params.scale_grop_factor,
            "weight_decay": 0.0,
        },
        {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
        {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
    ]

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

    num_decay_params = sum(p.numel() for p in decay)
    num_finetune_params = sum(p.numel() for p in finetune_head)
    total_params_with_decay = num_decay_params + num_finetune_params

    print("\n✅ Оптимизатор AdamW и планировщик настроены.")
    print(
        f"   - Группы параметров: {len(finetune_head)} (finetune), {len(decay)} (decay), {len(no_decay)} (no_decay)"
    )
    print(
        f"   - Warmup: {warmup_epochs} эпох ({warmup_steps} шагов), LR от {config.train_params.warmup_lr_init} до {base_lr}"
    )
    print(
        f"   - Cosine Annealing: {max_epochs - warmup_epochs} эпох, LR до {config.train_params.eta_min}"
    )

    print(
        f"   - Finetune params (x{config.train_params.scale_grop_factor} LR): {num_finetune_params:,} (~{num_finetune_params / 1e6:.2f}M)"
    )
    print(
        f"   - Decay params (base LR):   {num_decay_params:,} (~{num_decay_params / 1e6:.2f}M)"
    )
    print(
        f"   - Trainable params (decay+finetune): {total_params_with_decay:,} (~{total_params_with_decay / 1e6:.2f}M)"
    )
    return optimizer, scheduler
