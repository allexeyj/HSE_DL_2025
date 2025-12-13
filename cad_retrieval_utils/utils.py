import importlib.util
import os
import random
from pathlib import Path

import numpy as np
import torch
import wandb
from easydict import EasyDict as edict


def load_config(config_path: str) -> edict:
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Не удалось загрузить модуль из {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"{config_path} должен содержать переменную CONFIG")
    return module.CONFIG


def init_environment(config: edict) -> None:
    SEED = config.seed

    # Все используют один и тот же базовый сид
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    # CuDNN настройки
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Отключение TF32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Детерминированные алгоритмы
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # Создание директорий
    Path(config.paths.output_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print(f"✅ Детерминированная среда установлена с seed = {SEED}")


def wandb_init(config: edict) -> None:
    wandb.init(
        project=config.train_params.wandb_project,
        name=config.train_params.wandb_run_name,
        config=config,
    )

    wandb.define_metric("global_step")

    wandb.define_metric("train/loss_*", step_metric="global_step", summary="min")
    wandb.define_metric("train/weight_*", step_metric="global_step", summary="last")
    wandb.define_metric("train/lr", step_metric="global_step", summary="last")

    wandb.define_metric("val/loss_*", step_metric="global_step", summary="min")

    for k in config.k_list_for_recalls:
        wandb.define_metric(f"val/recall@{k}/text2pc", step_metric="global_step", summary="max")
        wandb.define_metric(f"val/recall@{k}/pc2text", step_metric="global_step", summary="max")

    # Добавляем метрику для среднего recall
    wandb.define_metric("val/recall@5/avg", step_metric="global_step", summary="max")