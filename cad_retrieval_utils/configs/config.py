from pathlib import Path

import torch
from easydict import EasyDict as edict

CONFIG = edict()

# --- Конфиг pretrained модели Recon для загрузки ---
CONFIG.model = edict({
    "NAME": "ReCon2",
    "group_size": 32,
    "num_group": 512,
    "mask_ratio": 0.7,
    "mask_type": "rand",
    "embed_dim": 1024,
    "depth": 24,
    "drop_path_rate": 0.2,
    "num_heads": 16,
    "decoder_depth": 4,
    "with_color": True,
    "stop_grad": False,
    "large_embedding": False,
    "img_queries": 13,
    "text_queries": 3,
    "contrast_type": "byol",
    "pretrained_model_name": "eva_large_patch14_336.in22k_ft_in22k_in1k",
})

# --- Общие параметры ---
CONFIG.npoints = 10_000
CONFIG.emb_dim = 1280
CONFIG.save_checkpoint_at_recall_k = [3]
CONFIG.img_size = 336
CONFIG.seed = 42
CONFIG.device = torch.device("cuda:0")
CONFIG.k_list_for_recalls = [1, 2, 3, 5, 10]
CONFIG.text_ratio = 0.3

# --- Параметры инференса ---
CONFIG.infer_img_batch_size = 64
CONFIG.infer_pc_batch_size = 16

# --- Параметры обучения ---
CONFIG.train_params = edict()


# DataLoader параметры
CONFIG.train_params.train_num_workers = 4  # Для train loader
CONFIG.train_params.val_num_workers = 4    # Для val loader
CONFIG.train_params.pin_memory = True      # Ускоряет передачу данных CPU->GPU
CONFIG.train_params.prefetch_factor = 2    # Количество батчей для предзагрузки на воркер



CONFIG.train_params.max_epoch = 40
CONFIG.train_params.batch_size = 32
CONFIG.train_params.lr = 2e-5
CONFIG.train_params.eta_min = 2e-7
CONFIG.train_params.warmup_epochs = 10
CONFIG.train_params.warmup_lr_init = 1e-7
CONFIG.train_params.weight_decay = 0.05
CONFIG.train_params.num_workers = 0 #в инференс
CONFIG.train_params.temperature = 0.07
CONFIG.train_params.use_amp = True
CONFIG.train_params.amp_type = "bf16"
CONFIG.train_params.grad_clip = 10.0
CONFIG.train_params.growth_interval = 500
CONFIG.train_params.growth_factor = 2.0
CONFIG.train_params.backoff_factor = 0.5
CONFIG.train_params.init_scale = 2**16
CONFIG.train_params.scale_grop_factor = 10
CONFIG.train_params.save_weights = True
CONFIG.train_params.n_experts = 8
CONFIG.train_params.wandb_project = "aijc-final-stage1"
CONFIG.train_params.wandb_run_name = "stage1_base_run_A100"

# --- Пути ---
CONFIG.paths = edict()

# Пути для Kaggle
TRAIN_BASE = Path('/root/.cache/kagglehub/datasets/alexhse14/final-train-dataset/versions/1/train')
TEST_BASE = Path('/root/.cache/kagglehub/datasets/alexhse14/test-final/versions/1/test')

CONFIG.paths.train_data_root = TRAIN_BASE
CONFIG.paths.test_data_root = TEST_BASE
CONFIG.paths.recon_ckpt = "/root/.cache/kagglehub/datasets/alexhse14/recon-large/versions/1/best_lvis.pth"
CONFIG.paths.output_dir = Path("./outputs_stage1")
CONFIG.paths.checkpoint_dir = CONFIG.paths.output_dir / "checkpoints"
CONFIG.paths.split_filepath = "splits/base_split.csv"
CONFIG.paths.train_img_embeddings = "/root/.cache/kagglehub/datasets/alexhse14/train-img-embeddings-stage1/versions/1/train_img_embeddings_stage1.npz"