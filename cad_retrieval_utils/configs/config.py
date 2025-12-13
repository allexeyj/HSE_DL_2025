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
CONFIG.seed = 42
CONFIG.device = torch.device("cuda:0")
CONFIG.k_list_for_recalls = [1, 2, 3, 5, 10]
CONFIG.text_ratio = 0.3

# --- Параметры Text Encoder ---
# Сколько последних слоев text transformer разморозить (0 = все заморожены)
# В EVA02-L-14-336 всего 12 ResidualAttentionBlock'ов
CONFIG.unfreeze_text_layers = 4  # Например, разморозить последние 2 слоя

# --- Параметры инференса ---
CONFIG.infer_text_batch_size = 32
CONFIG.infer_pc_batch_size = 16

# --- Параметры обучения ---
CONFIG.train_params = edict()
CONFIG.train_params.save_avg_recall = True
CONFIG.train_params.max_epoch = 40
CONFIG.train_params.batch_size = 32
CONFIG.train_params.lr = 2e-5
CONFIG.train_params.eta_min = 2e-7
CONFIG.train_params.warmup_epochs = 10
CONFIG.train_params.warmup_lr_init = 1e-7
CONFIG.train_params.weight_decay = 0.05
CONFIG.train_params.num_workers = 0
CONFIG.train_params.temperature = 0.07
CONFIG.train_params.use_amp = True
CONFIG.train_params.amp_type = "bf16"
CONFIG.train_params.grad_clip = 10.0
CONFIG.train_params.growth_interval = 500
CONFIG.train_params.growth_factor = 2.0
CONFIG.train_params.backoff_factor = 0.5
CONFIG.train_params.init_scale = 2**16
CONFIG.train_params.scale_grop_factor = 1  # Множитель LR для text_proj
CONFIG.train_params.scale_text_encoder_factor = 1  # Множитель LR для размороженных слоев text encoder
CONFIG.train_params.save_weights = True
CONFIG.train_params.wandb_project = "aijc-final-stage2"
CONFIG.train_params.wandb_run_name = "stage2_base_run_A100"

# --- Пути ---
CONFIG.paths = edict()
DATA_ROOT = Path("/root/.cache/kagglehub/datasets/alexeyj/aijc-dataset/versions/1/aijc-dataset")
CONFIG.paths.train_data_root = DATA_ROOT / "train"
CONFIG.paths.test_data_root = DATA_ROOT / "test"
CONFIG.paths.captions_file =  DATA_ROOT /  "train" / "captions.json"
CONFIG.paths.recon_ckpt = "/root/.cache/kagglehub/datasets/alexhse14/stage1-model/versions/1/pc_encoder_recall3_img2pc.pth"
CONFIG.paths.output_dir = Path("./outputs")
CONFIG.paths.checkpoint_dir = CONFIG.paths.output_dir / "checkpoints"
CONFIG.paths.split_filepath = "/aijc/splits/text_split.csv"
CONFIG.paths.submission_save_file = Path("./submission.csv")

CONFIG.paths.model_spec = {
    "text_proj": CONFIG.paths.checkpoint_dir / "text_proj_recall@5_pc2text.pth",
    "text_encoder": CONFIG.paths.checkpoint_dir / "text_encoder_recall@5_pc2text.pth",  # Новый путь
    "pc_encoder": CONFIG.paths.checkpoint_dir / "pc_encoder_recall@5_pc2text.pth"
}