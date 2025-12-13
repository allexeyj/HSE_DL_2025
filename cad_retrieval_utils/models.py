import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
import open_clip

from ReConV2.models.ReCon import ReCon2

from .losses import ContrastiveLoss


class BasePcEncoder(nn.Module):
    def __init__(self, config: edict):
        super().__init__()
        self.text_ratio = config.text_ratio
        self.pc_encoder_base = ReCon2(config.model)
        self.config = config

    def encode_pc(self, pc: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Кодирует point cloud, совмещая токены изображений и текста."""
        img_token, text_token, _, _ = self.pc_encoder_base.forward_features(pc)
        img_pred_feat = torch.mean(img_token, dim=1)
        text_pred_feat = torch.mean(text_token, dim=1)
        pc_feats = img_pred_feat + text_pred_feat * self.text_ratio
        return F.normalize(pc_feats, dim=-1) if normalize else pc_feats


class TextMeshRetrievalModel(nn.Module):
    """Модель для text-mesh retrieval"""

    def __init__(self, config: edict, recon_ckpt: str) -> None:
        super().__init__()

        # PC encoder
        self.pc_encoder = BasePcEncoder(config)

        # Загружаем веса ReConV2 (обязательно)
        ckpt = torch.load(recon_ckpt, map_location="cpu")
        self.pc_encoder.pc_encoder_base.load_state_dict(ckpt, strict=True)
        print("✅ ReCon++ checkpoint loaded successfully")

        # Замораживаем PC encoder полностью
        for param in self.pc_encoder.parameters():
            param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.pc_encoder.parameters())
        print(f"🔒 PC Encoder полностью заморожен: {frozen_params:,} params (~{frozen_params / 1e6:.2f}M)")

        # Text encoder с размороженными последними 4 слоями
        self.text_encoder = TextEncoder(config)

        # Loss
        self.contrastive_loss = ContrastiveLoss(
            init_temp=config.train_params.temperature
        )

    def forward(self, pc_batch: torch.Tensor, texts: list[str]) -> torch.Tensor:
        """Forward pass для обучения"""
        # Кодируем PC (без градиентов, т.к. заморожен)
        with torch.no_grad():
            pc_embeddings = self.pc_encoder.encode_pc(pc_batch, normalize=True)

        # Кодируем тексты (с градиентами)
        text_embeddings = self.text_encoder.encode_text(texts, normalize=True)

        # Вычисляем loss
        loss = self.contrastive_loss(pc_embeddings.detach(), text_embeddings)
        return loss


class TextEncoder(nn.Module):
    """Text encoder на базе open_clip EVA02-L-14-336 с разморозкой последних 4 слоев"""

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.config = config

        # Загружаем EVA модель и токенизатор
        model, _, _ = open_clip.create_model_and_transforms(
            'EVA02-L-14-336',
            pretrained='merged2b_s6b_b61k'
        )
        self.text_encoder = model
        self.tokenizer = open_clip.get_tokenizer('EVA02-L-14-336')

        # Замораживаем все веса изначально
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Размораживаем последние 4 слоя
        resblocks = self.text_encoder.text.transformer.resblocks
        total_blocks = len(resblocks)

        # Размораживаем последние 4 блока
        for block in resblocks[-4:]:
            for param in block.parameters():
                param.requires_grad = True

        # Также размораживаем финальный LayerNorm
        for param in self.text_encoder.text.ln_final.parameters():
            param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.text_encoder.parameters() if p.requires_grad)
        print(f"✅ Text encoder EVA02-L-14-336 loaded")
        print(f"   🔓 Разморожены последние 4 слоя text transformer")
        print(f"   📊 Trainable params in text encoder: {trainable_params:,} (~{trainable_params / 1e6:.2f}M)")

        # Проекционная голова (всегда обучаемая)
        text_dim = 768  # EVA02-L-14-336 text embedding dim
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, config.emb_dim),
            nn.ReLU(),
            nn.Linear(config.emb_dim, config.emb_dim)
        )

    def encode_text(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        """Кодирует список текстов"""
        # Токенизация
        tokens = self.tokenizer(texts).to(self.config.device)

        # Получаем эмбеддинги
        text_features = self.text_encoder.encode_text(tokens)

        # Проецируем через обучаемую голову
        text_embeddings = self.text_proj(text_features.float())

        return F.normalize(text_embeddings, dim=-1) if normalize else text_embeddings
class InferencePcEncoder(BasePcEncoder):
    """PC encoder для инференса"""

    def __init__(self, config: edict) -> None:
        super().__init__(config)

    def load_pc_encoder_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.pc_encoder_base.load_state_dict(checkpoint, strict=True)
        print(f"✅ PC encoder weights loaded from {checkpoint_path}")


class InferenceTextEncoder(nn.Module):
    """Text encoder для инференса"""

    def __init__(self, config: edict) -> None:
        super().__init__()
        # Создаем полный TextEncoder
        self.encoder = TextEncoder(config)

    def load_text_weights(self, text_proj_path: str, text_encoder_path: str | None = None) -> None:
        """Загружает веса text projection и (опционально) text encoder"""
        # Загружаем text projection (всегда)
        checkpoint = torch.load(text_proj_path, map_location="cpu")
        self.encoder.text_proj.load_state_dict(checkpoint, strict=True)
        print(f"✅ Text projection weights loaded from {text_proj_path}")

        # Загружаем веса text encoder если были разморожены слои
        if text_encoder_path is not None:
            checkpoint = torch.load(text_encoder_path, map_location="cpu")
            # Загружаем только те параметры, которые есть в чекпоинте
            missing, unexpected = self.encoder.text_encoder.load_state_dict(checkpoint, strict=False)
            print(f"✅ Text encoder weights loaded from {text_encoder_path}")
            if missing:
                print(f"   ℹ️  Missing keys (expected, frozen params): {len(missing)}")
            if unexpected:
                print(f"   ⚠️  Unexpected keys: {unexpected}")

    def encode_text(self, texts: list[str], normalize: bool = True) -> torch.Tensor:
        return self.encoder.encode_text(texts, normalize)