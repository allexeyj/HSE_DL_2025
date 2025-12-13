from typing import cast

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from ReConV2.models.ReCon import ReCon2

from .losses import MultiPositiveContrastiveLoss
from .moe import MoEImgHead


class BasePcEncoder(nn.Module):
    def __init__(self, config: edict):
        super().__init__()
        self.text_ratio = config.text_ratio
        self.pc_encoder_base = ReCon2(config.model)
        self.config = config

    def encode_pc(self, pc: torch.Tensor, normalize: bool) -> torch.Tensor:
        """Кодирует point cloud, совмещая эмбеддинги изображений и текста."""
        img_token, text_token, _, _ = self.pc_encoder_base.forward_features(pc)
        img_pred_feat = torch.mean(img_token, dim=1)
        text_pred_feat = torch.mean(text_token, dim=1)
        pc_feats = img_pred_feat + text_pred_feat * self.text_ratio
        return F.normalize(pc_feats, dim=-1) if normalize else pc_feats


class ImageEncoder(nn.Module):
    """Класс img encoder для инференса"""

    def __init__(self, config: edict) -> None:
        super().__init__()
        self.model = timm.create_model(
            config.model.pretrained_model_name, pretrained=True, num_classes=0
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        print(
            f"✅ Image encoder '{config.model.pretrained_model_name}' loaded (frozen)"
        )

        self.img_proj = MoEImgHead(
            config.model.embed_dim,
            config.emb_dim,
            n_experts=config.train_params.n_experts,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.model(x)
            return cast(torch.Tensor, features)

    def encode_image(self, image: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.model(image)
        image_embeddings = self.img_proj(image_features.float(), normalize=normalize)
        return cast(torch.Tensor, image_embeddings)

    def load_moe_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.img_proj.load_state_dict(checkpoint, strict=True)
        print(f"✅ MoE weights loaded from {checkpoint_path}")


class RetrievalModel(BasePcEncoder):
    """Класс модели для обучения"""

    def __init__(self, config: edict, recon_ckpt: str | None = None) -> None:
        super().__init__(config)

        if recon_ckpt:
            ckpt = torch.load(recon_ckpt, map_location="cpu")
            state_dict = {
                k.replace("module.", ""): v for k, v in ckpt["base_model"].items()
            }
            incompatible = self.pc_encoder_base.load_state_dict(
                state_dict, strict=False
            )
            print(incompatible)
            print("✅ ReCon++ checkpoint loaded with incompatible params above")

        self.img_proj = MoEImgHead(
            config.model.embed_dim,
            config.emb_dim,
            n_experts=config.train_params.n_experts,
        )

        self.contrastive_loss = MultiPositiveContrastiveLoss(
            init_temp=config.train_params.temperature
        )

    def forward(
            self, pc_batch: torch.Tensor, images_list: list[torch.Tensor]
    ) -> torch.Tensor:
        B = pc_batch.size(0)

        pc_embeddings = self.encode_pc(pc_batch, normalize=True)

        # Динамически определяем количество views
        img_feats_per_view = torch.stack(images_list).to(pc_batch.device)
        num_views = img_feats_per_view.size(1)
        assert num_views == 24 #в трейне 24 views

        img_feats_flat = img_feats_per_view.view(-1, img_feats_per_view.size(-1))
        img_feats_proj = self.img_proj(img_feats_flat, normalize=True)
        img_feats_per_view = img_feats_proj.view(B, num_views, -1)

        loss = self.contrastive_loss(pc_embeddings, img_feats_per_view, num_views=num_views)
        return cast(torch.Tensor, loss)


class InferencePcEncoder(BasePcEncoder):
    def __init__(self, config: edict) -> None:
        super().__init__(config)

    def load_pc_encoder_weights(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.pc_encoder_base.load_state_dict(checkpoint, strict=True)
        print(f"✅ PC encoder weights loaded from {checkpoint_path}")