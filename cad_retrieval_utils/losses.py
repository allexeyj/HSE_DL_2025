import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiPositiveContrastiveLoss(nn.Module):
    """
    Multi-positive InfoNCE loss для случая с несколькими видами изображений.
    Учитывает, что все виды одного объекта являются позитивными примерами.
    """

    logit_scale: torch.Tensor

    def __init__(self, init_temp: float = 0.07) -> None:
        super().__init__()
        self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / init_temp)))

    def forward(
        self, pc_emb: torch.Tensor, img_emb: torch.Tensor, num_views: int
    ) -> torch.Tensor:
        """
        Args:
            pc_emb: [B, D] - нормализованные эмбеддинги облаков точек
            img_emb: [B, V, D] - нормализованные эмбеддинги изображений (V видов)
            num_views: количество видов V

        Returns:
            Средний loss между pc->img и img->pc направлениями
        """
        B, V, D = img_emb.shape
        assert V == num_views, f"Expected {num_views} views, got {V}"

        img_flat = img_emb.reshape(B * V, D)  # [B*V, D]

        logits_pc2img = pc_emb @ img_flat.t()  # [B, B*V]
        logits_img2pc = img_flat @ pc_emb.t()  # [B*V, B]

        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_pc2img = logits_pc2img * logit_scale
        logits_img2pc = logits_img2pc * logit_scale

        # === PC -> Image loss (multiple positives) ===
        # Для каждого облака точек (PC) позитивными примерами являются все его 26 проекций-изображений.
        # Поэтому используется маска и logsumexp для агрегации логитов по всем позитивным парам.
        device = pc_emb.device
        pos_mask_pc2img = torch.zeros(B, B * V, device=device, dtype=torch.bool)
        for i in range(B):
            pos_mask_pc2img[i, i * V : (i + 1) * V] = True

        # Вычисляем loss: -log(sum_pos exp(logit) / sum_all exp(logit))
        # Используем logsumexp для численной стабильности
        logsumexp_all = logits_pc2img.logsumexp(dim=1)
        logsumexp_pos = logits_pc2img.masked_fill(
            ~pos_mask_pc2img, float("-inf")
        ).logsumexp(dim=1)
        loss_pc2img = -(logsumexp_pos - logsumexp_all).mean()

        # === Image -> PC loss (single positive) ===
        # Для каждого изображения позитивным примером является только одно исходное облако точек.
        # Это позволяет использовать стандартный cross_entropy, где таргеты - индексы правильных PC.
        targets_img2pc = torch.arange(B, device=device).repeat_interleave(V)  # [B*V]
        loss_img2pc = F.cross_entropy(logits_img2pc, targets_img2pc)

        return (loss_pc2img + loss_img2pc) / 2.0
