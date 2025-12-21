import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Стандартный симметричный InfoNCE loss для text-mesh пар
    """

    def __init__(self, init_temp: float = 0.07) -> None:
        super().__init__()
        self.register_buffer("logit_scale", torch.log(torch.tensor(1.0 / init_temp)))

    def forward(
            self, pc_emb: torch.Tensor, text_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pc_emb: [B, D] - нормализованные эмбеддинги облаков точек
            text_emb: [B, D] - нормализованные эмбеддинги текстов

        Returns:
            Средний loss между pc->text и text->pc направлениями
        """
        B = pc_emb.shape[0]

        # Вычисляем логиты
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_pc2text = (pc_emb @ text_emb.t()) * logit_scale  # [B, B]
        logits_text2pc = logits_pc2text.t()  # [B, B]

        # Таргеты - диагональные элементы (i-й PC соответствует i-му тексту)
        targets = torch.arange(B, device=pc_emb.device)

        # Cross-entropy loss в обоих направлениях
        loss_pc2text = F.cross_entropy(logits_pc2text, targets)
        loss_text2pc = F.cross_entropy(logits_text2pc, targets)

        return (loss_pc2text + loss_text2pc) / 2.0