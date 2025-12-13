from typing import cast

import numpy as np
import torch
import torchvision.transforms as T
from torchvision import transforms

from .type_defs import ImageTransform


def build_img_transforms(img_size: int) -> ImageTransform:
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return cast(ImageTransform, transform)


class PointcloudScaleAndTranslate:
    def __init__(
        self,
        scale_low: float = 2.0 / 3.0,
        scale_high: float = 3.0 / 2.0,
        translate_range: float = 0.2,
    ) -> None:
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc: torch.Tensor) -> torch.Tensor:
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(
                low=-self.translate_range, high=self.translate_range, size=[3]
            )

            pc[i, :, 0:3] = torch.mul(
                pc[i, :, 0:3], torch.from_numpy(xyz1).float().to(pc.device)
            ) + torch.from_numpy(xyz2).float().to(pc.device)

        return pc


train_transforms_torch = transforms.Compose([
    PointcloudScaleAndTranslate(),
])
