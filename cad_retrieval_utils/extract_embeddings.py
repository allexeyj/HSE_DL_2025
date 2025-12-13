import json
from pathlib import Path
from typing import cast

import numpy as np
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from .augmentations import build_img_transforms


class ImageDatasetForEmbeddings(Dataset):
    def __init__(self, image_dir: Path, captions_path: Path):
        self.image_dir = image_dir

        # Загружаем captions чтобы понять какие модели имеют текст
        with open(captions_path, 'r') as f:
            captions = json.load(f)

        # IDs моделей с текстом (0-524)
        text_model_ids = set(captions.keys())

        # Невалидные IDs которые надо пропустить
        self.invalid_ids = {'0654', '2764', '3818', '4263'}

        # Собираем все файлы изображений
        all_images = sorted(self.image_dir.glob("*.png"))

        # Фильтруем только модели 525+ (без текста) и не невалидные
        self.image_paths = []
        for img_path in all_images:
            model_id = img_path.stem.split('_')[0]

            # Пропускаем невалидные
            if model_id in self.invalid_ids:
                continue

            # Берем только модели без текста (525+)
            if model_id not in text_model_ids:
                self.image_paths.append(img_path)

        print(f"✅ Найдено {len(self.image_paths)} изображений для извлечения эмбеддингов")
        print(f"   (модели 525+, исключая невалидные)")

        # Проверяем что у всех моделей по 24 views
        model_views = {}
        for path in self.image_paths:
            model_id = path.stem.split('_')[0]
            if model_id not in model_views:
                model_views[model_id] = []
            view_idx = int(path.stem.split('_')[1])
            model_views[model_id].append(view_idx)

        # Проверка
        for model_id, views in model_views.items():
            if len(views) != 24:
                raise Exception(f"⚠️ Модель {model_id} имеет {len(views)} views вместо 24!")

        self.transform = build_img_transforms(336)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)
        return img_tensor, img_path.name


def extract_and_save_embeddings(
        train_images_dir: Path,
        captions_path: Path,
        output_path: Path,
        model_name: str = "eva_large_patch14_336.in22k_ft_in22k_in1k",
        batch_size: int = 64,
        device: str = "cuda",
):
    """Извлекает эмбеддинги для всех изображений моделей 525+ и сохраняет в npz"""

    # Создаем датасет
    dataset = ImageDatasetForEmbeddings(train_images_dir, captions_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Загружаем модель
    print(f"\n📦 Загрузка модели {model_name}...")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()

    # Извлекаем эмбеддинги
    all_embeddings = []
    all_filenames = []

    print("\n🔄 Извлечение эмбеддингов...")
    with torch.no_grad():
        for batch_imgs, batch_names in tqdm(dataloader, desc="Extracting embeddings"):
            batch_imgs = batch_imgs.to(device)
            embeddings = model(batch_imgs)

            all_embeddings.append(embeddings.cpu().numpy())
            all_filenames.extend(batch_names)

    # Объединяем все эмбеддинги
    embeddings_array = np.vstack(all_embeddings)
    filenames_array = np.array(all_filenames)

    # Сохраняем
    print(f"\n💾 Сохранение в {output_path}...")
    np.savez_compressed(
        output_path,
        embeddings=embeddings_array,
        filenames=filenames_array
    )

    print(f"✅ Сохранено {len(embeddings_array)} эмбеддингов")
    print(f"   Размерность: {embeddings_array.shape}")

    return embeddings_array, filenames_array