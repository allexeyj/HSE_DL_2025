#!/usr/bin/env python3
"""Скрипт для извлечения эмбеддингов изображений для первого этапа обучения"""

from pathlib import Path
from cad_retrieval_utils.extract_embeddings import extract_and_save_embeddings


def main():
    # Пути к данным
    TRAIN_BASE = Path('/kaggle/input/final-train-dataset/train')
    train_images_dir = TRAIN_BASE / 'images'
    captions_path = TRAIN_BASE / 'captions.json'

    # Путь для сохранения
    output_dir = Path('embeddings')
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'train_img_embeddings_stage1.npz'

    # Извлекаем эмбеддинги
    extract_and_save_embeddings(
        train_images_dir=train_images_dir,
        captions_path=captions_path,
        output_path=output_path,
        model_name="eva_large_patch14_336.in22k_ft_in22k_in1k",
        batch_size=256,
        device="cuda"
    )

    print(f"\n✅ Готово! Эмбеддинги сохранены в {output_path}")


if __name__ == "__main__":
    main()