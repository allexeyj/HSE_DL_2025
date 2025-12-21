#!/usr/bin/env python3
"""
Генерирует CSV файл со сплитом train/validation на основе IDs из кэша эмбеддингов.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_split_from_embeddings(
        embeddings_path: Path,
        output_path: Path,
        val_size: float = 0.2,
        random_seed: int = 42,
) -> None:
    """
    Генерирует split файл на основе model_ids из кэша эмбеддингов.

    Args:
        embeddings_path: Путь к .npz файлу с эмбеддингами
        output_path: Путь для сохранения CSV файла
        val_size: Доля валидационной выборки (0.2 = 20%)
        random_seed: Seed для воспроизводимости
    """

    print(f"📂 Загрузка эмбеддингов из {embeddings_path}...")
    data = np.load(embeddings_path)
    filenames = data["filenames"]

    # Извлекаем уникальные model_ids
    model_ids = set()
    for fname in filenames:
        model_id = fname.split('_')[0]  # Извлекаем ID из имени файла (строка вида "0525")
        model_ids.add(model_id)

    model_ids = sorted(list(model_ids))
    print(f"✅ Найдено {len(model_ids)} уникальных моделей")

    # Конвертируем в числа для сохранения в CSV (load_split_ids_from_csv ожидает числа)
    model_ids_numeric = [int(mid) for mid in model_ids]

    # Делим на train/val
    train_ids, val_ids = train_test_split(
        model_ids_numeric,
        test_size=val_size,
        random_state=random_seed,
        shuffle=True
    )

    print(f"📊 Разделение данных:")
    print(f"   • Train: {len(train_ids)} моделей ({len(train_ids) / len(model_ids_numeric) * 100:.1f}%)")
    print(f"   • Val: {len(val_ids)} моделей ({len(val_ids) / len(model_ids_numeric) * 100:.1f}%)")

    # Создаем DataFrame
    train_df = pd.DataFrame({
        'model_id': train_ids,
        'split': 'train'
    })

    val_df = pd.DataFrame({
        'model_id': val_ids,
        'split': 'validation'
    })

    # Объединяем и сортируем
    df = pd.concat([train_df, val_df], ignore_index=True)
    df = df.sort_values('model_id').reset_index(drop=True)

    # Сохраняем
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"\n✅ Split сохранен в {output_path}")
    print(f"   Всего записей: {len(df)}")

    # Показываем примеры
    print("\n📋 Примеры из файла:")
    print(df.head(10))

    # Статистика
    print("\n📈 Статистика:")
    print(df['split'].value_counts())


def main():
    parser = argparse.ArgumentParser(
        description="Генерирует train/validation split из кэша эмбеддингов"
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        default="embeddings/train_img_embeddings_stage1.npz",
        help="Путь к .npz файлу с эмбеддингами"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="splits/base_split.csv",
        help="Путь для сохранения CSV файла со сплитом"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Доля валидационной выборки (default: 0.2)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости (default: 42)"
    )

    args = parser.parse_args()

    if not args.embeddings.exists():
        print(f"❌ Файл эмбеддингов не найден: {args.embeddings}")
        print("   Сначала запустите extract_train_embeddings.py")
        exit(1)

    generate_split_from_embeddings(
        embeddings_path=args.embeddings,
        output_path=args.output,
        val_size=args.val_size,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()