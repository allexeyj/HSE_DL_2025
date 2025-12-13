import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import argparse


def generate_k_fold_splits(
        n_models: int = 525,
        n_folds: int = 5,
        seed: int = 42,
        output_dir: str = "../splits/k_folds"
) -> list[pd.DataFrame]:
    """
    Генерирует k-fold splits для model IDs от 0000 до 0524

    Args:
        n_models: Количество моделей
        n_folds: Количество фолдов
        seed: Random seed для воспроизводимости
        output_dir: Директория для сохранения файлов

    Returns:
        Список DataFrame'ов с splits
    """
    np.random.seed(seed)

    # Создаем список всех model_ids
    model_ids = [f"{i:04d}" for i in range(n_models)]
    model_indices = np.arange(n_models)

    # Перемешиваем индексы для рандомизации
    shuffled_indices = np.random.permutation(model_indices)

    # Создаем KFold splitter
    kf = KFold(n_splits=n_folds, shuffle=False)  # shuffle=False т.к. уже перемешали

    # Создаем директорию для сохранения
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    fold_dfs = []

    print(f"🔄 Генерация {n_folds}-fold кросс-валидации")
    print(f"   Всего моделей: {n_models}")
    print(f"   Seed: {seed}")
    print("=" * 60)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(shuffled_indices), 1):
        # Получаем реальные индексы после перемешивания
        train_indices = shuffled_indices[train_idx]
        val_indices = shuffled_indices[val_idx]

        # Создаем DataFrame для текущего fold
        data = []

        # Добавляем train samples
        for idx in train_indices:
            data.append({
                "model_id": model_ids[idx],
                "split": "train"
            })

        # Добавляем validation samples
        for idx in val_indices:
            data.append({
                "model_id": model_ids[idx],
                "split": "validation"
            })

        # Создаем и сортируем DataFrame
        df = pd.DataFrame(data)
        df = df.sort_values("model_id").reset_index(drop=True)

        # Сохраняем в файл
        fold_filename = output_path / f"fold_{fold_idx}_of_{n_folds}.csv"
        df.to_csv(fold_filename, index=False)

        fold_dfs.append(df)

        # Статистика
        n_train = len(train_indices)
        n_val = len(val_indices)

        print(f"\n📁 Fold {fold_idx}/{n_folds}:")
        print(f"   Train: {n_train} samples ({n_train / n_models * 100:.1f}%)")
        print(f"   Val: {n_val} samples ({n_val / n_models * 100:.1f}%)")
        print(f"   Saved to: {fold_filename}")

        # Примеры ID для проверки
        train_examples = [model_ids[i] for i in train_indices[:3]]
        val_examples = [model_ids[i] for i in val_indices[:3]]
        print(f"   Train examples: {train_examples}")
        print(f"   Val examples: {val_examples}")

    print("\n" + "=" * 60)
    print(f"✅ Все {n_folds} fold'ов сохранены в {output_path}")

    # Проверка на пересечения (не должно быть)
    print("\n🔍 Проверка на корректность splits:")
    all_val_sets = []
    for fold_idx, df in enumerate(fold_dfs, 1):
        val_ids = set(df[df["split"] == "validation"]["model_id"].tolist())
        all_val_sets.append(val_ids)

    # Проверяем что каждая модель попадает в validation ровно один раз
    all_val_ids = []
    for val_set in all_val_sets:
        all_val_ids.extend(val_set)

    unique_val_ids = set(all_val_ids)
    if len(unique_val_ids) == n_models and len(all_val_ids) == n_models:
        print("   ✅ Каждая модель попадает в validation ровно один раз")
    else:
        print(f"   ⚠️ Проблема: уникальных val IDs: {len(unique_val_ids)}, всего val записей: {len(all_val_ids)}")

    # Проверяем что нет пересечений между val sets разных фолдов
    overlaps_found = False
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            overlap = all_val_sets[i] & all_val_sets[j]
            if overlap:
                print(f"   ⚠️ Пересечение между fold {i + 1} и fold {j + 1}: {len(overlap)} моделей")
                overlaps_found = True

    if not overlaps_found:
        print("   ✅ Нет пересечений между validation sets разных fold'ов")

    return fold_dfs


def main():
    parser = argparse.ArgumentParser(
        description="Генерация k-fold splits для обучения",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--n_models",
        type=int,
        default=525,
        help="Количество моделей в датасете"
    )

    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Количество fold'ов для кросс-валидации"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed для воспроизводимости"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../splits/k_folds",
        help="Директория для сохранения split файлов"
    )

    args = parser.parse_args()

    generate_k_fold_splits(
        n_models=args.n_models,
        n_folds=args.n_folds,
        seed=args.seed,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()